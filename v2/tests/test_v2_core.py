"""CPU unit tests for SpecPilot v2 core infrastructure (sandbox, worktree, Phase 0 floors, NorthJob adapter)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.memory.distill import distill_entry, verify_entry
from v2.memory.store import (
    build_index,
    enrich_positive_entry,
    format_index_markdown,
    match_context,
    record_negative_entry,
    show_entry,
    update_entry_status,
)
from v2.phase0.floors import (
    DEVICE_PRESETS,
    H100_SXM5,
    compute_ceiling_report,
    compute_class_workloads,
    format_master_brief,
    get_device_spec,
)
from v2.phase0.tau_headroom import analyze_tau_headroom, best_first_tree_acceptance, oracle_step_acceptance
from v2.sandbox.bench_1gpu import stratify_easy_hard
from v2.sandbox.northjob_pool import attach_model_pvc, build_create_args
from v2.sandbox.pool_worker import STOP_EVENT, run_slot_loop
from v2.sandbox.run_gpu import pool_status, submit_and_wait
from v2.sandbox.worktree import create_worktree, export_patch


class TestPhase0Floors(unittest.TestCase):
    def test_qwen38_27b_weight_bytes_match_paper(self):
        """Paper Table A6: Qwen3.8-27B BF16 decoder linears + LM head ~= 51.1 GB."""
        target_cfg = json.loads(Path("/gpfs/models/huggingface.co/Qwen/Qwen3.8-27B/config.json").read_text())
        draft_cfg = json.loads(Path("/gpfs/users/jixinpeng/models/z-lab/Qwen3.8-27B-DFlash2/config.json").read_text())
        workloads = compute_class_workloads(target_cfg=target_cfg, draft_cfg=draft_cfg, verify_rows=8, draft_block=8)
        linear_gb = workloads["target_gemm"]["bytes"] / 1e9
        lm_head_gb = workloads["lm_head"]["bytes"] / 1e9
        total_gb = linear_gb + lm_head_gb
        # Table A6 reports 51.1 GB total BF16 weights for Qwen3.8-27B (linears + LM head)
        self.assertGreater(total_gb, 50.0)
        self.assertLess(total_gb, 52.5)
        self.assertAlmostEqual(lm_head_gb, 2.5428, places=2)

    def test_three_factor_ceiling_and_brief(self):
        report = compute_ceiling_report(
            measured_step_ms=25.0,
            measured_class_ms={
                "target_gemm": 18.0,
                "lm_head": 0.9,
                "gdn_recurrent": 1.2,
                "attention": 0.4,
                "drafter": 1.8,
            },
            measured_tau=4.65,
            tau_best=6.42,
            t_ar_ms=20.0,
        )
        f = report["factors"]
        self.assertAlmostEqual(f["tau_factor"], 6.42 / 4.65, places=3)
        self.assertAlmostEqual(f["ceiling"], f["tau_factor"] * f["kernel_factor"] * f["overhead_factor"], places=2)
        self.assertEqual(report["dominant_factor"], "tau_factor")
        brief = format_master_brief(report)
        self.assertIn("Phase 0 Ceiling Brief", brief)
        self.assertIn("target_gemm", brief)
        self.assertIn("System Overhead / Overlap Gap", brief)

    def test_tau_headroom_exact_replay_and_monotonic_budget(self):
        # Depth 0..3: target matches rank 0 at d=0, rank 1 at d=1, rank 0 at d=2, miss at d=3
        probs = [
            [0.8, 0.15, 0.05],
            [0.55, 0.40, 0.05],
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
        ]
        ranks = [0, 1, 0, -1]
        self.assertEqual(oracle_step_acceptance(ranks), 4)
        a2 = best_first_tree_acceptance(probs, ranks, budget=2)
        a8 = best_first_tree_acceptance(probs, ranks, budget=8)
        a32 = best_first_tree_acceptance(probs, ranks, budget=32)
        self.assertLessEqual(a2, a8)
        self.assertLessEqual(a8, a32)
        self.assertEqual(a32, 4)

        res = analyze_tau_headroom(
            steps=[{"accepted_tokens": 2, "pos_top_probs": probs, "pos_match_ranks": ranks}],
            budgets=(4, 8, 32),
            lookahead_agreement_rate=0.97,
        )
        self.assertGreaterEqual(res["tau_best"], res["tau_measured"])
        self.assertLessEqual(res["tau_best"], res["tau_oracle"])
        self.assertIsNotNone(res["estimated_lookahead_loss_tokens"])

        # Integrate into compute_ceiling_report with lookahead & overhead gap
        report = compute_ceiling_report(
            measured_step_ms=36.0,
            measured_class_ms={
                "target_gemm": 15.4,
                "lookahead_gemm": 11.1,
                "lm_head": 1.4,
                "gdn_recurrent": 1.2,
                "attention": 0.8,
                "drafter": 2.4,
            },
            measured_tau=res["tau_measured"],
            tau_best=res["tau_best"],
            verify_rows=32,
            draft_block=16,
            weight_bytes_per_elem=1.5,
            lookahead_weight_bytes_per_elem=0.75,
            lookahead_rows=16,
            lookahead_trigger_rate=1.0,
            drafter_passes_per_step=2.0,
            lm_head_passes_per_step=2.0,
            tau_headroom=res,
        )
        self.assertIn("lookahead_gemm", report["classes"])
        self.assertAlmostEqual(report["measured"]["unattributed_overhead_ms"], 3.7, places=2)
        brief = format_master_brief(report)
        self.assertIn("Acceptance Headroom (Exact Replay)", brief)
        self.assertIn("lookahead_gemm", brief)

    def test_floors_generic_pure_attention_model_and_device_presets(self):
        # Standard dense pure-attention Transformer (no GDN / layer_types fields)
        llama_like = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        draft_like = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        b200 = get_device_spec("B200-SXM6-192GB")
        ascend = get_device_spec("Ascend-910B2C")
        self.assertIn("H100-SXM5-80GB", DEVICE_PRESETS)
        w = compute_class_workloads(target_cfg=llama_like, draft_cfg=draft_like, verify_rows=16, draft_block=8)
        self.assertEqual(w["gdn_recurrent"]["bytes"], 0.0)
        self.assertGreater(w["target_gemm"]["bytes"], 10e9)
        self.assertLess(b200.floor_ms(w["target_gemm"]["flops"], w["target_gemm"]["bytes"]),
                        ascend.floor_ms(w["target_gemm"]["flops"], w["target_gemm"]["bytes"]))

    def test_floors_qwen35_35b_a3b_moe_and_mtp(self):
        cfg_path = Path("/gpfs/models/huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/config.json")
        if not cfg_path.is_file():
            self.skipTest("Qwen3.5-35B-A3B-FP8 config not present")
        target_cfg = json.loads(cfg_path.read_text())
        w = compute_class_workloads(
            target_cfg=target_cfg,
            draft_cfg={},  # built-in MTP
            verify_rows=4,
            draft_block=4,
            weight_bytes_per_elem=1.0,  # FP8
            draft_weight_bytes_per_elem=1.0,
        )
        # 4 tokens hitting ~30 distinct experts out of 256 across 40 layers in FP8 -> ~5-7 GB active bytes
        self.assertGreater(w["target_gemm"]["bytes"] / 1e9, 3.0)
        self.assertLess(w["target_gemm"]["bytes"] / 1e9, 12.0)
        self.assertGreater(w["gdn_recurrent"]["bytes"], 0.0)
        self.assertGreater(w["drafter"]["bytes"], 0.0)


class TestStratifiedBenchAndProtocol(unittest.TestCase):
    def test_stratify_easy_hard_detects_hard_regression(self):
        per_req = [
            {"id": "h1", "base_tau": 2.5, "cand_tau": 2.4, "ratio": 0.96},
            {"id": "h2", "base_tau": 3.0, "cand_tau": 2.9, "ratio": 0.98},
            {"id": "e1", "base_tau": 6.0, "cand_tau": 7.2, "ratio": 1.06},
            {"id": "e2", "base_tau": 7.0, "cand_tau": 8.5, "ratio": 1.08},
        ]
        st = stratify_easy_hard(per_req)
        self.assertTrue(st["hard_regression"])
        self.assertGreater(st["easy"]["geomean_gain_x"], 1.05)
        self.assertLess(st["hard"]["geomean_gain_x"], 0.99)

        per_req_good = [
            {"id": "h1", "base_tau": 2.5, "cand_tau": 3.0, "ratio": 1.04},
            {"id": "h2", "base_tau": 3.0, "cand_tau": 3.5, "ratio": 1.03},
            {"id": "e1", "base_tau": 6.0, "cand_tau": 7.2, "ratio": 1.06},
            {"id": "e2", "base_tau": 7.0, "cand_tau": 8.5, "ratio": 1.08},
        ]
        st_good = stratify_easy_hard(per_req_good)
        self.assertFalse(st_good["hard_regression"])

    def test_master_prompt_has_plateau_escape_and_zero_hints(self):
        prompt = (ROOT / "v2" / "agents" / "master_prompt.md").read_text()
        for required in (
            "Priority 0",
            "Phase 0 Refresh",
            "Stratified Generalization Check",
            "Theory-vs-Implementation Triage",
            "Plateau Escape & Co-Design Protocol",
            "Assumption & Coupling Audit",
        ):
            self.assertIn(required, prompt)
        # Must remain 100% generic (no paper/solution hints)
        for forbidden in ("BM=32", "Huffman", "W12", "W6", "2.13x", "DFloat11"):
            self.assertNotIn(forbidden, prompt)

    def test_direction_prompt_is_model_and_hardware_agnostic(self):
        dprompt = (ROOT / "v2" / "agents" / "direction_prompt.md").read_text()
        for required in ("stratified.easy", "stratified.hard", "hard_regression", "theory_falsified", "impl_underoptimized", "reused_from"):
            self.assertIn(required, dprompt)
        for hardcoded in ("Qwen3.8", "DFlash2", "H100", "SGLang v0.5.20"):
            self.assertNotIn(hardcoded, dprompt)


class TestDualLayerMemoryStore(unittest.TestCase):
    def _ctx(self, bs: int = 1, sm: str = "sm_90") -> dict:
        return {
            "model": {"target": "Qwen3.8-27B", "drafter": "DFlash2", "dtype": "bf16"},
            "hardware": {"gpu": "H100-SXM5-80GB", "sm": sm},
            "workload": {"batch_size": bs, "max_new_tokens": 2048, "decoding": "greedy_temp0"},
            "dataset": {"suite": "5-suite-dev-en", "stock_tau": 4.65},
        }

    def test_context_match_exact_and_partial_drift(self):
        c1 = self._ctx(bs=1, sm="sm_90")
        self.assertEqual(match_context(c1, c1)["match_level"], "exact")
        c2 = self._ctx(bs=16, sm="sm_120")
        m = match_context(c1, c2)
        self.assertEqual(m["match_level"], "partial")
        self.assertEqual(m["matched_dims"], 2)
        self.assertTrue(any("workload.batch_size: 1 -> 16" in d for d in m["diffs"]))
        self.assertTrue(any("hardware.sm: sm_90 -> sm_120" in d for d in m["diffs"]))

    def test_two_stage_index_show_provenance_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stock = root / "stock"
            cand = root / "cand"
            (stock / "python").mkdir(parents=True)
            (stock / "python" / "mod.py").write_text("x = 1\n")
            (cand / "python").mkdir(parents=True)
            (cand / "python" / "mod.py").write_text("x = 2\n")

            mem_base = root / "memory"
            distill_entry(
                entry_id="pos_r1",
                round_num=1,
                base_dir=stock,
                source_dir=cand,
                requires=[],
                options={"SP_OPT_DEMO": "1"},
                summary="Demo positive speedup",
                evidence={"dev_gain": 1.25},
                memory_root=mem_base / "entries",
            )
            enrich_positive_entry(
                entry_id="pos_r1",
                context=self._ctx(bs=1),
                bottleneck_tags=["hbm_bw_bound"],
                one_line_summary="Positive demo +25%",
                memory_base=mem_base,
            )
            # Byte-exact replay must remain intact after enrichment
            v = verify_entry("pos_r1", stock_dir=stock, memory_root=mem_base / "entries")
            self.assertTrue(v["byte_exact_match"])

            record_negative_entry(
                neg_id="neg_r1_impl",
                round_num=1,
                base_tree_sha=v["expected_tree_sha"],
                context=self._ctx(bs=1),
                bottleneck_tags=["hbm_bw_bound", "smem_bank_conflict"],
                hypothesis="Compressed kernel prototype",
                failure_class="impl_underoptimized",
                failed_precondition="SMEM LUT bank conflicts slowed decode by 40%",
                reopen_condition="Use register bit-parallel decode without SMEM LUT",
                one_line_summary="Compressed prototype slow on SMEM LUT (reopenable)",
                evidence={"tier2_ratio": 0.60},
                memory_base=mem_base,
            )

            # Stage 1: summary index with cross-workload query (bs=16) shows partial match
            rows = build_index(
                query_context=self._ctx(bs=16),
                bottleneck_filter=["hbm_bw_bound"],
                memory_base=mem_base,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["context_match"] == "partial" for r in rows))
            md = format_index_markdown(rows)
            self.assertIn("neg_r1_impl", md)
            self.assertIn("workload.batch_size: 1 -> 16", md)

            # Stage 2: expand single entry & log provenance
            prov = root / "provenance.jsonl"
            detail = show_entry(
                entry_id="neg_r1_impl",
                query_context=self._ctx(bs=16),
                provenance_log=prov,
                memory_base=mem_base,
            )
            self.assertEqual(detail["failure_class"], "impl_underoptimized")
            self.assertIn("register bit-parallel", detail["reopen_condition"])
            prov_lines = [json.loads(l) for l in prov.read_text().splitlines() if l.strip()]
            self.assertEqual(prov_lines[0]["reused_from"], "neg_r1_impl")

            # Maintainer lifecycle update preserves immutable evidence & removes from active index
            upd = update_entry_status(
                entry_id="neg_r1_impl",
                status="resolved",
                resolved_by="pos_r2",
                memory_base=mem_base,
            )
            self.assertEqual(upd["status"], "resolved")
            self.assertEqual(upd["resolved_by"], "pos_r2")
            self.assertEqual(upd["evidence"]["tier2_ratio"], 0.60)
            active_rows = build_index(query_context=self._ctx(bs=1), memory_base=mem_base)
            self.assertEqual([r["id"] for r in active_rows], ["pos_r1"])

    def test_memory_rejects_sealed_paths_and_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                record_negative_entry(
                    neg_id="bad_sealed",
                    round_num=1,
                    base_tree_sha="0" * 64,
                    context=self._ctx(),
                    bottleneck_tags=["sync_bubble"],
                    hypothesis="leaked eval/sealed/prompts.jsonl",
                    failure_class="theory_falsified",
                    failed_precondition="none",
                    reopen_condition="none",
                    one_line_summary="bad",
                    evidence={},
                    memory_base=Path(td),
                )


class TestSandboxPool(unittest.TestCase):
    def test_pool_worker_and_client_roundtrip(self):
        STOP_EVENT.clear()
        with tempfile.TemporaryDirectory() as td:
            pool_dir = Path(td)
            t = threading.Thread(target=run_slot_loop, args=(pool_dir, 0, 0.05, True), daemon=True)
            t.start()
            try:
                res = submit_and_wait(pool_dir, slot=0, cmd="echo hello_v2 && exit 0", cwd=td, timeout_s=10.0)
                self.assertEqual(res["exit_code"], 0)
                self.assertIn("hello_v2", res["stdout_tail"])
                self.assertFalse(res["timed_out"])

                # Non-zero exit code propagation
                res_fail = submit_and_wait(pool_dir, slot=0, cmd="echo boom && exit 7", cwd=td, timeout_s=10.0)
                self.assertEqual(res_fail["exit_code"], 7)

                # Timeout kills process cleanly
                res_to = submit_and_wait(pool_dir, slot=0, cmd="sleep 10", cwd=td, timeout_s=0.6)
                self.assertTrue(res_to["timed_out"])
                self.assertEqual(res_to["exit_code"], 124)

                st = pool_status(pool_dir)
                self.assertTrue(st["slots"]["gpu0"]["alive"])
            finally:
                STOP_EVENT.set()
                t.join(timeout=5)


class TestWorktreeAndNorthJob(unittest.TestCase):
    def test_worktree_isolation_and_patch(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            (base / "python").mkdir(parents=True)
            (base / "python" / "demo.py").write_text("x = 1\n")
            wt = Path(td) / "wt0"
            info = create_worktree(base, wt)
            self.assertEqual(len(info["base_tree_sha"]), 64)

            (wt / "python" / "demo.py").write_text("x = 2\n")
            patch_path = Path(td) / "cand.patch"
            exp = export_patch(wt, patch_path)
            self.assertNotEqual(exp["tree_sha"], info["base_tree_sha"])
            self.assertIn("-x = 1", patch_path.read_text())
            self.assertIn("+x = 2", patch_path.read_text())

    def test_northjob_model_pvc_attachment(self):
        args = build_create_args(job_name="sp-v2-test", gpus=8, mem_gib=512, user_cmd="echo ok")
        self.assertIn("--gpu-per-node", args)
        self.assertEqual(args[-1], "echo ok")
        fake_tpl = {
            "spec": {
                "tasks": [
                    {
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}],
                            }
                        },
                    }
                ]
            }
        }
        out = attach_model_pvc(fake_tpl, gpus=8, deadline=43200)
        mounts = out["spec"]["tasks"][0]["template"]["spec"]["containers"][0]["volumeMounts"]
        self.assertTrue(any(m["mountPath"] == "/gpfs/models" and m["readOnly"] for m in mounts))
        self.assertEqual(out["spec"]["maxRetry"], 0)


class TestTempGtZeroTargetOnly(unittest.TestCase):
    def test_tau_headroom_expected_acceptance_and_dedup(self):
        from v2.phase0.tau_headroom import (
            analyze_tau_headroom,
            best_first_tree_acceptance,
            best_first_tree_expected_acceptance,
        )

        # Depth 0: two children with target probs 0.4, 0.3; Depth 1: two children with 0.5, 0.2
        pos_draft = [[0.6, 0.4], [0.7, 0.3]]
        pos_target = [[0.4, 0.3], [0.5, 0.2]]
        # Budget 3 = root + 2 depth-0 children (draft probs 0.6, 0.4; next best depth-1 is 0.6*0.7=0.42 > 0.4!)
        # Wait: 0.6 > 0.42 > 0.4, so budget 3 selects: root, d0_r0 (0.4), d1_r0 under d0_r0 (0.4*0.5=0.2)
        e3 = best_first_tree_expected_acceptance(pos_draft, pos_target, budget=3)
        self.assertAlmostEqual(e3, 1.0 + 0.4 + 0.2, places=6)

        # Duplicate sibling dedup: second child has same token_id=99 -> must contribute 0!
        e_dedup = best_first_tree_expected_acceptance(
            [[0.6, 0.4]], [[0.4, 0.4]], budget=3, pos_token_ids=[[99, 99]]
        )
        self.assertAlmostEqual(e_dedup, 1.0 + 0.4, places=6)

        # Reduces identically to greedy when target probs are 0/1
        greedy_exact = best_first_tree_acceptance(pos_draft, [0, 1], budget=5)
        stoch_as_greedy = best_first_tree_expected_acceptance(pos_draft, [[1.0, 0.0], [0.0, 1.0]], budget=5)
        self.assertAlmostEqual(stoch_as_greedy, float(greedy_exact), places=6)

        # analyze_tau_headroom with temperature > 0
        steps = [{"accepted_tokens": 2, "pos_top_probs": pos_draft, "pos_target_probs": pos_target}]
        res = analyze_tau_headroom(steps, budgets=(3, 8), temperature=0.7)
        self.assertEqual(res["sampling_mode"], "stochastic_target_only")
        self.assertAlmostEqual(res["tau_best_by_budget"]["3"], 1.6, places=4)
        self.assertGreaterEqual(res["tau_best_by_budget"]["8"], res["tau_best_by_budget"]["3"])

    def test_t1s_positive_and_fault_injections(self):
        import math
        import random
        from v2.t1s_verify import (
            aggregate_t1s_metrics,
            compute_step_metrics,
            evaluate_t1s_verdict,
            load_stock_cache,
            normalize_top_logprobs,
            save_stock_cache,
            stock_cache_key,
        )

        rng = random.Random(123)
        temp = 0.7
        # Raw logprobs for 6 tokens at T=1.0
        base_raw = [(-0.8, 10), (-1.3, 11), (-1.9, 12), (-2.4, 13), (-3.0, 14), (-3.8, 15)]
        p_true = normalize_top_logprobs(base_raw, temperature=temp)
        tids = list(p_true.keys())
        probs = [p_true[t] for t in tids]

        def _sample(ps):
            u = rng.random()
            c = 0.0
            for tid, p in zip(tids, ps):
                c += p
                if u <= c:
                    return tid
            return tids[-1]

        N = 1500
        # 1. Stock baseline (tiny BF16 jitter ~0.002 logit noise)
        stock_steps = []
        for _ in range(N):
            y = _sample(probs)
            noisy = [(lp + rng.gauss(0.0, 0.003), tid) for lp, tid in base_raw]
            stock_steps.append(compute_step_metrics(
                ref_top_entries=base_raw, cand_top_entries=noisy, emitted_token=y, temperature=temp, rng=rng
            ))
        stock_m = aggregate_t1s_metrics(stock_steps)

        # 2. Lossless Candidate (same tiny BF16 jitter, genuinely sampled from p_true)
        cand_ok_steps = []
        for _ in range(N):
            y = _sample(probs)
            noisy = [(lp + rng.gauss(0.0, 0.003), tid) for lp, tid in base_raw]
            cand_ok_steps.append(compute_step_metrics(
                ref_top_entries=base_raw, cand_top_entries=noisy, emitted_token=y, temperature=temp, rng=rng
            ))
        cand_ok_m = aggregate_t1s_metrics(cand_ok_steps)
        v_ok = evaluate_t1s_verdict(stock_m, cand_ok_m, {"actually_stochastic": True})
        self.assertTrue(v_ok["t1s_pass"], f"Expected lossless candidate PASS, got {v_ok}")

        # 3. Fault Injection 1: Silent greedy collapse (always emits top-1 token 10)
        greedy_steps = [
            compute_step_metrics(ref_top_entries=base_raw, cand_top_entries=base_raw, emitted_token=10, temperature=temp, rng=rng)
            for _ in range(N)
        ]
        greedy_m = aggregate_t1s_metrics(greedy_steps)
        v_greedy = evaluate_t1s_verdict(stock_m, greedy_m, {"actually_stochastic": False})
        self.assertFalse(v_greedy["t1s_pass"])
        self.assertFalse(v_greedy["m5_routing_pass"])
        self.assertLess(greedy_m["z_nll"], -10.0)

        # 4. Fault Injection 2: Temperature shift (actually sampled at T=0.45 pretending to be T=0.7)
        p_cold = normalize_top_logprobs(base_raw, temperature=0.45)
        cold_probs = [p_cold[t] for t in tids]
        cold_steps = [
            compute_step_metrics(ref_top_entries=base_raw, cand_top_entries=base_raw, emitted_token=_sample(cold_probs), temperature=temp, rng=rng)
            for _ in range(N)
        ]
        cold_m = aggregate_t1s_metrics(cold_steps)
        v_cold = evaluate_t1s_verdict(stock_m, cold_m, {"actually_stochastic": True})
        self.assertFalse(v_cold["t1s_pass"])
        self.assertFalse(v_cold["m5_routing_pass"])

        # 5. Fault Injection 3: Corrupted Target Verify logits (operator/state noise sigma=0.3)
        corrupt_steps = []
        for _ in range(N):
            y = _sample(probs)
            bad = [(lp + rng.gauss(0.0, 0.30), tid) for lp, tid in base_raw]
            corrupt_steps.append(compute_step_metrics(
                ref_top_entries=base_raw, cand_top_entries=bad, emitted_token=y, temperature=temp, rng=rng
            ))
        corrupt_m = aggregate_t1s_metrics(corrupt_steps)
        v_corrupt = evaluate_t1s_verdict(stock_m, corrupt_m, {"actually_stochastic": True})
        self.assertFalse(v_corrupt["t1s_pass"])
        self.assertFalse(v_corrupt["m1_events_pass"])
        self.assertFalse(v_corrupt["m2_mean_tvd_pass"])

        # 6. Stock cache round-trip
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            k = stock_cache_key(
                model_path="Qwen3.8-27B", stock_worktree="/stock", prompt_ids=["a", "b"],
                temperature=0.7, top_p=1.0, top_k=-1, max_new_tokens=512,
            )
            self.assertIsNone(load_stock_cache(cdir, k))
            save_stock_cache(cdir, k, {"metrics": stock_m})
            loaded = load_stock_cache(cdir, k)
            self.assertEqual(loaded["metrics"]["mean_tvd"], stock_m["mean_tvd"])

    def test_cross_temperature_sampling_transfer_ranking(self):
        from v2.memory.store import assess_sampling_transfer, assess_transfer

        q_t07 = {"workload": {"batch_size": 1, "temperature": 0.7, "decoding": "stochastic_target_only"}}
        q_t0 = {"workload": {"batch_size": 1, "temperature": 0.0, "decoding": "greedy_temp0"}}

        self.assertEqual(assess_sampling_transfer("temp_agnostic", q_t07)["transfer_status"], "direct")
        self.assertEqual(assess_sampling_transfer("adapt_sampler_target_only", q_t07)["transfer_status"], "adapt_required")
        self.assertEqual(assess_sampling_transfer("greedy_only", q_t07)["transfer_status"], "incompatible")
        self.assertEqual(assess_sampling_transfer("greedy_only", q_t0)["transfer_status"], "direct")
        self.assertEqual(assess_sampling_transfer("adapt_sampler_target_only", q_t0)["transfer_status"], "direct")

        # 5-axis generalization: model / spec / GPU / batch / temp
        entry_tree = {
            "kind": "positive",
            "context": {
                "model": {"target": "Qwen3.8-27B", "arch_family": "hybrid_gdn_dense", "dtype": "bf16"},
                "spec": {"algo": "DFlash2"},
                "hardware": {"gpu": "H100-SXM5-80GB", "sm": "sm_90"},
                "workload": {"batch_size": 1, "temperature": 0.0},
            },
            "transfer_scope": {
                "code_replay_requires": ["model.target", "spec.algo", "hardware.sm"],
                "pattern_requires": {"model.arch_family": ["hybrid_gdn_dense", "dense"]},
                "pattern_excludes": {"model.arch_family": ["sparse_moe"]},
                "sampling_compatibility": "adapt_sampler_target_only",
                "batch_regime": "small_bs",
            },
        }
        # Same model/spec/GPU at temp=0.7 -> code_adapt (reuse fast tree kernel + swap Target-Only sampler!)
        self.assertEqual(
            assess_transfer(entry_tree, {
                "model": {"target": "Qwen3.8-27B", "arch_family": "hybrid_gdn_dense", "dtype": "bf16"},
                "spec": {"algo": "DFlash2"},
                "hardware": {"sm": "sm_90"},
                "workload": {"batch_size": 1, "temperature": 0.7},
            })["verdict"],
            "code_adapt",
        )
        # Different GPU (B200 sm_100) + different Dense model -> pattern_transfer (concept only, not raw patch)
        self.assertEqual(
            assess_transfer(entry_tree, {
                "model": {"target": "Llama-4-Dense", "arch_family": "dense", "dtype": "bf16"},
                "spec": {"algo": "EAGLE3"},
                "hardware": {"sm": "sm_100"},
                "workload": {"batch_size": 1, "temperature": 0.7},
            })["verdict"],
            "pattern_transfer",
        )
        # Sparse MoE -> blocked (prevents wide-tree negative transfer!)
        self.assertEqual(
            assess_transfer(entry_tree, {
                "model": {"target": "Qwen3.5-35B-A3B-FP8", "arch_family": "sparse_moe", "dtype": "fp8"},
                "spec": {"algo": "MTP"},
                "hardware": {"sm": "sm_90"},
                "workload": {"batch_size": 1, "temperature": 0.7},
            })["verdict"],
            "blocked",
        )
        # Large batch (bs=32) -> blocked (small_bs regime guard!)
        self.assertEqual(
            assess_transfer(entry_tree, {
                "model": {"target": "Qwen3.8-27B", "arch_family": "hybrid_gdn_dense", "dtype": "bf16"},
                "spec": {"algo": "DFlash2"},
                "hardware": {"sm": "sm_90"},
                "workload": {"batch_size": 32, "temperature": 0.0},
            })["verdict"],
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()
