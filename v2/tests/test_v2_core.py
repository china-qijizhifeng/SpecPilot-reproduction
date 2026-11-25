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

from v2.phase0.floors import H100_SXM5, compute_ceiling_report, compute_class_workloads, format_master_brief
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


if __name__ == "__main__":
    unittest.main()
