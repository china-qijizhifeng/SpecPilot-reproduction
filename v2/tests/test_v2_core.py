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
