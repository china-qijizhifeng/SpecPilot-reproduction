#!/usr/bin/env python3
"""Fast 1-GPU Tier-3 A/B benchmark for SpecPilot v2 Direction subagents.

Compares option OFF (stock baseline) vs option ON (candidate) on the same GPU slot
over 10 dev requests (2 per suite, max_new_tokens=512), reporting:
- tok/s, avg_tau, avg_step_ms for both arms
- candidate / baseline throughput ratio (gain_x)
- exact output match count (out of 10)
Note: 1-GPU serial A/B is ONLY for fast Tier-3 headroom screening (>= 3%).
Formal promotion always goes through the isolated 2xH100 swapped-GPU Controller.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.phase0.measure_stock import _free_port, _http_post_json, _load_dev_prompts, _stop_server, _wait_health


def _launch(engine_dir: Path, port: int, nccl_port: int, draft_tokens: int, extra_env: dict[str, str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(engine_dir)
    env["PORT"] = str(port)
    env["NCCL_PORT"] = str(nccl_port)
    env["DRAFT_TOKENS"] = str(draft_tokens)
    env.update({str(k): str(v) for k, v in extra_env.items()})
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(
        ["bash", "-lc", f'bash "{ROOT}/harness/sglang/serve.sh"'],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _run_arm(engine_dir: Path, prompts: list[dict], draft_tokens: int, extra_env: dict[str, str], max_new_tokens: int, log_path: Path) -> dict:
    port = _free_port()
    nccl_port = _free_port()
    proc = _launch(engine_dir, port, nccl_port, draft_tokens, extra_env, log_path)
    try:
        _wait_health(port, timeout_s=600.0)
        url = f"http://127.0.0.1:{port}/generate"
        for p in prompts[:2]:
            _http_post_json(url, {"input_ids": p["input_ids"], "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": 32}})

        rows = []
        tot_tok = 0
        tot_steps = 0
        tot_s = 0.0
        for p in prompts:
            t0 = time.perf_counter()
            resp = _http_post_json(
                url,
                {"input_ids": p["input_ids"], "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": max_new_tokens}},
            )
            dt = time.perf_counter() - t0
            meta = resp.get("meta_info", {})
            out_ids = resp.get("output_ids", [])
            comp = int(meta.get("completion_tokens", len(out_ids)))
            vct = int(meta.get("spec_verify_ct", 0))
            lat = float(meta.get("e2e_latency", dt))
            rows.append(
                {
                    "id": p["id"],
                    "suite": p["suite"],
                    "completion_tokens": comp,
                    "verify_ct": vct,
                    "e2e_s": round(lat, 4),
                    "tok_per_s": round(comp / max(lat, 1e-6), 2),
                    "tau": round(comp / vct, 4) if vct > 0 else None,
                    "output_ids": out_ids,
                }
            )
            tot_tok += comp
            tot_steps += vct
            tot_s += lat
        return {
            "rows": rows,
            "avg_tok_per_s": round(tot_tok / max(tot_s, 1e-6), 2),
            "avg_tau": round(tot_tok / tot_steps, 4) if tot_steps > 0 else None,
            "avg_step_ms": round(1000.0 * tot_s / tot_steps, 4) if tot_steps > 0 else None,
            "total_tokens": tot_tok,
            "total_steps": tot_steps,
            "total_s": round(tot_s, 4),
        }
    finally:
        _stop_server(proc)


def stratify_easy_hard(per_req: list[dict]) -> dict:
    """Split per-request results into Easy (high baseline tau) and Hard (low baseline tau) buckets.

    Zero extra GPU runtime; exposes dev-set regime overfitting before Sealed evaluation.
    """
    if not per_req:
        return {"easy": None, "hard": None, "hard_regression": False}

    def _key(r: dict) -> float:
        if r.get("base_tau") is not None:
            return float(r["base_tau"])
        return float(r.get("base_tps", 0.0))

    ordered = sorted(per_req, key=_key)
    mid = max(1, len(ordered) // 2)
    hard_rows = ordered[:mid]
    easy_rows = ordered[mid:] if len(ordered) > 1 else ordered

    def _agg(rows: list[dict]) -> dict:
        ratios = [max(float(r["ratio"]), 1e-6) for r in rows]
        gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        base_taus = [float(r["base_tau"]) for r in rows if r.get("base_tau") is not None]
        cand_taus = [float(r["cand_tau"]) for r in rows if r.get("cand_tau") is not None]
        return {
            "count": len(rows),
            "geomean_gain_x": round(gm, 4),
            "avg_base_tau": round(sum(base_taus) / len(base_taus), 4) if base_taus else None,
            "avg_cand_tau": round(sum(cand_taus) / len(cand_taus), 4) if cand_taus else None,
            "request_ids": [r["id"] for r in rows],
        }

    hard_agg = _agg(hard_rows)
    easy_agg = _agg(easy_rows)
    return {
        "easy": easy_agg,
        "hard": hard_agg,
        "hard_regression": hard_agg["geomean_gain_x"] < 0.99,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-option-env", type=str, default="{}", help="JSON dict of env vars for baseline arm")
    parser.add_argument("--option-env", type=str, default="{}", help="JSON dict of env vars for candidate ON arm")
    parser.add_argument("--base-draft-tokens", type=int, default=8)
    parser.add_argument("--cand-draft-tokens", type=int, default=8)
    parser.add_argument("--per-suite", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_dev_prompts(per_suite=args.per_suite)
    base_env = json.loads(args.base_option_env)
    cand_env = json.loads(args.option_env)

    print(f"[bench_1gpu] running BASELINE ({base_env})...", flush=True)
    base = _run_arm(args.worktree, prompts, args.base_draft_tokens, base_env, args.max_new_tokens, args.out_dir / "base_server.log")
    print(f"[bench_1gpu] BASELINE: {base['avg_tok_per_s']} tok/s, tau={base['avg_tau']}, step={base['avg_step_ms']} ms", flush=True)

    print(f"[bench_1gpu] running CANDIDATE (option ON: {cand_env}, draft={args.cand_draft_tokens})...", flush=True)
    cand = _run_arm(args.worktree, prompts, args.cand_draft_tokens, cand_env, args.max_new_tokens, args.out_dir / "cand_server.log")
    print(f"[bench_1gpu] CANDIDATE: {cand['avg_tok_per_s']} tok/s, tau={cand['avg_tau']}, step={cand['avg_step_ms']} ms", flush=True)

    exact_matches = 0
    per_req = []
    for br, cr in zip(base["rows"], cand["rows"]):
        same = br["output_ids"] == cr["output_ids"]
        if same:
            exact_matches += 1
        per_req.append(
            {
                "id": br["id"],
                "base_tps": br["tok_per_s"],
                "cand_tps": cr["tok_per_s"],
                "ratio": round(cr["tok_per_s"] / max(br["tok_per_s"], 1e-6), 4),
                "base_tau": br["tau"],
                "cand_tau": cr["tau"],
                "exact_match": same,
            }
        )

    gain_x = round(cand["avg_tok_per_s"] / max(base["avg_tok_per_s"], 1e-6), 4)
    stratified = stratify_easy_hard(per_req)
    summary = {
        "worktree": str(args.worktree),
        "option_env": cand_env,
        "base_draft_tokens": args.base_draft_tokens,
        "cand_draft_tokens": args.cand_draft_tokens,
        "baseline": {k: v for k, v in base.items() if k != "rows"},
        "candidate": {k: v for k, v in cand.items() if k != "rows"},
        "gain_x": gain_x,
        "stratified": stratified,
        "exact_matches": f"{exact_matches}/{len(prompts)}",
        "passes_3pct_tier3": gain_x >= 1.03,
        "per_request": per_req,
    }
    (args.out_dir / "tier3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
