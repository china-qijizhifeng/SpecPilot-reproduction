#!/usr/bin/env python3
"""Isolated 2xH100 Phase 4 Sealed Final verifier (T3: Unseen Requests + 4096-token Token Forcing).

Runs ONLY inside a dedicated 2xH100 NorthJob. Never modifies controller/, eval/sealed/, or ledger/.
Uses strictly UNSEEN requests (indices 6..11 per suite = 30 requests, verified disjoint from dev 0..5):
  1. Two physical GPU-swap paired rounds (GPU 0 <-> GPU 1, max_new_tokens=2048) scored with
     protected `controller.stats.stratified_geomean_ci`.
  2. Long-output generation (5 unseen requests, 1/suite, max_new_tokens=4096) on both stock and candidate.
  3. FP32 LM-head teacher-forced Token Forcing on BOTH the 30 unseen (2048-cap) outputs AND
     the 5 unseen long (4096-cap) outputs, checking rate rule (<= stock + 0.5pp @ 95% CI) and
     event rule (margin > 2*delta = 0.50, binomial test).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller.http_transport import sha_ids
from controller.stats import binom_one_sided, stratified_geomean_ci
from controller.token_force import _wilson_low
from v2.controller_verify_2gpu import (
    STOCK_ENGINE,
    _gen_one,
    _score_outputs,
    _start_arm,
    _start_fp32_scorer,
    _wilson_high,
)
from v2.phase0.measure_stock import _free_port, _stop_server, _wait_health


def _load_unseen_prompts(start_idx: int = 6, per_suite: int = 6) -> tuple[list[dict], set[str]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/gpfs/models/huggingface.co/Qwen/Qwen3.8-27B", local_files_only=True)
    dev_dir = ROOT / "assets/upstream/dev"
    suites = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
    seen_hashes: set[str] = set()
    unseen: list[dict] = []

    for suite in suites:
        items = [json.loads(line) for line in (dev_dir / f"{suite}.jsonl").read_text().splitlines() if line.strip()]
        for idx, item in enumerate(items[: start_idx + per_suite]):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": item["turns"][0]}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True,
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            ids = list(ids)
            h = sha_ids(ids)
            if idx < start_idx:
                seen_hashes.add(h)
            else:
                if h in seen_hashes:
                    raise ValueError(f"unseen prompt {item['id']} collides with dev seen set!")
                unseen.append({"id": item["id"], "suite": suite, "input_ids": ids, "input_sha256": h})
    return unseen, seen_hashes


def _judge_tf(stock_tf: dict, cand_tf: dict) -> dict:
    cand_lo = _wilson_low(cand_tf["mismatches"], cand_tf["total_tokens"], z=1.959964)
    stock_hi = _wilson_high(stock_tf["mismatches"], stock_tf["total_tokens"], z=1.959964)
    rate_pass = (cand_lo - stock_hi) <= 0.005
    stock_event_hi = _wilson_high(stock_tf["large_margin_events"], stock_tf["total_tokens"], z=1.959964)
    event_p = binom_one_sided(cand_tf["large_margin_events"], cand_tf["total_tokens"], max(stock_event_hi, 1e-6))
    event_pass = (cand_tf["large_margin_events"] <= stock_tf["large_margin_events"]) or (event_p >= 0.025)
    return {
        "stock": stock_tf,
        "candidate": cand_tf,
        "rate_pass": rate_pass,
        "event_p": event_p,
        "event_pass": event_pass,
        "pass": bool(rate_pass and event_pass),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-id", required=True)
    parser.add_argument("--cand-worktree", type=Path, required=True)
    parser.add_argument("--option-env", type=str, default="{}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-idx", type=int, default=6)
    parser.add_argument("--per-suite", type=int, default=6)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.002)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_env = json.loads(args.option_env)
    prompts, seen_hashes = _load_unseen_prompts(start_idx=args.start_idx, per_suite=args.per_suite)
    # 5 unseen long-output prompts (1 per suite, cap=4096)
    long_prompts = [prompts[i * args.per_suite] for i in range(5)]

    req_log_ratios: dict[str, list[float]] = {p["id"]: [] for p in prompts}
    r1_stock: list[dict] = []
    r1_cand: list[dict] = []
    long_stock: list[dict] = []
    long_cand: list[dict] = []
    all_identical = True

    for rnd in (1, 2):
        stock_gpu, cand_gpu = (0, 1) if rnd == 1 else (1, 0)
        p_s, nccl_s = _free_port(), _free_port()
        p_c, nccl_c = _free_port(), _free_port()
        print(f"[sealed_final] Round {rnd}: stock=GPU{stock_gpu}, cand=GPU{cand_gpu}", flush=True)
        proc_s = _start_arm(stock_gpu, STOCK_ENGINE, p_s, nccl_s, 8, {}, args.out_dir / f"r{rnd}_stock.log")
        proc_c = _start_arm(cand_gpu, args.cand_worktree, p_c, nccl_c, 8, cand_env, args.out_dir / f"r{rnd}_cand.log")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                ex.submit(_wait_health, p_s, 1200.0).result()
                ex.submit(_wait_health, p_c, 1200.0).result()

            for wp in prompts[:2]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_s, wp, 32)
                    fc = ex.submit(_gen_one, p_c, wp, 32)
                    fs.result(); fc.result()

            for p in prompts:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_s, p, 2048)
                    fc = ex.submit(_gen_one, p_c, p, 2048)
                    rs, rc = fs.result(), fc.result()
                req_log_ratios[p["id"]].append(math.log(rc["tok_per_s"] / max(rs["tok_per_s"], 1e-6)))
                if rs["output_ids"] != rc["output_ids"]:
                    all_identical = False
                if rnd == 1:
                    r1_stock.append(rs)
                    r1_cand.append(rc)

            # On Round 1, also generate 4096-cap long outputs for Phase 4 long-output forcing
            if rnd == 1:
                print("[sealed_final] Generating 5 unseen 4096-cap long outputs...", flush=True)
                for lp in long_prompts:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        fs = ex.submit(_gen_one, p_s, lp, 4096)
                        fc = ex.submit(_gen_one, p_c, lp, 4096)
                        long_stock.append(fs.result())
                        long_cand.append(fc.result())
        finally:
            _stop_server(proc_s)
            _stop_server(proc_c)

    groups: dict[str, list[float]] = {}
    for p in prompts:
        groups.setdefault(p["suite"], []).append(sum(req_log_ratios[p["id"]]) / 2.0)
    ci = stratified_geomean_ci(groups, iters=10000, seed=0)
    print(f"[sealed_final] Unseen 30-req 2-GPU swap CI: geomean={ci['geomean']:.5f} [{ci['lower']:.5f}, {ci['upper']:.5f}]", flush=True)

    # FP32 LM-head teacher-forced Token Forcing on BOTH 2048 unseen AND 4096 long outputs
    print("[sealed_final] Starting FP32 LM-head scorer on GPU 0...", flush=True)
    p_sc, nccl_sc = _free_port(), _free_port()
    proc_sc = _start_fp32_scorer(0, p_sc, nccl_sc, args.out_dir / "fp32_scorer.log")
    try:
        _wait_health(p_sc, 1200.0)
        tf_2048 = _judge_tf(
            _score_outputs(p_sc, prompts, r1_stock, delta=args.delta),
            _score_outputs(p_sc, prompts, r1_cand, delta=args.delta),
        )
        tf_4096 = _judge_tf(
            _score_outputs(p_sc, long_prompts, long_stock, delta=args.delta),
            _score_outputs(p_sc, long_prompts, long_cand, delta=args.delta),
        )
    finally:
        _stop_server(proc_sc)

    min_gain = (0.01 + 2.0 * args.noise) if all_identical else max(0.01 + 2.0 * args.noise, 0.03)
    perf_pass = (ci["lower"] - 1.0) > min_gain
    sealed_passed = bool(perf_pass and tf_2048["pass"] and tf_4096["pass"])

    s_tps = sum(r["completion_tokens"] for r in r1_stock) / max(sum(r["latency_s"] for r in r1_stock), 1e-6)
    c_tps = sum(r["completion_tokens"] for r in r1_cand) / max(sum(r["latency_s"] for r in r1_cand), 1e-6)
    s_tau = sum(r["completion_tokens"] for r in r1_stock) / max(sum(r["verify_ct"] for r in r1_stock), 1)
    c_tau = sum(r["completion_tokens"] for r in r1_cand) / max(sum(r["verify_ct"] for r in r1_cand), 1)

    verdict = {
        "protocol": "v2-phase4-sealed-final",
        "cand_id": args.cand_id,
        "cand_worktree": str(args.cand_worktree),
        "option_env": cand_env,
        "unseen_requests": len(prompts),
        "disjoint_from_dev_count": len(seen_hashes),
        "performance": {
            "geomean_gain": round(ci["geomean"], 5),
            "ci95_lower": round(ci["lower"], 5),
            "ci95_upper": round(ci["upper"], 5),
            "perf_pass": perf_pass,
            "stock_tps": round(s_tps, 2),
            "cand_tps": round(c_tps, 2),
            "stock_tau": round(s_tau, 4),
            "cand_tau": round(c_tau, 4),
        },
        "token_forcing_2048": tf_2048,
        "token_forcing_4096": tf_4096,
        "sealed_final_passed": sealed_passed,
        "timestamp_unix": time.time(),
    }
    (args.out_dir / "sealed_verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    return 0 if sealed_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
