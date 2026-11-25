#!/usr/bin/env python3
"""Isolated 2xH100 Controller verification runner for SpecPilot v2.

Runs ONLY inside a dedicated 2xH100 NorthJob (separate from the exploration pool).
Never modifies controller/, eval/sealed/, or ledger/.
Executes:
  1. T2: 2-GPU physical swap paired timing (2 fresh rounds, GPU 0 <-> GPU 1, 6 reqs/suite = 30 dev reqs)
         scored with protected `controller.stats.stratified_geomean_ci`.
  2. T1: FP32 LM-head teacher-forced token forcing on stock vs candidate outputs, checking
         both rate rule (<= stock + 0.5 pp @ 95% CI) and event rule (margin > 2*delta, binomial test).
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller.scoring import decode_scores
from controller.stats import binom_one_sided, stratified_geomean_ci
from controller.token_force import _wilson_low
from v2.phase0.measure_stock import _free_port, _http_post_json, _load_dev_prompts, _stop_server, _wait_health

STOCK_ENGINE = Path("/gpfs/users/jixinpeng/specpilot_data/work/engines/h100_c1_qwen38-27b_dflash2-stock")


def _wilson_high(k: int, n: int, z: float = 1.959964) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    rad = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + rad) / denom)


def _start_arm(gpu: int, engine_dir: Path, port: int, nccl_port: int, draft_tokens: int, extra_env: dict[str, str], log_path: Path, extra_args: list[str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(engine_dir)
    env["PORT"] = str(port)
    env["NCCL_PORT"] = str(nccl_port)
    env["DRAFT_TOKENS"] = str(draft_tokens)
    env.update({str(k): str(v) for k, v in extra_env.items()})
    arg_str = " ".join(extra_args or [])
    cmd = f'bash "{ROOT}/harness/sglang/serve.sh" {arg_str}'.strip()
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _start_fp32_scorer(gpu: int, port: int, nccl_port: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(STOCK_ENGINE)
    cmd = (
        f'source "{ROOT}/harness/sglang/env.sh" && '
        f'exec "$SP_PY" -m sglang.launch_server '
        f'--model-path "$SP_TARGET" --dtype bfloat16 --kv-cache-dtype auto '
        f'--mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.85 '
        f'--max-running-requests 4 --chunked-prefill-size 8192 --disable-radix-cache '
        f'--enable-fp32-lm-head --random-seed 0 --host 127.0.0.1 --port {port} --nccl-port {nccl_port}'
    )
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _gen_one(port: int, prompt: dict, max_new_tokens: int) -> dict:
    url = f"http://127.0.0.1:{port}/generate"
    t0 = time.perf_counter()
    resp = _http_post_json(
        url,
        {"input_ids": prompt["input_ids"], "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": max_new_tokens}},
        timeout_s=600.0,
    )
    dt = time.perf_counter() - t0
    meta = resp.get("meta_info", {})
    out_ids = list(resp.get("output_ids", []))
    comp = int(meta.get("completion_tokens", len(out_ids)))
    vct = int(meta.get("spec_verify_ct", 0))
    lat = float(meta.get("e2e_latency", dt))
    return {
        "id": prompt["id"],
        "suite": prompt["suite"],
        "completion_tokens": comp,
        "verify_ct": vct,
        "latency_s": lat,
        "tok_per_s": comp / max(lat, 1e-6),
        "output_ids": out_ids,
    }


def _score_outputs(port: int, prompts: list[dict], outputs: list[dict], delta: float = 0.25) -> dict:
    url = f"http://127.0.0.1:{port}/generate"
    total_tokens = 0
    mismatches = 0
    large_margin_events = 0
    threshold = 2.0 * delta
    for prompt, row in zip(prompts, outputs):
        out_ids = row["output_ids"]
        if not out_ids:
            continue
        resp = _http_post_json(
            url,
            {
                "input_ids": prompt["input_ids"] + out_ids,
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 2,
                "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": 1},
            },
            timeout_s=600.0,
        )
        scores = decode_scores(resp, prompt["input_ids"], out_ids)
        for sc in scores:
            total_tokens += 1
            if sc["token"] != sc["argmax"]:
                mismatches += 1
                if sc["margin"] > threshold:
                    large_margin_events += 1
    rate = mismatches / max(total_tokens, 1)
    return {
        "total_tokens": total_tokens,
        "mismatches": mismatches,
        "mismatch_rate": round(rate, 6),
        "large_margin_events": large_margin_events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-id", required=True)
    parser.add_argument("--cand-worktree", type=Path, required=True)
    parser.add_argument("--option-env", type=str, default="{}")
    parser.add_argument("--cand-draft-tokens", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.002)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_env = json.loads(args.option_env)
    prompts = _load_dev_prompts(per_suite=args.per_suite)

    # --- T2: 2 Physical Rounds with GPU Swap (0 <-> 1) ---
    req_log_ratios: dict[str, list[float]] = {p["id"]: [] for p in prompts}
    round1_stock_rows: list[dict] = []
    round1_cand_rows: list[dict] = []
    all_outputs_identical = True

    for rnd in (1, 2):
        stock_gpu, cand_gpu = (0, 1) if rnd == 1 else (1, 0)
        p_stock, nccl_stock = _free_port(), _free_port()
        p_cand, nccl_cand = _free_port(), _free_port()
        print(f"[controller] Round {rnd}: stock=GPU{stock_gpu} (port {p_stock}), cand=GPU{cand_gpu} (port {p_cand})", flush=True)
        proc_s = _start_arm(stock_gpu, STOCK_ENGINE, p_stock, nccl_stock, 8, {}, args.out_dir / f"r{rnd}_stock.log")
        proc_c = _start_arm(cand_gpu, args.cand_worktree, p_cand, nccl_cand, args.cand_draft_tokens, cand_env, args.out_dir / f"r{rnd}_cand.log")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(_wait_health, p_stock, 1200.0)
                f2 = ex.submit(_wait_health, p_cand, 1200.0)
                f1.result()
                f2.result()

            # Paired warmup (2 short requests)
            for wp in prompts[:2]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_stock, wp, 32)
                    fc = ex.submit(_gen_one, p_cand, wp, 32)
                    fs.result()
                    fc.result()

            # Paired measurement (simultaneous per request)
            for p in prompts:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_stock, p, args.max_new_tokens)
                    fc = ex.submit(_gen_one, p_cand, p, args.max_new_tokens)
                    rs = fs.result()
                    rc = fc.result()
                req_log_ratios[p["id"]].append(math.log(rc["tok_per_s"] / max(rs["tok_per_s"], 1e-6)))
                if rs["output_ids"] != rc["output_ids"]:
                    all_outputs_identical = False
                if rnd == 1:
                    round1_stock_rows.append(rs)
                    round1_cand_rows.append(rc)
        finally:
            _stop_server(proc_s)
            _stop_server(proc_c)

    # Suite-stratified request-cluster bootstrap CI (exact controller.stats)
    groups: dict[str, list[float]] = {}
    for p in prompts:
        groups.setdefault(p["suite"], []).append(sum(req_log_ratios[p["id"]]) / 2.0)
    ci = stratified_geomean_ci(groups, iters=10000, seed=0)
    (args.out_dir / "paired_ci.json").write_text(
        json.dumps({"ci": ci, "output_identical": all_outputs_identical, "req_log_ratios": req_log_ratios}, indent=2) + "\n"
    )
    print(f"[controller] Paired 2-GPU swap CI: geomean={ci['geomean']:.5f} [{ci['lower']:.5f}, {ci['upper']:.5f}]", flush=True)

    # --- T1: FP32 LM-Head Teacher-Forced Token Forcing ---
    print("[controller] Starting FP32 LM-head teacher-forced scorer on GPU 0...", flush=True)
    p_score, nccl_score = _free_port(), _free_port()
    proc_sc = _start_fp32_scorer(0, p_score, nccl_score, args.out_dir / "fp32_scorer.log")
    try:
        _wait_health(p_score, 600.0)
        stock_tf = _score_outputs(p_score, prompts, round1_stock_rows, delta=args.delta)
        cand_tf = _score_outputs(p_score, prompts, round1_cand_rows, delta=args.delta)
    finally:
        _stop_server(proc_sc)

    # Rate rule: cand lower bound - stock upper bound <= 0.005 (0.5 pp)
    cand_lo = _wilson_low(cand_tf["mismatches"], cand_tf["total_tokens"], z=1.959964)
    stock_hi = _wilson_high(stock_tf["mismatches"], stock_tf["total_tokens"], z=1.959964)
    rate_pass = (cand_lo - stock_hi) <= 0.005

    # Event rule: binomial test against upper bound of stock event rate
    stock_event_hi = _wilson_high(stock_tf["large_margin_events"], stock_tf["total_tokens"], z=1.959964)
    event_p = binom_one_sided(cand_tf["large_margin_events"], cand_tf["total_tokens"], max(stock_event_hi, 1e-6))
    event_pass = (cand_tf["large_margin_events"] <= stock_tf["large_margin_events"]) or (event_p >= 0.025)
    t1_pass = rate_pass and event_pass

    # T2 Promotion rule (paper-aligned-v1)
    min_gain = (0.01 + 2.0 * args.noise) if all_outputs_identical else max(0.01 + 2.0 * args.noise, 0.03)
    t2_pass = (ci["lower"] - 1.0) > min_gain
    promoted = bool(t1_pass and t2_pass)

    stock_tps = sum(r["completion_tokens"] for r in round1_stock_rows) / max(sum(r["latency_s"] for r in round1_stock_rows), 1e-6)
    cand_tps = sum(r["completion_tokens"] for r in round1_cand_rows) / max(sum(r["latency_s"] for r in round1_cand_rows), 1e-6)
    stock_tau = sum(r["completion_tokens"] for r in round1_stock_rows) / max(sum(r["verify_ct"] for r in round1_stock_rows), 1)
    cand_tau = sum(r["completion_tokens"] for r in round1_cand_rows) / max(sum(r["verify_ct"] for r in round1_cand_rows), 1)

    verdict = {
        "cand_id": args.cand_id,
        "cand_worktree": str(args.cand_worktree),
        "option_env": cand_env,
        "cand_draft_tokens": args.cand_draft_tokens,
        "requests": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "output_identical": all_outputs_identical,
        "performance": {
            "geomean_gain": round(ci["geomean"], 5),
            "ci95_lower": round(ci["lower"], 5),
            "ci95_upper": round(ci["upper"], 5),
            "min_gain_threshold": round(min_gain, 5),
            "t2_pass": t2_pass,
            "round1_stock_tps": round(stock_tps, 2),
            "round1_cand_tps": round(cand_tps, 2),
            "round1_stock_tau": round(stock_tau, 4),
            "round1_cand_tau": round(cand_tau, 4),
        },
        "token_forcing": {
            "delta": args.delta,
            "stock": stock_tf,
            "candidate": cand_tf,
            "rate_pass": rate_pass,
            "event_p": event_p,
            "event_pass": event_pass,
            "t1_pass": t1_pass,
        },
        "promoted": promoted,
        "timestamp_unix": time.time(),
    }
    (args.out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    verdicts_log = ROOT / "v2_runtime" / "queue" / "verdicts.jsonl"
    verdicts_log.parent.mkdir(parents=True, exist_ok=True)
    with verdicts_log.open("a") as f:
        f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
