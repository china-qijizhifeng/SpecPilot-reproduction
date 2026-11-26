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
from v2.phase0.measure_stock import (
    _free_port,
    _http_post_json,
    _load_dev_prompts,
    _make_sampling_params,
    _stop_server,
    _wait_health,
)
from v2.t1s_verify import (
    aggregate_t1s_metrics,
    compute_step_metrics,
    evaluate_t1s_verdict,
    load_stock_cache,
    save_stock_cache,
    stock_cache_key,
)

STOCK_ENGINE = Path("/gpfs/users/jixinpeng/specpilot_data/work/engines/h100_c1_qwen38-27b_dflash2-stock")


def _wilson_high(k: int, n: int, z: float = 1.959964) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    rad = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + rad) / denom)


def _load_prompts_for_model(per_suite: int, model_path: str | None) -> list[dict]:
    if not model_path:
        return _load_dev_prompts(per_suite=per_suite)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dev_dir = ROOT / "assets/upstream/dev"
    suites = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
    prompts: list[dict] = []
    for suite in suites:
        count = 0
        with (dev_dir / f"{suite}.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": item["turns"][0]}],
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=True,
                )
                if not isinstance(ids, list):
                    ids = ids["input_ids"]
                prompts.append({"id": item["id"], "suite": suite, "input_ids": [int(x) for x in ids]})
                count += 1
                if count >= per_suite:
                    break
    return prompts


def _start_arm(
    gpu: int,
    engine_dir: Path,
    port: int,
    nccl_port: int,
    draft_tokens: int,
    extra_env: dict[str, str],
    log_path: Path,
    extra_args: list[str] | None = None,
    *,
    model_path: str | None = None,
    spec_algo: str = "DFLASH",
    spec_num_steps: int = 3,
    spec_eagle_topk: int = 1,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(engine_dir)
    env["PORT"] = str(port)
    env["NCCL_PORT"] = str(nccl_port)
    env["DRAFT_TOKENS"] = str(draft_tokens)
    env.update({str(k): str(v) for k, v in extra_env.items()})
    arg_str = " ".join(extra_args or [])
    if model_path and spec_algo.upper() != "DFLASH":
        cmd = (
            f'source "{ROOT}/harness/sglang/env.sh" && '
            f'exec "$SP_PY" -m sglang.launch_server '
            f'--model-path "{model_path}" '
            f'--speculative-algorithm {spec_algo} '
            f'--speculative-num-steps {spec_num_steps} '
            f'--speculative-eagle-topk {spec_eagle_topk} '
            f'--speculative-num-draft-tokens {draft_tokens} '
            f'--kv-cache-dtype auto --mamba-ssm-dtype float32 '
            f'--attention-backend fa3 --mem-fraction-static 0.85 '
            f'--max-running-requests 4 --cuda-graph-max-bs-decode 4 '
            f'--chunked-prefill-size 8192 --disable-radix-cache --random-seed 0 '
            f'--host 127.0.0.1 --port {port} --nccl-port {nccl_port} {arg_str}'
        ).strip()
    else:
        cmd = f'bash "{ROOT}/harness/sglang/serve.sh" {arg_str}'.strip()
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _start_fp32_scorer(
    gpu: int,
    port: int,
    nccl_port: int,
    log_path: Path,
    *,
    stock_engine: Path = STOCK_ENGINE,
    model_path: str | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(stock_engine)
    model_arg = f'"{model_path}"' if model_path else '"$SP_TARGET"'
    dtype_arg = "" if model_path else "--dtype bfloat16 "
    cmd = (
        f'source "{ROOT}/harness/sglang/env.sh" && '
        f'exec "$SP_PY" -m sglang.launch_server '
        f'--model-path {model_arg} {dtype_arg}--kv-cache-dtype auto '
        f'--mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.85 '
        f'--max-running-requests 4 --chunked-prefill-size 8192 --disable-radix-cache '
        f'--enable-fp32-lm-head --random-seed 0 --host 127.0.0.1 --port {port} --nccl-port {nccl_port}'
    )
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _gen_one(
    port: int,
    prompt: dict,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    return_logprob: bool = False,
    top_logprobs_num: int = 20,
) -> dict:
    url = f"http://127.0.0.1:{port}/generate"
    payload: dict = {
        "input_ids": prompt["input_ids"],
        "sampling_params": _make_sampling_params(
            max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k
        ),
    }
    if return_logprob:
        payload["return_logprob"] = True
        payload["top_logprobs_num"] = int(top_logprobs_num)
    t0 = time.perf_counter()
    resp = _http_post_json(url, payload, timeout_s=600.0)
    dt = time.perf_counter() - t0
    meta = resp.get("meta_info", {})
    out_ids = list(resp.get("output_ids", []))
    comp = int(meta.get("completion_tokens", len(out_ids)))
    vct = int(meta.get("spec_verify_ct", 0))
    lat = float(meta.get("e2e_latency", dt))
    row = {
        "id": prompt["id"],
        "suite": prompt["suite"],
        "completion_tokens": comp,
        "verify_ct": vct,
        "latency_s": lat,
        "tok_per_s": comp / max(lat, 1e-6),
        "output_ids": out_ids,
    }
    if return_logprob:
        row["output_top_logprobs"] = meta.get("output_top_logprobs")
        row["output_token_logprobs"] = meta.get("output_token_logprobs")
    return row


def _check_stochasticity(
    port: int,
    prompt: dict,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = -1,
) -> dict:
    rg = _gen_one(port, prompt, 64, temperature=0.0)
    rs1 = _gen_one(port, prompt, 64, temperature=temperature, top_p=top_p, top_k=top_k)
    rs2 = _gen_one(port, prompt, 64, temperature=temperature, top_p=top_p, top_k=top_k)
    g, s1, s2 = rg["output_ids"], rs1["output_ids"], rs2["output_ids"]
    return {
        "s1_differs_from_greedy": s1 != g,
        "s2_differs_from_s1": s2 != s1,
        "actually_stochastic": (s1 != g) and (s2 != s1),
    }


def _score_outputs_t1s(
    port: int,
    prompts: list[dict],
    outputs: list[dict],
    temperature: float,
    top_p: float = 1.0,
    top_k: int = -1,
    top_logprobs_num: int = 20,
) -> dict:
    import random

    url = f"http://127.0.0.1:{port}/generate"
    rng = random.Random(0)
    step_metrics: list[dict] = []
    for prompt, row in zip(prompts, outputs):
        out_ids = row["output_ids"]
        if not out_ids:
            continue
        in_ids = prompt["input_ids"]
        resp = _http_post_json(
            url,
            {
                "input_ids": in_ids + out_ids,
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": int(top_logprobs_num),
                "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": 1},
            },
            timeout_s=600.0,
        )
        meta = resp.get("meta_info", {})
        in_top = meta.get("input_top_logprobs") or []
        in_tok = meta.get("input_token_logprobs") or []
        cand_top = row.get("output_top_logprobs") or []
        cand_tok = row.get("output_token_logprobs") or []
        n_in = len(in_ids)
        for i, tok_id in enumerate(out_ids):
            pos = n_in + i
            if pos >= len(in_top) or not in_top[pos]:
                continue
            ref_top_i = in_top[pos]
            ref_lp_i = float(in_tok[pos][0]) if (pos < len(in_tok) and in_tok[pos] and in_tok[pos][0] is not None) else None
            cand_top_i = cand_top[i] if (i < len(cand_top) and cand_top[i]) else None
            cand_lp_i = float(cand_tok[i][0]) if (i < len(cand_tok) and cand_tok[i] and cand_tok[i][0] is not None) else None
            m = compute_step_metrics(
                ref_top_entries=ref_top_i,
                cand_top_entries=cand_top_i,
                emitted_token=int(tok_id),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ref_tok_raw_lp=ref_lp_i,
                cand_tok_raw_lp=cand_lp_i,
                cand_already_temp_scaled=True,
                rng=rng,
            )
            if m:
                step_metrics.append(m)
    return aggregate_t1s_metrics(step_metrics, tvd_event_threshold=0.05)


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
    parser.add_argument("--stock-worktree", type=Path, default=STOCK_ENGINE)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--spec-algo", type=str, default="DFLASH")
    parser.add_argument("--spec-num-steps", type=int, default=3)
    parser.add_argument("--spec-eagle-topk", type=int, default=1)
    parser.add_argument("--stock-draft-tokens", type=int, default=8)
    parser.add_argument("--option-env", type=str, default="{}")
    parser.add_argument("--cand-draft-tokens", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--t1s-top-logprobs", type=int, default=20)
    parser.add_argument("--t1s-stock-cache-dir", type=Path, default=ROOT / "v2_runtime" / "cache" / "t1s_stock")
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.002)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_env = json.loads(args.option_env)
    prompts = _load_prompts_for_model(per_suite=args.per_suite, model_path=args.model_path)
    is_stochastic = float(args.temperature) > 0.0

    # Check cached Stock T1-S baseline when temperature > 0
    t1s_cache_k = None
    cached_stock_t1s = None
    if is_stochastic:
        t1s_cache_k = stock_cache_key(
            model_path=args.model_path or "Qwen3.8-27B",
            stock_worktree=str(args.stock_worktree),
            prompt_ids=[p["id"] for p in prompts],
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        cached_stock_t1s = load_stock_cache(args.t1s_stock_cache_dir, t1s_cache_k)
        if cached_stock_t1s is not None:
            print(f"[controller] Loaded cached Stock T1-S baseline (key={t1s_cache_k})", flush=True)

    # --- T2: 2 Physical Rounds with GPU Swap (0 <-> 1) ---
    req_log_ratios: dict[str, list[float]] = {p["id"]: [] for p in prompts}
    round1_stock_rows: list[dict] = []
    round1_cand_rows: list[dict] = []
    t1s_stock_rows: list[dict] = []
    t1s_cand_rows: list[dict] = []
    cand_stoch_check: dict | None = None
    all_outputs_identical = True

    for rnd in (1, 2):
        stock_gpu, cand_gpu = (0, 1) if rnd == 1 else (1, 0)
        p_stock, nccl_stock = _free_port(), _free_port()
        p_cand, nccl_cand = _free_port(), _free_port()
        print(f"[controller] Round {rnd} (temp={args.temperature}): stock=GPU{stock_gpu} (port {p_stock}), cand=GPU{cand_gpu} (port {p_cand})", flush=True)
        proc_s = _start_arm(
            stock_gpu, args.stock_worktree, p_stock, nccl_stock, args.stock_draft_tokens, {},
            args.out_dir / f"r{rnd}_stock.log",
            model_path=args.model_path, spec_algo=args.spec_algo,
            spec_num_steps=args.spec_num_steps, spec_eagle_topk=args.spec_eagle_topk,
        )
        proc_c = _start_arm(
            cand_gpu, args.cand_worktree, p_cand, nccl_cand, args.cand_draft_tokens, cand_env,
            args.out_dir / f"r{rnd}_cand.log",
            model_path=args.model_path, spec_algo=args.spec_algo,
            spec_num_steps=args.spec_num_steps, spec_eagle_topk=args.spec_eagle_topk,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(_wait_health, p_stock, 1200.0)
                f2 = ex.submit(_wait_health, p_cand, 1200.0)
                f1.result()
                f2.result()

            # Paired warmup (2 short requests)
            for wp in prompts[:2]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_stock, wp, 32, args.temperature, args.top_p, args.top_k)
                    fc = ex.submit(_gen_one, p_cand, wp, 32, args.temperature, args.top_p, args.top_k)
                    fs.result()
                    fc.result()

            # Paired measurement (simultaneous per request; return_logprob=False for clean timing)
            for p in prompts:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fs = ex.submit(_gen_one, p_stock, p, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
                    fc = ex.submit(_gen_one, p_cand, p, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
                    rs = fs.result()
                    rc = fc.result()
                req_log_ratios[p["id"]].append(math.log(rc["tok_per_s"] / max(rs["tok_per_s"], 1e-6)))
                if rs["output_ids"] != rc["output_ids"]:
                    all_outputs_identical = False
                if rnd == 1:
                    round1_stock_rows.append(rs)
                    round1_cand_rows.append(rc)

            # In Round 1 for temp > 0: collect T1-S decode top-logprobs + stochasticity smoke
            if rnd == 1 and is_stochastic:
                cand_stoch_check = _check_stochasticity(
                    p_cand, prompts[0], args.temperature, args.top_p, args.top_k
                )
                print(f"[controller] T1-S stochasticity smoke: {cand_stoch_check}", flush=True)
                for p in prompts:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        fc = ex.submit(
                            _gen_one, p_cand, p, args.max_new_tokens,
                            args.temperature, args.top_p, args.top_k, True, args.t1s_top_logprobs,
                        )
                        fs = None
                        if cached_stock_t1s is None:
                            fs = ex.submit(
                                _gen_one, p_stock, p, args.max_new_tokens,
                                args.temperature, args.top_p, args.top_k, True, args.t1s_top_logprobs,
                            )
                        t1s_cand_rows.append(fc.result())
                        if fs is not None:
                            t1s_stock_rows.append(fs.result())
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

    # --- T1 / T1-S: FP32 LM-Head Teacher-Forced Token Forcing ---
    print("[controller] Starting FP32 LM-head teacher-forced scorer on GPU 0...", flush=True)
    p_score, nccl_score = _free_port(), _free_port()
    proc_sc = _start_fp32_scorer(
        0, p_score, nccl_score, args.out_dir / "fp32_scorer.log",
        stock_engine=args.stock_worktree, model_path=args.model_path,
    )
    try:
        _wait_health(p_score, 600.0)
        if is_stochastic:
            if cached_stock_t1s is not None:
                stock_t1s = cached_stock_t1s["metrics"]
            else:
                stock_t1s = _score_outputs_t1s(
                    p_score, prompts, t1s_stock_rows,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    top_logprobs_num=args.t1s_top_logprobs,
                )
                if t1s_cache_k is not None:
                    save_stock_cache(args.t1s_stock_cache_dir, t1s_cache_k, {
                        "cache_key": t1s_cache_k,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "metrics": stock_t1s,
                        "timestamp_unix": time.time(),
                    })
            cand_t1s = _score_outputs_t1s(
                p_score, prompts, t1s_cand_rows,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                top_logprobs_num=args.t1s_top_logprobs,
            )
            t1s_eval = evaluate_t1s_verdict(stock_t1s, cand_t1s, cand_stoch_check)
            t1_pass = bool(t1s_eval["t1s_pass"])
            token_forcing_block = {
                "mode": "T1-S_target_only",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "stock_cached": cached_stock_t1s is not None,
                "stock": stock_t1s,
                "candidate": cand_t1s,
                "stochasticity_check": cand_stoch_check,
                "gates": t1s_eval,
                "t1_pass": t1_pass,
            }
        else:
            stock_tf = _score_outputs(p_score, prompts, round1_stock_rows, delta=args.delta)
            cand_tf = _score_outputs(p_score, prompts, round1_cand_rows, delta=args.delta)
            cand_lo = _wilson_low(cand_tf["mismatches"], cand_tf["total_tokens"], z=1.959964)
            stock_hi = _wilson_high(stock_tf["mismatches"], stock_tf["total_tokens"], z=1.959964)
            rate_pass = (cand_lo - stock_hi) <= 0.005
            stock_event_hi = _wilson_high(stock_tf["large_margin_events"], stock_tf["total_tokens"], z=1.959964)
            event_p = binom_one_sided(cand_tf["large_margin_events"], cand_tf["total_tokens"], max(stock_event_hi, 1e-6))
            event_pass = (cand_tf["large_margin_events"] <= stock_tf["large_margin_events"]) or (event_p >= 0.025)
            t1_pass = rate_pass and event_pass
            token_forcing_block = {
                "mode": "T1_greedy",
                "delta": args.delta,
                "stock": stock_tf,
                "candidate": cand_tf,
                "rate_pass": rate_pass,
                "event_p": event_p,
                "event_pass": event_pass,
                "t1_pass": t1_pass,
            }
    finally:
        _stop_server(proc_sc)

    # T2 Promotion rule (paper-aligned-v1)
    min_gain = (0.01 + 2.0 * args.noise) if (not is_stochastic and all_outputs_identical) else max(0.01 + 2.0 * args.noise, 0.03)
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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
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
        "token_forcing": token_forcing_block,
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
