#!/usr/bin/env python3
"""T1-S: Temperature > 0 Target-Only Lossless Verification Metrics & Gate.

Mirrors the temp=0 T1 FP32 Token Forcing design in `controller/stats.py` and
`v2/controller_verify_2gpu.py`, extended to stochastic Target-Only decoding:
  - M1: Large-divergence events (TVD_t > 0.05) one-sided binomial test vs Stock Wilson hi
  - M2: Mean TVD <= max(1.25 * stock, stock + 0.002)
  - M3: P99 TVD  <= max(1.25 * stock, stock + 0.010)
  - M4: Mean |dlogp(y_t)| <= max(1.25 * stock, stock + 0.005)
  - M5: Anti-misrouting guard (stochasticity smoke + Martingale Z_NLL)

Also supports caching the Stock T1-S baseline per (model, stock_sha, prompt_sha,
temperature, top_p, top_k) so Phase 0 computes it once and later rounds reuse it.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from controller.stats import binom_one_sided


def _wilson_high(k: int, n: int, z: float = 1.959964) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + margin) / denom)


def _kolmogorov_p_value(d_stat: float, n: int) -> float:
    """Asymptotic two-sided Kolmogorov-Smirnov p-value for sample size n."""
    if n <= 0 or d_stat <= 0.0:
        return 1.0
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d_stat
    if lam <= 0.0:
        return 1.0
    # Smirnov series: 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 lam^2)
    total = 0.0
    for k in range(1, 100):
        term = math.exp(-2.0 * (k * k) * (lam * lam))
        total += term if (k % 2 == 1) else -term
        if term < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * total))


def ks_uniform_test(u_samples: list[float]) -> tuple[float, float]:
    """One-sample KS test of u_samples against Uniform(0, 1)."""
    n = len(u_samples)
    if n == 0:
        return 0.0, 1.0
    xs = sorted(max(0.0, min(1.0, float(x))) for x in u_samples)
    d_max = 0.0
    for i, x in enumerate(xs):
        d_plus = (i + 1) / n - x
        d_minus = x - i / n
        if d_plus > d_max:
            d_max = d_plus
        if d_minus > d_max:
            d_max = d_minus
    return d_max, _kolmogorov_p_value(d_max, n)


def normalize_top_logprobs(
    top_entries: list[tuple[float, int]] | list[list[Any]],
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> dict[int, float]:
    """Convert raw SGLang top_logprobs [(logprob, token_id, ...), ...] to target sampling probs."""
    if not top_entries:
        return {}
    temp = max(float(temperature), 1e-6)
    items: list[tuple[int, float]] = []
    for entry in top_entries:
        lp = float(entry[0])
        tid = int(entry[1])
        items.append((tid, lp / temp))
    # Sort descending by scaled logprob
    items.sort(key=lambda kv: kv[1], reverse=True)
    if top_k is not None and int(top_k) > 0:
        items = items[: int(top_k)]
    max_s = items[0][1]
    exps = [(tid, math.exp(s - max_s)) for tid, s in items]
    z = sum(v for _, v in exps)
    if z <= 0.0:
        return {}
    probs = [(tid, v / z) for tid, v in exps]

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        kept: list[tuple[int, float]] = []
        cum = 0.0
        for tid, p in probs:
            kept.append((tid, p))
            cum += p
            if cum >= float(top_p):
                break
        kz = sum(p for _, p in kept)
        probs = [(tid, p / kz) for tid, p in kept] if kz > 0.0 else probs

    return {tid: p for tid, p in probs}


def compute_step_metrics(
    *,
    ref_top_entries: list[Any],
    cand_top_entries: list[Any] | None,
    emitted_token: int,
    temperature: float,
    top_k: int = -1,
    top_p: float = 1.0,
    ref_tok_raw_lp: float | None = None,
    cand_tok_raw_lp: float | None = None,
    cand_already_temp_scaled: bool = False,
    rng: random.Random | None = None,
) -> dict[str, float | int | None]:
    """Compute per-token TVD, dlogp, NLL/entropy/var, top-1, and randomized PIT."""
    p_ref = normalize_top_logprobs(ref_top_entries, temperature=temperature, top_k=top_k, top_p=top_p)
    if not p_ref:
        return {}

    eps = 1e-12
    # Reference entropy, variance of -log p, top-1, and PIT
    sorted_ref = sorted(p_ref.items(), key=lambda kv: kv[1], reverse=True)
    argmax_ref, max_p_ref = sorted_ref[0]
    ent = -sum(p * math.log(max(p, eps)) for _, p in sorted_ref)
    var_nll = sum(p * ((-math.log(max(p, eps)) - ent) ** 2) for _, p in sorted_ref)

    p_ref_y = p_ref.get(int(emitted_token))
    if p_ref_y is None:
        # Tail token outside top-K: approximate using remaining mass or raw logprob
        p_ref_y = 1e-6
    nll_y = -math.log(max(p_ref_y, eps))

    # Randomized PIT U_t = F^-(y_t) + v_t * p_ref(y_t)
    cdf_before = 0.0
    found = False
    for tid, p in sorted_ref:
        if tid == int(emitted_token):
            found = True
            break
        cdf_before += p
    v = rng.random() if rng is not None else 0.5
    pit_u = (cdf_before + v * p_ref_y) if found else min(1.0, cdf_before + v * max(1.0 - cdf_before, 1e-6))

    tvd: float | None = None
    abs_dlogp: float | None = None
    if cand_top_entries:
        cand_temp = 1.0 if cand_already_temp_scaled else temperature
        p_cand = normalize_top_logprobs(cand_top_entries, temperature=cand_temp, top_k=top_k, top_p=top_p)
        if p_cand:
            union_keys = set(p_ref.keys()) | set(p_cand.keys())
            tvd = 0.5 * sum(abs(p_ref.get(k, 0.0) - p_cand.get(k, 0.0)) for k in union_keys)
            p_cand_y = p_cand.get(int(emitted_token))
            if p_cand_y is not None and p_ref.get(int(emitted_token)) is not None:
                abs_dlogp = abs(math.log(max(p_cand_y, eps)) - math.log(max(p_ref_y, eps)))
    if abs_dlogp is None and ref_tok_raw_lp is not None and cand_tok_raw_lp is not None:
        if cand_already_temp_scaled:
            abs_dlogp = abs(float(cand_tok_raw_lp) - float(ref_tok_raw_lp) / max(float(temperature), 1e-6))
        else:
            abs_dlogp = abs(float(cand_tok_raw_lp) - float(ref_tok_raw_lp)) / max(float(temperature), 1e-6)

    return {
        "tvd": tvd,
        "abs_dlogp": abs_dlogp,
        "nll_y": nll_y,
        "entropy_ref": ent,
        "var_nll_ref": var_nll,
        "hit_top1": 1 if int(emitted_token) == int(argmax_ref) else 0,
        "max_p_ref": max_p_ref,
        "pit_u": pit_u,
    }


def aggregate_t1s_metrics(
    step_metrics: list[dict[str, Any]],
    tvd_event_threshold: float = 0.05,
) -> dict[str, Any]:
    """Aggregate per-token T1-S metrics into distribution + Martingale/PIT summary."""
    valid = [m for m in step_metrics if m]
    n = len(valid)
    if n == 0:
        return {
            "total_tokens": 0,
            "dist_tokens": 0,
            "large_div_events": 0,
            "mean_tvd": None,
            "p99_tvd": None,
            "mean_abs_dlogp": None,
            "z_nll": 0.0,
            "z_top1": 0.0,
            "pit_ks_stat": 0.0,
            "pit_ks_p": 1.0,
        }

    tvds = [float(m["tvd"]) for m in valid if m.get("tvd") is not None]
    dlogps = [float(m["abs_dlogp"]) for m in valid if m.get("abs_dlogp") is not None]
    large_div = sum(1 for x in tvds if x > tvd_event_threshold)
    mean_tvd = (sum(tvds) / len(tvds)) if tvds else None
    if tvds:
        s_tvds = sorted(tvds)
        p99_idx = min(len(s_tvds) - 1, int(math.floor(0.99 * len(s_tvds))))
        p99_tvd = s_tvds[p99_idx]
    else:
        p99_tvd = None
    mean_dlogp = (sum(dlogps) / len(dlogps)) if dlogps else None

    # Martingale Z_NLL
    num_nll = sum(float(m["nll_y"]) - float(m["entropy_ref"]) for m in valid)
    var_nll = sum(max(float(m["var_nll_ref"]), 0.0) for m in valid)
    z_nll = num_nll / math.sqrt(var_nll) if var_nll > 1e-12 else 0.0

    # Martingale Z_top1
    num_t1 = sum(float(m["hit_top1"]) - float(m["max_p_ref"]) for m in valid)
    var_t1 = sum(max(float(m["max_p_ref"]) * (1.0 - float(m["max_p_ref"])), 0.0) for m in valid)
    z_top1 = num_t1 / math.sqrt(var_t1) if var_t1 > 1e-12 else 0.0

    # Randomized PIT KS test
    ks_stat, ks_p = ks_uniform_test([float(m["pit_u"]) for m in valid])

    return {
        "total_tokens": n,
        "dist_tokens": len(tvds),
        "tvd_event_threshold": tvd_event_threshold,
        "large_div_events": large_div,
        "mean_tvd": round(mean_tvd, 6) if mean_tvd is not None else None,
        "p99_tvd": round(p99_tvd, 6) if p99_tvd is not None else None,
        "mean_abs_dlogp": round(mean_dlogp, 6) if mean_dlogp is not None else None,
        "z_nll": round(z_nll, 4),
        "z_top1": round(z_top1, 4),
        "pit_ks_stat": round(ks_stat, 5),
        "pit_ks_p": round(ks_p, 6),
    }


def evaluate_t1s_verdict(
    stock_metrics: dict[str, Any],
    cand_metrics: dict[str, Any],
    stochasticity_check: dict[str, Any] | None = None,
    z_limit: float = 3.5,
) -> dict[str, Any]:
    """Evaluate T1-S M1..M5 pass/fail gates against Stock baseline."""
    has_dist = (
        cand_metrics.get("mean_tvd") is not None
        and stock_metrics.get("mean_tvd") is not None
        and cand_metrics.get("dist_tokens", 0) > 0
        and stock_metrics.get("dist_tokens", 0) > 0
    )

    if has_dist:
        s_ev = int(stock_metrics["large_div_events"])
        c_ev = int(cand_metrics["large_div_events"])
        s_n = int(stock_metrics["dist_tokens"])
        c_n = int(cand_metrics["dist_tokens"])
        stock_ev_hi = _wilson_high(s_ev, s_n, z=1.959964)
        event_p = binom_one_sided(c_ev, c_n, max(stock_ev_hi, 1e-6))
        m1_pass = (c_ev <= s_ev) or (event_p >= 0.025)

        s_mtvd = float(stock_metrics["mean_tvd"])
        c_mtvd = float(cand_metrics["mean_tvd"])
        m2_pass = c_mtvd <= max(1.25 * s_mtvd, s_mtvd + 0.002)

        s_p99 = float(stock_metrics["p99_tvd"])
        c_p99 = float(cand_metrics["p99_tvd"])
        m3_pass = c_p99 <= max(1.25 * s_p99, s_p99 + 0.010)

        if cand_metrics.get("mean_abs_dlogp") is not None and stock_metrics.get("mean_abs_dlogp") is not None:
            s_dl = float(stock_metrics["mean_abs_dlogp"])
            c_dl = float(cand_metrics["mean_abs_dlogp"])
            m4_pass = c_dl <= max(1.25 * s_dl, s_dl + 0.005)
        else:
            m4_pass = True
    else:
        event_p = None
        m1_pass = True
        m2_pass = True
        m3_pass = True
        m4_pass = True

    stoch_ok = True if stochasticity_check is None else bool(stochasticity_check.get("actually_stochastic", False))
    c_znll = float(cand_metrics.get("z_nll", 0.0))
    s_znll = float(stock_metrics.get("z_nll", 0.0))
    z_nll_ok = (abs(c_znll) <= z_limit) or (abs(c_znll - s_znll) <= z_limit)
    c_zt1 = float(cand_metrics.get("z_top1", 0.0))
    s_zt1 = float(stock_metrics.get("z_top1", 0.0))
    z_t1_ok = (abs(c_zt1) <= z_limit) or (abs(c_zt1 - s_zt1) <= z_limit)
    m5_pass = bool(stoch_ok and z_nll_ok and z_t1_ok)

    t1s_pass = bool(m1_pass and m2_pass and m3_pass and m4_pass and m5_pass)
    return {
        "has_decode_dist_comparison": has_dist,
        "m1_events_pass": m1_pass,
        "m1_event_p": round(event_p, 6) if event_p is not None else None,
        "m2_mean_tvd_pass": m2_pass,
        "m3_p99_tvd_pass": m3_pass,
        "m4_dlogp_pass": m4_pass,
        "m5_routing_pass": m5_pass,
        "m5_stochastic_ok": stoch_ok,
        "m5_z_nll_ok": z_nll_ok,
        "m5_z_top1_ok": z_t1_ok,
        "t1s_pass": t1s_pass,
    }


def stock_cache_key(
    *,
    model_path: str,
    stock_worktree: str,
    prompt_ids: list[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    payload = {
        "model_path": str(model_path),
        "stock_worktree": str(stock_worktree),
        "prompt_ids": list(prompt_ids),
        "temperature": round(float(temperature), 6),
        "top_p": round(float(top_p), 6),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_stock_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"t1s_stock_{key}.json"
    if path.is_file():
        return json.loads(path.read_text())
    return None


def save_stock_cache(cache_dir: Path, key: str, data: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"t1s_stock_{key}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path
