#!/usr/bin/env python3
"""Hardware-exact byte/FLOP floor table and 3-factor Ceiling for Qwen3.8-27B + DFlash2.

Implements SpecPilot §3.1 Eq. (3)-(4):
    S(N) = tau(N) / c(N),   c(N) = T_step(N) / T_AR
    Ceiling = (tau_best / tau) * (T_prof / T_floor) * (T_step / T_prof)
    T_floor(class) = max(FLOPs / (0.60 * dense_peak), bytes / (0.90 * hbm_bw))

All dimensions are loaded directly from the target & drafter config.json files
so there is zero magic hardcoding and zero unattributed 'other_unknown' guesswork.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    bf16_peak_flops: float  # FLOP/s dense BF16
    hbm_bw_bytes_s: float   # bytes/s
    flop_util: float = 0.60
    bw_util: float = 0.90

    def floor_ms(self, flops: float, nbytes: float) -> float:
        t_flop_s = flops / (self.flop_util * self.bf16_peak_flops)
        t_byte_s = nbytes / (self.bw_util * self.hbm_bw_bytes_s)
        return max(t_flop_s, t_byte_s) * 1e3


H100_SXM5 = DeviceSpec(
    name="H100-SXM5-80GB",
    bf16_peak_flops=989e12,
    hbm_bw_bytes_s=3.35e12,
)

DEFAULT_TARGET_CFG = Path("/gpfs/models/huggingface.co/Qwen/Qwen3.8-27B/config.json")
DEFAULT_DRAFT_CFG = Path("/gpfs/users/jixinpeng/models/z-lab/Qwen3.8-27B-DFlash2/config.json")


def compute_class_workloads(
    *,
    target_cfg: dict,
    draft_cfg: dict,
    verify_rows: int = 8,
    draft_block: int = 8,
    kv_len: int = 1024,
    weight_bytes_per_elem: float = 2.0,
    draft_weight_bytes_per_elem: float = 2.0,
    lookahead_weight_bytes_per_elem: float = 0.0,
    lookahead_rows: int = 0,
    lookahead_trigger_rate: float = 0.0,
    drafter_passes_per_step: float = 1.0,
    lm_head_passes_per_step: float = 1.0,
) -> dict[str, dict]:
    """Compute exact bytes & FLOPs per speculative step for each kernel class."""
    tc = target_cfg.get("text_config", target_cfg)
    h = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    vocab = int(tc["vocab_size"])
    layer_types = list(tc["layer_types"])
    n_layers = len(layer_types)
    n_gdn = sum(1 for t in layer_types if t == "linear_attention")
    n_attn = sum(1 for t in layer_types if t == "full_attention")

    # Full attention dimensions
    n_q_heads = int(tc["num_attention_heads"])
    n_kv_heads = int(tc["num_key_value_heads"])
    head_dim = int(tc["head_dim"])
    q_out = n_q_heads * head_dim * (2 if tc.get("attn_output_gate", False) else 1)
    k_out = n_kv_heads * head_dim
    v_out = n_kv_heads * head_dim
    o_in = n_q_heads * head_dim

    # GDN (Gated DeltaNet) dimensions
    gdn_k_heads = int(tc["linear_num_key_heads"])
    gdn_k_dim = int(tc["linear_key_head_dim"])
    gdn_v_heads = int(tc["linear_num_value_heads"])
    gdn_v_dim = int(tc["linear_value_head_dim"])
    gdn_conv_k = int(tc.get("linear_conv_kernel_dim", 4))
    gdn_qkvz_out = 2 * gdn_k_heads * gdn_k_dim + 2 * gdn_v_heads * gdn_v_dim
    gdn_ba_out = 2 * gdn_v_heads
    gdn_out_in = gdn_v_heads * gdn_v_dim

    # 1) target_gemm: all decoder linear projections (MLP + GDN proj + Attn proj)
    mlp_params_per_layer = h * (2 * inter) + inter * h  # gate_up + down
    gdn_proj_params = h * gdn_qkvz_out + h * gdn_ba_out + gdn_out_in * h
    attn_proj_params = h * (q_out + k_out + v_out) + o_in * h
    target_linear_params = (
        n_layers * mlp_params_per_layer
        + n_gdn * gdn_proj_params
        + n_attn * attn_proj_params
    )
    target_gemm_bytes = target_linear_params * weight_bytes_per_elem
    target_gemm_flops = 2.0 * target_linear_params * verify_rows

    # 2) lm_head (scaled by passes per step when multi-pass lookahead runs lm_head)
    lm_head_params = h * vocab
    lm_head_bytes = lm_head_params * weight_bytes_per_elem * lm_head_passes_per_step
    lm_head_flops = 2.0 * lm_head_params * verify_rows * lm_head_passes_per_step

    # 3) gdn_recurrent: FP32 state [n_v_heads, k_dim, v_dim] read + write per layer + conv state
    state_elems_per_layer = gdn_v_heads * gdn_k_dim * gdn_v_dim
    conv_channels = 2 * gdn_k_heads * gdn_k_dim + gdn_v_heads * gdn_v_dim
    gdn_state_bytes = n_gdn * (2 * state_elems_per_layer * 4 + 2 * conv_channels * gdn_conv_k * 2)
    # Per-token rank-1 delta rule update: ~6 * state_elems FLOPs per token per layer
    gdn_state_flops = n_gdn * verify_rows * 6.0 * state_elems_per_layer

    # 4) attention: KV cache read across full_attention layers (BF16 K and V)
    attn_kv_bytes = n_attn * 2 * n_kv_heads * head_dim * kv_len * 2
    attn_flops = n_attn * verify_rows * 4.0 * n_q_heads * head_dim * kv_len

    # 5) drafter (DFlash2): 5 transformer layers + feature fusion + selector + lm_head
    dh = int(draft_cfg["hidden_size"])
    dinter = int(draft_cfg["intermediate_size"])
    dlayers = int(draft_cfg["num_hidden_layers"])
    dq = int(draft_cfg["num_attention_heads"]) * int(draft_cfg["head_dim"])
    dkv = int(draft_cfg["num_key_value_heads"]) * int(draft_cfg["head_dim"])
    d_target_layers = len(draft_cfg.get("dflash_config", {}).get("target_layer_ids", [5, 19, 33, 47, 61]))
    d_layer_params = dh * (dq + 2 * dkv) + dq * dh + dh * (2 * dinter) + dinter * dh
    d_fc_params = (d_target_layers * dh) * dh
    # Drafter shares target LM head (or runs its own lm_head over draft_block rows)
    d_params = dlayers * d_layer_params + d_fc_params + h * vocab
    drafter_bytes = d_params * draft_weight_bytes_per_elem * drafter_passes_per_step
    drafter_flops = 2.0 * d_params * draft_block * drafter_passes_per_step

    out = {
        "target_gemm": {
            "params": target_linear_params,
            "bytes": target_gemm_bytes,
            "flops": target_gemm_flops,
        },
        "lm_head": {
            "params": lm_head_params,
            "bytes": lm_head_bytes,
            "flops": lm_head_flops,
        },
        "gdn_recurrent": {
            "bytes": gdn_state_bytes,
            "flops": gdn_state_flops,
        },
        "attention": {
            "bytes": attn_kv_bytes,
            "flops": attn_flops,
        },
        "drafter": {
            "params": d_params,
            "bytes": drafter_bytes,
            "flops": drafter_flops,
        },
    }

    if lookahead_trigger_rate > 0.0 and lookahead_weight_bytes_per_elem > 0.0:
        la_rows = lookahead_rows if lookahead_rows > 0 else verify_rows
        out["lookahead_gemm"] = {
            "params": target_linear_params,
            "bytes": target_linear_params * lookahead_weight_bytes_per_elem * lookahead_trigger_rate,
            "flops": 2.0 * target_linear_params * la_rows * lookahead_trigger_rate,
        }

    return out


def compute_ceiling_report(
    *,
    measured_step_ms: float,
    measured_class_ms: dict[str, float],
    measured_tau: float,
    tau_best: float,
    t_ar_ms: float | None = None,
    verify_rows: int = 8,
    draft_block: int = 8,
    kv_len: int = 1024,
    weight_bytes_per_elem: float = 2.0,
    draft_weight_bytes_per_elem: float = 2.0,
    lookahead_weight_bytes_per_elem: float = 0.0,
    lookahead_rows: int = 0,
    lookahead_trigger_rate: float = 0.0,
    drafter_passes_per_step: float = 1.0,
    lm_head_passes_per_step: float = 1.0,
    tau_headroom: dict | None = None,
    device: DeviceSpec = H100_SXM5,
    target_cfg: dict | None = None,
    draft_cfg: dict | None = None,
) -> dict:
    """Compute the full 3-factor SpecPilot Ceiling report and bottleneck ranking."""
    t_cfg = target_cfg if target_cfg is not None else json.loads(DEFAULT_TARGET_CFG.read_text())
    d_cfg = draft_cfg if draft_cfg is not None else json.loads(DEFAULT_DRAFT_CFG.read_text())
    workloads = compute_class_workloads(
        target_cfg=t_cfg,
        draft_cfg=d_cfg,
        verify_rows=verify_rows,
        draft_block=draft_block,
        kv_len=kv_len,
        weight_bytes_per_elem=weight_bytes_per_elem,
        draft_weight_bytes_per_elem=draft_weight_bytes_per_elem,
        lookahead_weight_bytes_per_elem=lookahead_weight_bytes_per_elem,
        lookahead_rows=lookahead_rows,
        lookahead_trigger_rate=lookahead_trigger_rate,
        drafter_passes_per_step=drafter_passes_per_step,
        lm_head_passes_per_step=lm_head_passes_per_step,
    )

    class_rows = {}
    t_floor_ms = 0.0
    t_prof_ms = 0.0
    for cls, w in workloads.items():
        f_ms = device.floor_ms(w["flops"], w["bytes"])
        m_ms = float(measured_class_ms.get(cls, 0.0))
        t_floor_ms += f_ms
        t_prof_ms += m_ms
        class_rows[cls] = {
            "bytes_gb": round(w["bytes"] / 1e9, 4),
            "tflops": round(w["flops"] / 1e12, 4),
            "floor_ms": round(f_ms, 4),
            "measured_ms": round(m_ms, 4),
            "gap_ms": round(max(m_ms - f_ms, 0.0), 4),
            "slowdown_vs_floor": round(m_ms / f_ms, 3) if f_ms > 0 and m_ms > 0 else None,
            "share_of_step_pct": round(100.0 * m_ms / measured_step_ms, 2) if measured_step_ms > 0 else 0.0,
        }

    # Include any residual profiled class (e.g. norms/sampling/other) in T_prof
    other_ms = float(measured_class_ms.get("other", 0.0))
    if other_ms > 0:
        t_prof_ms += other_ms
        class_rows["other"] = {
            "bytes_gb": 0.0,
            "tflops": 0.0,
            "floor_ms": 0.0,
            "measured_ms": round(other_ms, 4),
            "gap_ms": round(other_ms, 4),
            "slowdown_vs_floor": None,
            "share_of_step_pct": round(100.0 * other_ms / measured_step_ms, 2) if measured_step_ms > 0 else 0.0,
        }

    tau_factor = tau_best / max(measured_tau, 1e-9)
    kernel_factor = t_prof_ms / max(t_floor_ms, 1e-9) if t_prof_ms > 0 else 1.0
    overhead_factor = measured_step_ms / max(t_prof_ms, 1e-9) if t_prof_ms > 0 else 1.0
    ceiling = tau_factor * kernel_factor * overhead_factor

    factors = {
        "tau_factor": round(tau_factor, 4),
        "kernel_factor": round(kernel_factor, 4),
        "overhead_factor": round(overhead_factor, 4),
        "ceiling": round(ceiling, 4),
    }
    dominant = max(
        [("tau_factor", tau_factor), ("kernel_factor", kernel_factor), ("overhead_factor", overhead_factor)],
        key=lambda kv: kv[1],
    )[0]

    # Weight streaming share (target_gemm + lm_head)
    weight_bytes_gb = (workloads["target_gemm"]["bytes"] + workloads["lm_head"]["bytes"]) / 1e9
    weight_floor_ms = class_rows["target_gemm"]["floor_ms"] + class_rows["lm_head"]["floor_ms"]
    weight_meas_ms = class_rows["target_gemm"]["measured_ms"] + class_rows["lm_head"]["measured_ms"]

    speedup_over_ar = None
    if t_ar_ms and t_ar_ms > 0 and measured_step_ms > 0:
        speedup_over_ar = round(measured_tau / (measured_step_ms / t_ar_ms), 4)

    unattributed_overhead_ms = max(measured_step_ms - t_prof_ms, 0.0)
    unattributed_overhead_pct = (
        round(100.0 * unattributed_overhead_ms / measured_step_ms, 2) if measured_step_ms > 0 else 0.0
    )

    return {
        "device": device.name,
        "params": {
            "verify_rows": verify_rows,
            "draft_block": draft_block,
            "kv_len": kv_len,
            "weight_bytes_per_elem": weight_bytes_per_elem,
            "draft_weight_bytes_per_elem": draft_weight_bytes_per_elem,
            "lookahead_weight_bytes_per_elem": lookahead_weight_bytes_per_elem,
            "lookahead_trigger_rate": lookahead_trigger_rate,
            "drafter_passes_per_step": drafter_passes_per_step,
            "lm_head_passes_per_step": lm_head_passes_per_step,
        },
        "measured": {
            "tau": round(measured_tau, 4),
            "tau_best": round(tau_best, 4),
            "t_step_ms": round(measured_step_ms, 4),
            "t_prof_ms": round(t_prof_ms, 4),
            "t_floor_ms": round(t_floor_ms, 4),
            "unattributed_overhead_ms": round(unattributed_overhead_ms, 4),
            "unattributed_overhead_pct": unattributed_overhead_pct,
            "t_ar_ms": round(t_ar_ms, 4) if t_ar_ms else None,
            "speedup_over_ar": speedup_over_ar,
            "tok_per_s": round(1000.0 * measured_tau / measured_step_ms, 2) if measured_step_ms > 0 else 0.0,
        },
        "factors": factors,
        "dominant_factor": dominant,
        "tau_headroom": tau_headroom,
        "weight_streaming": {
            "target_linear_gb": round(workloads["target_gemm"]["bytes"] / 1e9, 4),
            "lm_head_gb": round(workloads["lm_head"]["bytes"] / 1e9, 4),
            "total_weight_gb": round(weight_bytes_gb, 4),
            "floor_share_pct": round(100.0 * weight_floor_ms / t_floor_ms, 2) if t_floor_ms > 0 else 0.0,
            "measured_share_pct": round(100.0 * weight_meas_ms / measured_step_ms, 2) if measured_step_ms > 0 else 0.0,
        },
        "classes": class_rows,
    }


def format_master_brief(report: dict) -> str:
    """Render a concise 1-page Phase 0 Markdown brief for the Master agent."""
    m = report["measured"]
    f = report["factors"]
    ws = report["weight_streaming"]
    lines = [
        "# Phase 0 Ceiling Brief",
        "",
        f"- **Device**: `{report['device']}` | **Verify rows**: `{report['params']['verify_rows']}` | **Draft block**: `{report['params']['draft_block']}`",
        f"- **Current speed**: `{m['tok_per_s']} tok/s` (`tau = {m['tau']}`, `T_step = {m['t_step_ms']} ms`, `T_prof = {m['t_prof_ms']} ms`, `T_floor = {m['t_floor_ms']} ms`)",
        f"- **System Overhead / Overlap Gap (`T_step - T_prof`)**: **`{m['unattributed_overhead_ms']} ms`** (`{m['unattributed_overhead_pct']}%` of step — host glue, launch gaps, sync bubbles)",
        f"- **Three-Factor Ceiling**: **`{f['ceiling']}x`** = `tau ({f['tau_factor']}x)` × `kernel ({f['kernel_factor']}x)` × `overhead ({f['overhead_factor']}x)`",
        f"- **Dominant Bottleneck**: **`{report['dominant_factor']}`**",
        f"- **Target Weight Streaming**: `{ws['total_weight_gb']} GB` (`{ws['target_linear_gb']} GB` decoder linears + `{ws['lm_head_gb']} GB` LM head) → `{ws['floor_share_pct']}%` of floor, `{ws['measured_share_pct']}%` of measured step",
        "",
        "## Per-Class Breakdown (Ranked by Gap to Floor)",
        "",
        "| Class | Bytes (GB) | Floor (ms) | Measured (ms) | Gap (ms) | Slowdown vs Floor | % of Step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(report["classes"].items(), key=lambda kv: kv[1]["gap_ms"], reverse=True)
    for cls, row in ranked:
        svf = f"{row['slowdown_vs_floor']}x" if row["slowdown_vs_floor"] is not None else "-"
        lines.append(
            f"| `{cls}` | {row['bytes_gb']} | {row['floor_ms']} | {row['measured_ms']} | {row['gap_ms']} | {svf} | {row['share_of_step_pct']}% |"
        )
    th = report.get("tau_headroom")
    if th:
        lines.extend(
            [
                "",
                "## Acceptance Headroom (Exact Replay)",
                "",
                f"- **Observed `tau`**: `{th['tau_measured']}` | **Best-first `tau_best`**: `{th['tau_best']}` (`{th['tau_factor_best']}x`) | **Oracle `tau`**: `{th['tau_oracle']}` (`{th['tau_factor_oracle']}x`)",
            ]
        )
        if th.get("tau_best_by_budget"):
            curve = ", ".join(f"M={k}: {v}" for k, v in th["tau_best_by_budget"].items())
            lines.append(f"- **Budget curve**: {curve}")
        if th.get("lookahead_agreement_rate") is not None:
            lines.append(
                f"- **Lookahead agreement**: `{th['lookahead_agreement_rate']}` (est. loss: `{th['estimated_lookahead_loss_tokens']} tok/step`)"
            )
    return "\n".join(lines) + "\n"
