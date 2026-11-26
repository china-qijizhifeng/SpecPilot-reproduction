#!/usr/bin/env python3
"""Single-GPU Phase 0 measurement script for SpecPilot v2.

Runs directly inside a persistent GPU pool slot (zero NorthJob cold-start):
1. Measures clean unprofiled AR step time (T_AR) and stock/candidate step time (T_step, tau, tok/s)
   on representative dev prompts via SGLang HTTP server.
2. Runs a standalone PyTorch/SGLang micro-profile of the exact 6 kernel classes on the same GPU
   using CUDA events (target_gemm, lm_head, gdn_recurrent, attention, drafter, other) so 100% of
   step time is cleanly attributed.
3. Combines with exact-replay tau(N) curve to emit ceiling_report.json and ceiling_brief.md.
"""
from __future__ import annotations

import argparse
import json
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

from v2.phase0.floors import compute_ceiling_report, format_master_brief


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(port: int, timeout_s: float = 600.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"server on port {port} did not become healthy within {timeout_s}s")


def _http_post_json(url: str, payload: dict, timeout_s: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_dev_prompts(per_suite: int = 2) -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/gpfs/models/huggingface.co/Qwen/Qwen3.8-27B", local_files_only=True)
    dev_dir = ROOT / "assets/upstream/dev"
    suites = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
    prompts = []
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
                prompts.append({"id": item["id"], "suite": suite, "input_ids": list(ids)})
                count += 1
                if count >= per_suite:
                    break
    return prompts


def _start_server(engine_dir: Path, port: int, nccl_port: int, mode: str, draft_tokens: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["SP_ROOT"] = str(ROOT)
    env["SP_ENGINE"] = str(engine_dir)
    env["PORT"] = str(port)
    env["NCCL_PORT"] = str(nccl_port)
    env["DRAFT_TOKENS"] = str(draft_tokens)

    if mode == "ar":
        cmd = (
            f'source "{ROOT}/harness/sglang/env.sh" && '
            f'exec "$SP_PY" -m sglang.launch_server '
            f'--model-path "$SP_TARGET" --dtype bfloat16 --kv-cache-dtype auto '
            f'--mamba-ssm-dtype float32 --attention-backend fa3 --mem-fraction-static 0.90 '
            f'--max-running-requests 4 --cuda-graph-max-bs-decode 4 --chunked-prefill-size 8192 '
            f'--disable-radix-cache --random-seed 0 --host 127.0.0.1 --port {port} --nccl-port {nccl_port}'
        )
    else:
        cmd = f'bash "{ROOT}/harness/sglang/serve.sh"'

    log_fh = log_path.open("wb")
    proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    fh = getattr(proc, "_log_fh", None)
    if fh:
        fh.close()


def _make_sampling_params(
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> dict:
    if float(temperature) <= 0.0:
        return {"temperature": 0.0, "top_k": 1, "max_new_tokens": int(max_new_tokens)}
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }


def _benchmark_server(
    port: int,
    prompts: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> dict:
    url = f"http://127.0.0.1:{port}/generate"
    # Warmup (2 short requests)
    for p in prompts[:2]:
        _http_post_json(
            url,
            {
                "input_ids": p["input_ids"],
                "sampling_params": _make_sampling_params(32, temperature=temperature, top_p=top_p, top_k=top_k),
            },
        )

    rows = []
    total_tokens = 0
    total_steps = 0
    total_decode_s = 0.0
    for p in prompts:
        t0 = time.perf_counter()
        resp = _http_post_json(
            url,
            {
                "input_ids": p["input_ids"],
                "sampling_params": _make_sampling_params(max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k),
            },
        )
        dt = time.perf_counter() - t0
        meta = resp.get("meta_info", {})
        comp_toks = int(meta.get("completion_tokens", len(resp.get("output_ids", []))))
        # SGLang returns spec_verify_ct in meta_info for speculative decoding
        verify_ct = int(meta.get("spec_verify_ct", 0))
        e2e_lat = float(meta.get("e2e_latency", dt))
        rows.append(
            {
                "id": p["id"],
                "suite": p["suite"],
                "prompt_len": len(p["input_ids"]),
                "completion_tokens": comp_toks,
                "verify_ct": verify_ct,
                "e2e_s": round(e2e_lat, 4),
                "tok_per_s": round(comp_toks / max(e2e_lat, 1e-6), 2),
                "tau": round(comp_toks / verify_ct, 4) if verify_ct > 0 else None,
            }
        )
        total_tokens += comp_toks
        total_steps += verify_ct
        total_decode_s += e2e_lat

    avg_tps = total_tokens / max(total_decode_s, 1e-6)
    avg_tau = (total_tokens / total_steps) if total_steps > 0 else 1.0
    avg_step_ms = (1000.0 * total_decode_s / total_steps) if total_steps > 0 else (1000.0 / max(avg_tps, 1e-6))
    return {
        "rows": rows,
        "total_tokens": total_tokens,
        "total_steps": total_steps,
        "total_s": round(total_decode_s, 4),
        "avg_tok_per_s": round(avg_tps, 2),
        "avg_tau": round(avg_tau, 4),
        "avg_step_ms": round(avg_step_ms, 4),
    }


def _microprofile_classes_cuda(verify_rows: int = 8, draft_block: int = 8, kv_len: int = 1024) -> dict[str, float]:
    """Directly time BF16 cuBLAS GEMMs + LM head + GDN/Attention/Drafter shapes on this GPU.

    Uses exact Qwen3.8-27B & DFlash2 shapes with cold-L2 flushing so we have an
    exact, reproducible per-class measured GPU time (no unattributed nvjet_* mystery).
    """
    import torch

    tc = json.loads(Path("/gpfs/models/huggingface.co/Qwen/Qwen3.8-27B/config.json").read_text())["text_config"]
    dc = json.loads(Path("/gpfs/users/jixinpeng/models/z-lab/Qwen3.8-27B-DFlash2/config.json").read_text())
    h = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    vocab = int(tc["vocab_size"])
    n_gdn = sum(1 for t in tc["layer_types"] if t == "linear_attention")
    n_attn = sum(1 for t in tc["layer_types"] if t == "full_attention")
    n_layers = n_gdn + n_attn

    # Cold-L2 flush buffer (128 MB > H100 50MB L2)
    l2_flush = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")

    def time_mm(m: int, k: int, n: int, iters: int = 20) -> float:
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        # warmup
        for _ in range(5):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            l2_flush.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.nn.functional.linear(x, w)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        del x, w
        times.sort()
        return times[len(times) // 2]

    # Target decoder linear projections
    t_gate_up = time_mm(verify_rows, h, 2 * inter)
    t_down = time_mm(verify_rows, inter, h)
    t_gdn_qkvz = time_mm(verify_rows, h, 16384)
    t_gdn_ba = time_mm(verify_rows, h, 96)
    t_gdn_out = time_mm(verify_rows, 6144, h)
    t_attn_qkv = time_mm(verify_rows, h, 12288 + 1024 + 1024)
    t_attn_o = time_mm(verify_rows, 6144, h)

    target_gemm_ms = (
        n_layers * (t_gate_up + t_down)
        + n_gdn * (t_gdn_qkvz + t_gdn_ba + t_gdn_out)
        + n_attn * (t_attn_qkv + t_attn_o)
    )
    lm_head_ms = time_mm(verify_rows, h, vocab)

    # Drafter (5 layers + fc + lm_head)
    dh = int(dc["hidden_size"])
    dinter = int(dc["intermediate_size"])
    t_d_qkv = time_mm(draft_block, dh, 4096 + 2048)
    t_d_o = time_mm(draft_block, 4096, dh)
    t_d_mlp = time_mm(draft_block, dh, 2 * dinter) + time_mm(draft_block, dinter, dh)
    t_d_fc = time_mm(draft_block, 5 * dh, dh)
    drafter_ms = 5 * (t_d_qkv + t_d_o + t_d_mlp) + t_d_fc + time_mm(draft_block, h, vocab)

    # GDN recurrent state update (real SGLang/FLA fused Triton kernel across 48 layers)
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule

    gq = torch.randn((1, verify_rows, 16, 128), dtype=torch.bfloat16, device="cuda")
    gk = torch.randn((1, verify_rows, 16, 128), dtype=torch.bfloat16, device="cuda")
    gv = torch.randn((1, verify_rows, 48, 128), dtype=torch.bfloat16, device="cuda")
    gg = torch.randn((1, verify_rows, 48), dtype=torch.float32, device="cuda") * -0.1
    gbeta = torch.sigmoid(torch.randn((1, verify_rows, 48), dtype=torch.bfloat16, device="cuda"))
    gstate = torch.randn((1, 48, 128, 128), dtype=torch.float32, device="cuda")
    for _ in range(5):
        fused_recurrent_gated_delta_rule(
            q=gq, k=gk, v=gv, g=gg, beta=gbeta, initial_state=gstate, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
    torch.cuda.synchronize()
    gdn_times = []
    for _ in range(25):
        l2_flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fused_recurrent_gated_delta_rule(
            q=gq, k=gk, v=gv, g=gg, beta=gbeta, initial_state=gstate, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        e.record()
        torch.cuda.synchronize()
        gdn_times.append(s.elapsed_time(e))
    gdn_times.sort()
    gdn_recurrent_ms = n_gdn * gdn_times[len(gdn_times) // 2]

    # Attention KV read proxy (16 layers x 4 kv heads x 256 dim x kv_len)
    q = torch.randn((1, 24, verify_rows, 256), dtype=torch.bfloat16, device="cuda")
    k_c = torch.randn((1, 24, kv_len, 256), dtype=torch.bfloat16, device="cuda")
    v_c = torch.randn((1, 24, kv_len, 256), dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        torch.nn.functional.scaled_dot_product_attention(q, k_c, v_c)
    torch.cuda.synchronize()
    s.record()
    torch.nn.functional.scaled_dot_product_attention(q, k_c, v_c)
    e.record()
    torch.cuda.synchronize()
    attention_ms = n_attn * s.elapsed_time(e)

    torch.cuda.empty_cache()
    return {
        "target_gemm": round(target_gemm_ms, 4),
        "lm_head": round(lm_head_ms, 4),
        "gdn_recurrent": round(gdn_recurrent_ms, 4),
        "attention": round(attention_ms, 4),
        "drafter": round(drafter_ms, 4),
        "detail_per_layer_ms": {
            "gate_up": round(t_gate_up, 4),
            "down": round(t_down, 4),
            "gdn_qkvz": round(t_gdn_qkvz, 4),
            "gdn_ba": round(t_gdn_ba, 4),
            "gdn_out": round(t_gdn_out, 4),
            "attn_qkv": round(t_attn_qkv, 4),
            "attn_o": round(t_attn_o, 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tau-best-replay", type=float, default=6.42, help="Best-first tree / oracle tau from exact replay")
    parser.add_argument("--skip-ar", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_dev_prompts(per_suite=args.per_suite)

    # 1. Cold-L2 class microprofile (before server takes 90% VRAM)
    print("[phase0] running cold-L2 per-class CUDA microprofile...", flush=True)
    class_profile = _microprofile_classes_cuda(verify_rows=args.draft_tokens, draft_block=args.draft_tokens)
    (args.out_dir / "class_profile.json").write_text(json.dumps(class_profile, indent=2) + "\n")

    # 2. Measure stock/candidate server
    port = _free_port()
    nccl_port = _free_port()
    print(f"[phase0] starting speculative server on port {port} (temp={args.temperature})...", flush=True)
    proc = _start_server(args.engine_dir, port, nccl_port, "spec", args.draft_tokens, args.out_dir / "spec_server.log")
    try:
        _wait_health(port)
        spec_bench = _benchmark_server(
            port,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    finally:
        _stop_server(proc)
    (args.out_dir / "spec_bench.json").write_text(json.dumps(spec_bench, indent=2) + "\n")

    # 3. Optional AR baseline
    t_ar_ms = None
    if not args.skip_ar:
        port = _free_port()
        nccl_port = _free_port()
        print(f"[phase0] starting AR server on port {port}...", flush=True)
        proc = _start_server(args.engine_dir, port, nccl_port, "ar", args.draft_tokens, args.out_dir / "ar_server.log")
        try:
            _wait_health(port)
            ar_bench = _benchmark_server(
                port,
                prompts[:5],
                max_new_tokens=min(args.max_new_tokens, 256),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            t_ar_ms = ar_bench["avg_step_ms"]
            (args.out_dir / "ar_bench.json").write_text(json.dumps(ar_bench, indent=2) + "\n")
        finally:
            _stop_server(proc)

    measured_classes = {k: v for k, v in class_profile.items() if isinstance(v, (int, float))}
    report = compute_ceiling_report(
        measured_step_ms=spec_bench["avg_step_ms"],
        measured_class_ms=measured_classes,
        measured_tau=spec_bench["avg_tau"],
        tau_best=args.tau_best_replay,
        t_ar_ms=t_ar_ms,
        verify_rows=args.draft_tokens,
        draft_block=args.draft_tokens,
    )
    (args.out_dir / "ceiling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    brief = format_master_brief(report)
    (args.out_dir / "ceiling_brief.md").write_text(brief)
    print(brief, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
