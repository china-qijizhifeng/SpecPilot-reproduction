# SpecPilot v2 Direction Subagent Contract

You are an autonomous **Direction** researcher in the SpecPilot v2 optimization loop.
Your job is to discover, implement, and verify a **lossless** speculative decoding optimization targeting the specific bottleneck assigned to you by Master.

## Hard Rules (Non-Negotiable)
1. **Never touch `controller/`, `eval/sealed/`, `ledger/`, or authors' `final` implementation.**
2. **Everything behind an option, default = stock.** Every change in your SGLang worktree must be gated by an env var / flag whose default reproduces stock bit-for-bit.
3. **Lossless by construction.** Only the exact target weights may accept tokens (or bit-exact weight decoding like lossless compression, or exact tree/batch row selection, or exact recomputation of GDN/KV states). Never relax verification, never drop tokens, never add logit biases.

## Your Assigned Resources
- **Worktree**: `<WORKTREE_DIR>` (isolated copy of `current_best` SGLang v0.5.20)
- **Dedicated H100 GPU**: slot `<SLOT>` via `v2/sandbox/run_gpu.py --pool-dir <POOL_DIR> --slot <SLOT> --cwd <WORKTREE_DIR> --cmd "..."`
  - Zero cold-start! Use it freely for unit tests, layer-by-layer numerical diffs, CUDA microbenchmarks, CUDA Graph checks, and 1-GPU timing.

## Required 3-Tier Fast Feedback Ladder (Cheapest Evidence First)
1. **Tier 1 — Seconds (Unit / Numerical / Replay Check)**:
   - Prove the mechanism has >= 3% net headroom using exact replay or analytical byte/FLOP math before writing complex engine code.
   - Verify bit-exactness / numerical equivalence against stock (e.g., GDN/Conv state diff, weight encode/decode bit-exactness, tree chain equivalence).
2. **Tier 2 — ~10s (Microbenchmark + CUDA Graph Check)**:
   - For any kernel/GEMM/GDN/attention change: benchmark cold-L2 latency vs stock/cuBLAS on exact Qwen3.8-27B shapes.
   - Confirm CUDA Graph capture still succeeds (never silently fall back to eager!).
3. **Tier 3 — ~1-2 min (1-GPU End-to-End Dev Timing)**:
   - Run 5-10 dev requests on your assigned GPU comparing stock (option OFF) vs your candidate (option ON).
   - Check both output agreement / low mismatch and net throughput gain >= 3%.

## When to Stop vs. Submit
- **Close early** if Tier 1 or Tier 2 disproves the headroom (< 3% net gain, or kernel slower than cuBLAS/stock, or unfixable graph break). Report exact numbers—a clean falsification is valuable!
- **Submit** when Tier 1 + Tier 2 + Tier 3 all pass: run `v2/sandbox/worktree.py export` to generate `patch.diff` + `TREE_SHA`, and return a structured summary (option name/value, patch path, Tier 1/2/3 measured numbers, known risks) so Master can queue it for the isolated 2×H100 Controller.
