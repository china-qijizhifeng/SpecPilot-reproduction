# SpecPilot v2 — Autonomous Systems & Algorithm Discovery Framework

`v2/` is a general, model- and hardware-agnostic autonomous optimization loop for LLM speculative decoding (and general inference systems), structured into four clean, isolated layers:

## Architecture

```
v2/
├── phase0/                  # Dual-end 3-Factor Ceiling (T_floor + T_prof/T_step + exact-replay tau_best)
│   ├── floors.py            # Device presets (H100/H200/B200/PRO-6000/Ascend-910B2C), hybrid & dense Transformer floors
│   ├── tau_headroom.py      # Model-agnostic exact-replay acceptance ceiling across candidate budgets
│   └── measure_stock.py     # Baseline profiling & ceiling brief generator
├── agents/                  # Role protocols (strictly blind, zero solution hints)
│   ├── master_prompt.md     # Priority 0 (T1 repair) + Phase 0 Refresh + Plateau Escape & Co-Design + 2-Stage Memory
│   └── direction_prompt.md  # 3-Tier fast feedback ladder (Tier 1 exactness -> Tier 2 cold-L2 -> Tier 3 stratified A/B)
├── sandbox/                 # Zero-cold-start multi-device exploration pool & worktree isolation
│   ├── pool_worker.py       # Persistent N-GPU GPFS inbox worker (0.25s poll, heartbeats, timeout/cancel)
│   ├── run_gpu.py           # Client CLI for submitting jobs to dedicated pool slots
│   ├── worktree.py          # Pure-Python isolated worktree + unified diff & TREE_SHA exporter
│   ├── bench_1gpu.py        # Tier-3 A/B benchmark with zero-overhead Easy/Hard stratified buckets
│   └── northjob_pool.py     # Cluster launcher adapter (read-only model PVC, dedup, dry-run default)
├── memory/                  # Dual-Layer Structured Memory (Cross-Round + Cross-Experiment)
│   ├── distill.py           # Positive entry distillation + byte-exact TREE_SHA replay verification from stock
│   └── store.py             # 4D Context Fingerprint (model/hardware/workload/dataset) + Conditional Negative Memory + 2-stage CLI
├── controller_queue.py      # Agent-facing read/submit queue (agents never touch controller/ or eval/sealed/)
├── controller_verify_2gpu.py# Isolated 2-GPU swapped-device Controller (T1 FP32 Token Forcing + T2 paired CI)
├── controller_sealed_final.py# Phase 4 Sealed Final (strictly unseen requests + 2048 & 4096 Token Forcing)
└── tests/
    └── test_v2_core.py      # Offline CPU unit tests for all core invariants
```

## Core Design Principles

1. **All Changes Behind Default-OFF Options**: Every optimization is gated behind an environment flag whose default reproduces `stock` bit-for-bit.
2. **Physical Separation of Explorer vs. Judge**: Exploration agents run on isolated worktrees and persistent pools; promotion verdicts and sealed evaluations run exclusively in dedicated Controller jobs without agent write access.
3. **Priority 0 (Lossy T1 Repair First)**: Any high-gain candidate rejected by Token Forcing is diagnosed and resolved before opening new exploratory directions.
4. **Plateau Escape & Stratified Generalization**: When incremental rounds yield `< 3%`, Master audits self-imposed couplings (Co-Design) and checks both `easy` and `hard` dev buckets to prevent overfitting.
5. **Context-Scoped Dual-Layer Memory**: Both positive and negative entries carry a 4D deployment context (`model`, `hardware`, `workload`, `dataset`) + hardware bottleneck tags, retrieved via Stage-1 1-line summary index (`store.py index`) and Stage-2 on-demand expansion (`store.py show --log-provenance`).

## Quick Commands

```bash
# Run all v2 CPU unit tests
python3 -m unittest discover -s v2/tests -p 'test_*.py' -v

# Stage 1: scan 1-line Structured Memory index against current 4D context
python3 v2/memory/store.py index --context-json '{"model":{"target":"Qwen3.8-27B"},"hardware":{"sm":"sm_90"},"workload":{"batch_size":1},"dataset":{"suite":"5-suite-dev-en"}}'

# Stage 2: expand a single memory entry on demand (with provenance logging)
python3 v2/memory/store.py show --id r6_la_pipeline_ewma --log-provenance /tmp/provenance.jsonl

# Verify byte-exact transitive reconstruction from clean stock
python3 v2/memory/distill.py verify --entry-id r6_la_pipeline_ewma
```
