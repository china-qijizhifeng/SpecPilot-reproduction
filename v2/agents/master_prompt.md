# SpecPilot v2 Blind Master Protocol

## Strict Isolation Rules
- Never read `docs/`, `HANDOFF.md`, `PLANS.md`, `README.md`, `eval/sealed/`, `memory/`, `ledger/`, or external papers/references. Never use web search.
- Never modify `controller/`, `eval/`, or `ledger/`.
- Every mechanism must be gated behind a default-OFF `SP_OPT_*` flag that reproduces `stock` identically when unset.

## Priority 0: Resolve Previous-Round T1-Rejected / Lossy Candidates First
Before proposing any new optimization directions for a new round:
1. **Check Previous-Round Controller Feedback**: If the previous round produced any candidate that passed performance (`t2_pass = true`) or showed strong Tier-3 speedup, **but failed T1 Token Forcing (`t1_pass = false`, e.g., `large_margin_events > 0` or `rate_pass = false`)**, **resolving that T1 failure is mandatory Priority 0**.
2. **Triage & Repair**:
   - Dispatch a dedicated Diagnostic/Repair Direction with the exact Controller T1 feedback (`mismatch_rate`, `large_margin_events`, `event_p`, worktree diff, `option_env`).
   - Require the Direction to classify the root cause via ablation & math:
     - **Fixable numerical / state / CUDA-graph / rollback bug** → fix cleanly, verify `0` large-margin events on full 30-req 2048-cap FP32 Token Forcing + default-OFF bit-exactness, and re-submit to the independent 2×H100 Controller (`T1 + T2 → T3 Sealed Final → T4 Memory`).
     - **Fundamentally lossy approximation** → prove mathematically and empirically why it cannot be made lossless without losing the gain, disable/archive that sub-mechanism, and preserve any lossless sub-components.
3. **Gate Before New Directions**:
   - Do **not** launch new exploratory directions for the next round until every high-gain T1-rejected candidate from the previous round is cleanly resolved (either promoted & sealed, or formally falsified as fundamentally lossy).

## Phase 0 Refresh: Re-Measure Full 3-Factor Ceiling Every Round
Before designing directions for any new round, always re-measure the **full 3-factor Ceiling** on the current promoted baseline (`v2/phase0/floors.py` + `v2/phase0/tau_headroom.py`):
1. **Acceptance factor (`tau_best / tau`)**: exact-replay acceptance ceiling across candidate budgets and any multi-pass disagreement loss.
2. **Kernel efficiency factor (`T_prof / T_floor`)**: per-class hardware byte/FLOP floors vs profiled GPU kernel time (including any unoptimized/uncompressed layers).
3. **System overhead & overlap factor (`T_step / T_prof`)**: explicit unattributed host glue, launch gaps, CPU-GPU syncs, and unoverlapped pipeline bubbles (`T_step - T_prof`).

## Phase 1: Ceiling-Driven Parallel Exploration
Only after Priority 0 is clear and Phase 0 Ceiling is refreshed:
1. Rank bottlenecks directly from the refreshed 3-factor Ceiling (do not rely on stale Round-0 numbers).
2. Propose orthogonal, hardware-grounded directions with explicit expected net gain, new cost, unknowns, and minimal falsification tests.
3. Dispatch parallel Directions on isolated worktrees over the persistent GPU pool (Tier 1 correctness → Tier 2 cold-L2 microbenchmark → Tier 3 1-GPU A/B).
4. **Stratified Generalization Check**: Always inspect both `easy` (high-acceptance) and `hard` (low-acceptance) dev buckets in Tier-3 (`bench_1gpu.py` `stratified` summary). Do not promote heuristics that only win on `easy` prompts while regressing on `hard` prompts (`hard_regression = true`) unless replaced by a genuinely input-adaptive mechanism.
5. **Theory-vs-Implementation Triage**: Never mark a direction "physically falsified" solely because a first prototype kernel was slow on Tier 2. Explicitly separate:
   - *Theory/floor insufficient* (hardware bytes/FLOPs or acceptance math cannot yield >= 3% net gain) → archive cleanly;
   - *Theory/floor strong (>= 8% lower bytes/FLOPs or higher tau), first implementation under-optimized* → flag for dedicated multi-iteration kernel refinement rather than discarding the idea.
6. Integrate winning lossless mechanisms into a single clean candidate and submit to the independent 2×H100 Controller.

## Plateau Escape & Co-Design Protocol (Anti-Local-Optimum Rule)
When a regular incremental round yields `< 3%` net gain (or when exploratory directions hit sharp latency cliffs upon changing a shape, capacity, or scheduling parameter):
1. **Do NOT stop immediately.** First run a mandatory **Assumption & Coupling Audit**:
   - Distinguish *true physical hardware walls* (bytes/FLOPs already near device floor across all classes AND `T_step - T_prof` negligible AND `tau` near `tau_best`) from *self-imposed implementation constraints* (hardcoded tile/block dimensions, fixed buffer/tree capacities, synchronous stage boundaries, or unoverlapped glue carried over from earlier rounds).
2. **Trigger a Co-Design Round** whenever a high-headroom opportunity is blocked by coupling across 2+ modules:
   - Define an explicit shared interface contract across the coupled modules;
   - Dispatch parallel Directions to update each coupled module to the new contract;
   - Evaluate the joint integrated system end-to-end (rather than rejecting one module in isolation against an incompatible legacy kernel).
3. **Principled Stop**: Only recommend stopping when all three Ceiling factors (`tau_factor`, `kernel_factor`, `overhead_factor`) are jointly near their physical limits after checking for cross-module coupling and unoverlapped system overhead.

## Optional Cross-Round & Cross-Experiment Memory Retrieval (Two-Stage)
When a campaign explicitly enables local Structured Memory retrieval (disabled during strict cold-start blindness audits):
1. **Stage 1 (Compact 1-Line Summary Index)**: Query `python3 v2/memory/store.py index --context-json '<4D_context>'` to scan both positive entries and conditional negative entries without bloating context.
2. **Stage 2 (On-Demand Expansion + Provenance)**: Expand only the 1–3 most relevant entries via `python3 v2/memory/store.py show --id <ID> --log-provenance <path>`. Any mechanism reused from Memory MUST be explicitly tagged `reused_from: <ID>` (never claimed as a fresh cold-start discovery).
3. **Anti-Negative-Transfer Discipline**:
   - `exact` (4/4 context match on `model`, `hardware`, `workload`, `dataset`): positive contracts and `theory_falsified` bounds can be trusted directly; check `reopen_condition` on `impl_underoptimized` negative entries before retrying with a new implementation strategy.
   - `partial` or `mismatch` (cross-experiment drift): treat old positive AND negative conclusions strictly as **hypotheses to re-verify**, never as immutable laws — previously failed ideas may succeed under the new regime, and previously winning ideas may regress.

