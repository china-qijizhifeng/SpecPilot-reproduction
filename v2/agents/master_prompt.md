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
   - Once resolved, **re-measure the 3-factor Ceiling** on the updated promoted baseline before deciding whether sufficient headroom remains to propose new directions.

## Phase 1: Ceiling-Driven Parallel Exploration
Only after Priority 0 is clear:
1. Inspect the updated 3-factor Ceiling breakdown (target weight streaming, GDN/attention verify, drafter overhead, acceptance length $\tau$, launch/sync overhead).
2. Propose orthogonal, hardware-grounded directions with explicit expected net gain, new cost, unknowns, and minimal falsification tests.
3. Dispatch parallel Directions on isolated worktrees over the persistent GPU pool (Tier 1 kernel correctness → Tier 2 microbenchmark → Tier 3 1-GPU A/B).
4. Integrate winning lossless mechanisms into a single clean candidate and submit to the independent 2×H100 Controller.
