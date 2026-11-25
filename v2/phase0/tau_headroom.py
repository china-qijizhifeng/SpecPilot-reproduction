#!/usr/bin/env python3
"""Model-agnostic exact-replay acceptance headroom (tau_best) analyzer (SpecPilot §3.1).

Computes the numerator factor (`tau_best / tau`) of the 3-factor Ceiling from
observed per-step draft candidate probabilities and target greedy token matches:
- `tau_measured`: observed mean accepted tokens per verify step
- `tau_best_by_budget`: exact-replay best-first tree expected acceptance across
  candidate budgets M (e.g. 8, 16, 32, 64, 96)
- `tau_oracle`: upper bound if every position matching any top-K draft candidate
  were verified
- `lookahead_loss_tokens`: estimated acceptance loss from approximate lookahead
  disagreement when multi-pass lookahead telemetry is present

This module is completely model- and hardware-agnostic and never reads sealed data.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Iterable


def best_first_tree_acceptance(
    pos_top_probs: list[list[float]],
    pos_match_ranks: list[int],
    budget: int,
) -> int:
    """Exact replay of a greedy best-first candidate tree of size `budget` on one step.

    Args:
        pos_top_probs: For each draft depth d = 0..D-1, descending top-K candidate
            probabilities [p_{d,0}, p_{d,1}, ...] at depth d.
        pos_match_ranks: For each depth d = 0..D-1, the 0-based rank r where the
            target's greedy token appears in the draft top-K list, or -1 if absent.
        budget: Total tree node budget M (including root at depth 0).

    Returns:
        Number of accepted tokens (>= 1, since the target always produces 1 bonus token
        at the first mismatch / leaf).
    """
    if budget <= 1 or not pos_top_probs:
        return 1

    max_depth = len(pos_top_probs)
    # Priority queue of (-cumulative_prob, depth, is_on_target_path)
    # Root (depth=0) is always verified; we expand children at depths 0..max_depth-1.
    # Each node at depth d represents choosing rank r for position d along a prefix.
    pq: list[tuple[float, int, bool]] = []
    # Seed depth-0 candidates
    for r, p in enumerate(pos_top_probs[0]):
        if p <= 0.0:
            continue
        on_target = r == pos_match_ranks[0]
        heapq.heappush(pq, (-float(p), 0, on_target))

    used = 1  # root slot
    accepted_prefix = 0
    # Track whether depth d on the true target path was included in the selected tree
    target_selected: dict[int, bool] = {}

    while pq and used < budget:
        neg_prob, depth, on_target = heapq.heappop(pq)
        used += 1
        prob = -neg_prob
        if on_target:
            target_selected[depth] = True
        next_d = depth + 1
        if next_d < max_depth:
            for r, p_child in enumerate(pos_top_probs[next_d]):
                if p_child <= 0.0:
                    continue
                child_on_target = on_target and (r == pos_match_ranks[next_d])
                heapq.heappush(pq, (-(prob * float(p_child)), next_d, child_on_target))

    # Count contiguous accepted prefix along the true target path
    for d in range(max_depth):
        if target_selected.get(d, False):
            accepted_prefix += 1
        else:
            break

    # +1 for the guaranteed bonus token produced by the target model
    return accepted_prefix + 1


def oracle_step_acceptance(pos_match_ranks: list[int]) -> int:
    """Oracle acceptance length if every contiguous position present in draft top-K is hit."""
    acc = 0
    for r in pos_match_ranks:
        if r >= 0:
            acc += 1
        else:
            break
    return acc + 1


def analyze_tau_headroom(
    steps: Iterable[dict],
    budgets: tuple[int, ...] = (8, 16, 32, 64, 96),
    lookahead_agreement_rate: float | None = None,
) -> dict:
    """Compute exact-replay acceptance ceiling curve over a sequence of step traces.

    Each step dict supports:
      - `accepted_tokens` (int): actual tokens committed on that step
      - `pos_top_probs` (list[list[float]]): per-depth top-K draft probabilities
      - `pos_match_ranks` (list[int]): per-depth rank of target greedy token (-1 if miss)
    """
    step_list = list(steps)
    if not step_list:
        raise ValueError("steps must be non-empty")

    n = len(step_list)
    tau_measured = sum(int(s["accepted_tokens"]) for s in step_list) / n
    oracle_vals = [oracle_step_acceptance(s["pos_match_ranks"]) for s in step_list]
    tau_oracle = sum(oracle_vals) / n

    by_budget: dict[str, float] = {}
    for b in budgets:
        vals = [
            best_first_tree_acceptance(s["pos_top_probs"], s["pos_match_ranks"], budget=int(b))
            for s in step_list
        ]
        by_budget[str(b)] = round(sum(vals) / n, 4)

    tau_best = max(by_budget.values()) if by_budget else round(tau_oracle, 4)

    la_loss = None
    if lookahead_agreement_rate is not None:
        agree = max(0.0, min(1.0, float(lookahead_agreement_rate)))
        # Approximate expected tokens lost to lookahead disagreement when chaining
        la_loss = round(max(tau_best - tau_measured, 0.0) * (1.0 - agree) / max(1.0 - agree + 0.05, 1e-6), 4)

    return {
        "num_steps": n,
        "tau_measured": round(tau_measured, 4),
        "tau_best": round(tau_best, 4),
        "tau_oracle": round(tau_oracle, 4),
        "tau_best_by_budget": by_budget,
        "tau_factor_best": round(tau_best / max(tau_measured, 1e-9), 4),
        "tau_factor_oracle": round(tau_oracle / max(tau_measured, 1e-9), 4),
        "lookahead_agreement_rate": round(lookahead_agreement_rate, 4) if lookahead_agreement_rate is not None else None,
        "estimated_lookahead_loss_tokens": la_loss,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="8,16,32,64,96")
    parser.add_argument("--lookahead-agreement", type=float, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    steps = [json.loads(line) for line in args.trace_jsonl.read_text().splitlines() if line.strip()]
    res = analyze_tau_headroom(steps, budgets=budgets, lookahead_agreement_rate=args.lookahead_agreement)
    text = json.dumps(res, indent=2, ensure_ascii=False) + "\n"
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
