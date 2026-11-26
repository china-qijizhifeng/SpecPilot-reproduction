#!/usr/bin/env python3
"""Dual-Layer Structured Memory Store (Positive + Conditional Negative Entries).

Combines:
1. 4D Deployment Context Fingerprint (`model`, `hardware`, `workload`, `dataset`)
   to prevent Negative Memory Transfer across regimes (batch size, GPU arch, domain).
2. Hardware/System Bottleneck Signature tags (e.g. `hbm_bw_bound`, `sync_bubble`,
   `smem_bank_conflict`, `node_economics`, `numerical_stability`) for cross-experiment
   retrieval even when raw code patches cannot apply directly.
3. Two-Stage On-Demand Retrieval:
   - Stage 1 (`index`): compact 1-line summary table per entry (zero context bloat).
   - Stage 2 (`show`): expand full mechanism card / failure preconditions / reopen
     conditions / optional diff for a single chosen ID, with provenance logging.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MEMORY_BASE = ROOT / "v2_runtime" / "memory"
POSITIVE_DIR = MEMORY_BASE / "entries"
NEGATIVE_DIR = MEMORY_BASE / "negative_entries"

CONTEXT_DIMS = ("model", "hardware", "workload", "dataset")
VALID_FAILURE_CLASSES = ("theory_falsified", "impl_underoptimized", "t1_lossy_rejected")
VALID_STATUSES = ("active", "superseded", "resolved")

_FORBIDDEN_RE = re.compile(r"(eval/sealed/[^\s\"']+|\bsk-[A-Za-z0-9]{16,}|password\s*[:=])", re.IGNORECASE)


def _assert_safe_payload(payload: Any) -> None:
    blob = json.dumps(payload, ensure_ascii=False)
    m = _FORBIDDEN_RE.search(blob)
    if m:
        raise ValueError(f"Forbidden sealed path or credential pattern detected in memory payload: {m.group(0)!r}")


def match_context(entry_ctx: dict | None, query_ctx: dict | None) -> dict:
    """Compare an entry's 4D deployment context against the current task's context.

    Returns:
      - `match_level`: 'exact' (4/4), 'partial' (1..3/4), 'mismatch' (0/4), or 'unscoped'
      - `matched_dims`: int in 0..4
      - `diffs`: list of 'dim.field: old -> new' strings explaining every drift
    """
    if not query_ctx:
        return {"match_level": "unscoped", "matched_dims": 0, "diffs": []}
    if not entry_ctx:
        return {"match_level": "unscoped", "matched_dims": 0, "diffs": ["entry has no 4D context"]}

    matched_dims = 0
    diffs: list[str] = []
    present_dims = 0

    for dim in CONTEXT_DIMS:
        q_val = query_ctx.get(dim)
        if q_val is None:
            continue
        present_dims += 1
        e_val = entry_ctx.get(dim)
        if e_val is None:
            diffs.append(f"{dim}: missing -> {q_val}")
            continue
        if isinstance(q_val, dict) and isinstance(e_val, dict):
            dim_ok = True
            for k, qv in q_val.items():
                ev = e_val.get(k)
                if ev != qv:
                    dim_ok = False
                    diffs.append(f"{dim}.{k}: {ev} -> {qv}")
            if dim_ok:
                matched_dims += 1
        elif e_val == q_val:
            matched_dims += 1
        else:
            diffs.append(f"{dim}: {e_val} -> {q_val}")

    if present_dims == 0:
        level = "unscoped"
    elif matched_dims == present_dims and not diffs:
        level = "exact"
    elif matched_dims > 0:
        level = "partial"
    else:
        level = "mismatch"

    return {
        "match_level": level,
        "matched_dims": matched_dims,
        "total_dims": present_dims,
        "diffs": diffs,
    }


def record_negative_entry(
    *,
    neg_id: str,
    round_num: int,
    base_tree_sha: str,
    context: dict,
    bottleneck_tags: list[str],
    hypothesis: str,
    failure_class: str,
    failed_precondition: str,
    reopen_condition: str,
    one_line_summary: str,
    evidence: dict,
    status: str = "active",
    resolved_by: str | None = None,
    patch_text: str | None = None,
    memory_base: Path = MEMORY_BASE,
) -> dict:
    """Write a Conditional Negative Memory entry (structured error notebook)."""
    if failure_class not in VALID_FAILURE_CLASSES:
        raise ValueError(f"failure_class must be one of {VALID_FAILURE_CLASSES}, got {failure_class!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    for dim in CONTEXT_DIMS:
        if dim not in context:
            raise ValueError(f"4D context missing required dimension: {dim!r}")

    manifest = {
        "entry_id": neg_id,
        "kind": "negative",
        "round": round_num,
        "status": status,
        "resolved_by": resolved_by,
        "base_tree_sha": base_tree_sha,
        "context": context,
        "bottleneck_tags": sorted(set(bottleneck_tags)),
        "hypothesis": hypothesis,
        "failure_class": failure_class,
        "failed_precondition": failed_precondition,
        "reopen_condition": reopen_condition,
        "one_line_summary": one_line_summary,
        "evidence": evidence,
        "recorded_unix": time.time(),
    }
    _assert_safe_payload(manifest)
    if patch_text:
        _assert_safe_payload({"patch": patch_text})

    dst = memory_base / "negative_entries" / neg_id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if patch_text is not None:
        (dst / "patch.diff").write_text(patch_text)
    return manifest


def enrich_positive_entry(
    *,
    entry_id: str,
    context: dict,
    bottleneck_tags: list[str],
    one_line_summary: str,
    mechanism_card: dict | None = None,
    status: str = "active",
    memory_base: Path = MEMORY_BASE,
) -> dict:
    """Non-destructively attach 4D context & bottleneck tags to an existing positive entry.

    Does NOT touch delta.tar.gz or source_tree_sha, preserving byte-exact replay invariant.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    for dim in CONTEXT_DIMS:
        if dim not in context:
            raise ValueError(f"4D context missing required dimension: {dim!r}")

    mpath = memory_base / "entries" / entry_id / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["kind"] = "positive"
    manifest["status"] = status
    manifest["context"] = context
    manifest["bottleneck_tags"] = sorted(set(bottleneck_tags))
    manifest["one_line_summary"] = one_line_summary
    if mechanism_card is not None:
        manifest["mechanism_card"] = mechanism_card
    _assert_safe_payload(manifest)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _load_all_entries(memory_base: Path = MEMORY_BASE) -> list[tuple[dict, Path]]:
    items: list[tuple[dict, Path]] = []
    for kind, sub in (("positive", "entries"), ("negative", "negative_entries")):
        root = memory_base / sub
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            mpath = d / "manifest.json"
            if not mpath.is_file():
                continue
            m = json.loads(mpath.read_text())
            m.setdefault("kind", kind)
            m.setdefault("status", "active")
            items.append((m, d))
    return items


def build_index(
    *,
    query_context: dict | None = None,
    bottleneck_filter: list[str] | None = None,
    kind: str = "all",
    include_inactive: bool = False,
    memory_base: Path = MEMORY_BASE,
) -> list[dict]:
    """Stage-1 retrieval: build ranked 1-line summary index across positive & negative entries."""
    bset = set(bottleneck_filter or [])
    rows: list[dict] = []

    for m, _dir in _load_all_entries(memory_base):
        if kind != "all" and m["kind"] != kind:
            continue
        if not include_inactive and m.get("status", "active") != "active":
            continue

        tags = set(m.get("bottleneck_tags", []))
        tag_hits = len(tags & bset) if bset else 0
        if bset and tag_hits == 0:
            continue

        cm = match_context(m.get("context"), query_context)
        one_line = m.get("one_line_summary") or m.get("summary", "")[:120]
        rows.append(
            {
                "id": m["entry_id"],
                "kind": m["kind"],
                "round": m.get("round", 0),
                "status": m.get("status", "active"),
                "failure_class": m.get("failure_class"),
                "bottleneck_tags": sorted(tags),
                "tag_hits": tag_hits,
                "context_match": cm["match_level"],
                "matched_dims": cm["matched_dims"],
                "context_diffs": cm["diffs"],
                "one_line_summary": one_line,
            }
        )

    level_rank = {"exact": 3, "partial": 2, "unscoped": 1, "mismatch": 0}
    rows.sort(
        key=lambda r: (
            r["tag_hits"],
            level_rank.get(r["context_match"], 0),
            r["matched_dims"],
            r["round"],
        ),
        reverse=True,
    )
    return rows


def format_index_markdown(rows: list[dict]) -> str:
    """Render Stage-1 1-line summary index as compact Markdown for Master context."""
    lines = [
        "# Structured Memory Summary Index (Stage 1)",
        "",
        "Use `python3 v2/memory/store.py show --id <ID>` to inspect full details on demand.",
        "",
        "| ID | Kind | Status | Bottleneck | Context Match | Summary |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        tags = ", ".join(r["bottleneck_tags"]) or "-"
        cm = r["context_match"]
        if r["context_diffs"]:
            cm = f"{cm} ({'; '.join(r['context_diffs'][:2])})"
        kind_lbl = r["kind"] if not r.get("failure_class") else f"neg:{r['failure_class']}"
        lines.append(f"| `{r['id']}` | {kind_lbl} | {r['status']} | `{tags}` | {cm} | {r['one_line_summary']} |")
    return "\n".join(lines) + "\n"


def show_entry(
    *,
    entry_id: str,
    query_context: dict | None = None,
    include_patch: bool = False,
    provenance_log: Path | None = None,
    memory_base: Path = MEMORY_BASE,
) -> dict:
    """Stage-2 retrieval: expand full details of a single entry and optionally log provenance."""
    for m, d in _load_all_entries(memory_base):
        if m["entry_id"] == entry_id:
            cm = match_context(m.get("context"), query_context)
            out = dict(m)
            out["query_context_match"] = cm
            if include_patch and (d / "patch.diff").is_file():
                out["patch_diff"] = (d / "patch.diff").read_text()
            if provenance_log is not None:
                provenance_log.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "timestamp_unix": time.time(),
                    "reused_from": entry_id,
                    "kind": m["kind"],
                    "context_match": cm["match_level"],
                    "context_diffs": cm["diffs"],
                }
                with provenance_log.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return out
    raise KeyError(f"Memory entry not found: {entry_id!r}")


def update_entry_status(
    *,
    entry_id: str,
    status: str,
    resolved_by: str | None = None,
    superseded_by: str | None = None,
    reopen_condition: str | None = None,
    memory_base: Path = MEMORY_BASE,
) -> dict:
    """Controller/maintainer-only lifecycle update for positive or negative entries.

    Only mutates governance metadata (`status`, `resolved_by`, `superseded_by`,
    `reopen_condition`). Never modifies immutable evidence, delta.tar.gz, or TREE_SHA.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    for m, d in _load_all_entries(memory_base):
        if m["entry_id"] == entry_id:
            m["status"] = status
            if resolved_by is not None:
                m["resolved_by"] = resolved_by
            if superseded_by is not None:
                m["superseded_by"] = superseded_by
            if reopen_condition is not None:
                m["reopen_condition"] = reopen_condition
            m["updated_unix"] = time.time()
            _assert_safe_payload(m)
            (d / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
            return m
    raise KeyError(f"Memory entry not found: {entry_id!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-base", type=Path, default=MEMORY_BASE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Stage 1: print compact 1-line summary index")
    p_idx.add_argument("--context-json", type=str, default=None)
    p_idx.add_argument("--bottleneck", action="append", default=[])
    p_idx.add_argument("--kind", choices=("all", "positive", "negative"), default="all")
    p_idx.add_argument("--include-inactive", action="store_true")
    p_idx.add_argument("--json", action="store_true", dest="as_json")

    p_show = sub.add_parser("show", help="Stage 2: expand full details of one entry")
    p_show.add_argument("--id", required=True)
    p_show.add_argument("--context-json", type=str, default=None)
    p_show.add_argument("--include-patch", action="store_true")
    p_show.add_argument("--log-provenance", type=Path, default=None)

    p_upd = sub.add_parser("update-status", help="Maintainer/Controller: update lifecycle status without altering evidence")
    p_upd.add_argument("--id", required=True)
    p_upd.add_argument("--status", choices=VALID_STATUSES, required=True)
    p_upd.add_argument("--resolved-by", default=None)
    p_upd.add_argument("--superseded-by", default=None)
    p_upd.add_argument("--reopen-condition", default=None)

    args = parser.parse_args(argv)
    q_ctx = json.loads(args.context_json) if args.context_json else None

    if args.cmd == "index":
        rows = build_index(
            query_context=q_ctx,
            bottleneck_filter=args.bottleneck,
            kind=args.kind,
            include_inactive=args.include_inactive,
            memory_base=args.memory_base,
        )
        if args.as_json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            print(format_index_markdown(rows), end="")
        return 0

    if args.cmd == "show":
        detail = show_entry(
            entry_id=args.id,
            query_context=q_ctx,
            include_patch=args.include_patch,
            provenance_log=args.log_provenance,
            memory_base=args.memory_base,
        )
        print(json.dumps(detail, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "update-status":
        updated = update_entry_status(
            entry_id=args.id,
            status=args.status,
            resolved_by=args.resolved_by,
            superseded_by=args.superseded_by,
            reopen_condition=args.reopen_condition,
            memory_base=args.memory_base,
        )
        print(json.dumps(updated, indent=2, ensure_ascii=False))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
