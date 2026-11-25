#!/usr/bin/env python3
"""Candidate submission queue & isolated Controller dispatcher for SpecPilot v2.

Agents (Direction / Integration / Kernel) may ONLY submit patches here.
They never modify controller/, eval/sealed/, or ledger/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = ROOT / "v2_runtime" / "queue"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit_candidate(
    *,
    cand_id: str,
    direction: str,
    worktree: Path,
    patch_path: Path,
    option_env: dict[str, str],
    summary: str,
    local_metrics: dict,
    queue_root: Path = QUEUE_ROOT,
) -> dict:
    dst = queue_root / "pending" / cand_id
    if dst.exists():
        raise FileExistsError(f"candidate {cand_id!r} already exists at {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(patch_path, dst / "patch.diff")
    tree_sha = (worktree / "TREE_SHA").read_text().strip() if (worktree / "TREE_SHA").is_file() else ""
    record = {
        "cand_id": cand_id,
        "direction": direction,
        "worktree": str(worktree.resolve()),
        "tree_sha": tree_sha,
        "patch_sha256": _sha256(dst / "patch.diff"),
        "option_env": option_env,
        "summary": summary,
        "local_metrics": local_metrics,
        "submitted_unix": time.time(),
        "status": "pending",
    }
    (dst / "meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def list_queue(queue_root: Path = QUEUE_ROOT) -> dict:
    pending = []
    p_dir = queue_root / "pending"
    if p_dir.is_dir():
        for d in sorted(p_dir.iterdir()):
            meta = d / "meta.json"
            if meta.is_file():
                pending.append(json.loads(meta.read_text()))
    verdicts = []
    v_file = queue_root / "verdicts.jsonl"
    if v_file.is_file():
        for line in v_file.read_text().splitlines():
            if line.strip():
                verdicts.append(json.loads(line))
    return {"pending": pending, "verdicts": verdicts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--cand-id", required=True)
    s.add_argument("--direction", required=True)
    s.add_argument("--worktree", type=Path, required=True)
    s.add_argument("--patch", type=Path, required=True)
    s.add_argument("--option-env", type=str, default="{}", help="JSON dict of option env vars")
    s.add_argument("--summary", required=True)
    s.add_argument("--local-metrics", type=str, default="{}", help="JSON dict of Tier 1/2/3 metrics")

    sub.add_parser("list")

    args = parser.parse_args(argv)
    if args.action == "submit":
        rec = submit_candidate(
            cand_id=args.cand_id,
            direction=args.direction,
            worktree=args.worktree,
            patch_path=args.patch,
            option_env=json.loads(args.option_env),
            summary=args.summary,
            local_metrics=json.loads(args.local_metrics),
        )
        print(json.dumps(rec, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(list_queue(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
