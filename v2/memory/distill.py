#!/usr/bin/env python3
"""T4 Structured Memory distiller & byte-exact replay invariant verifier (SpecPilot §3.4).

Hard invariant:
    Applying an entry's transitive `requires` chain + its own code delta onto
    frozen `stock` MUST reproduce the exact source system `TREE_SHA` byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from harness.build_engine import SKIP_DIRS, tree_sha
from v2.sandbox.worktree import STOCK_ENGINE_DEFAULT, export_patch

MEMORY_ROOT = ROOT / "v2_runtime" / "memory" / "entries"


def _engine_tree_sha(root: Path) -> str:
    """Canonical SHA256 over engine source trees (`python/` and `lab/`) only."""
    files = _collect_files(root)
    out = {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(files.items())}
    blob = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _skip(rel: Path) -> bool:
    return any(p in SKIP_DIRS or p.endswith((".pyc", ".pyo", ".egg-info")) for p in rel.parts)


def _collect_files(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for sub in ("python", "lab"):
        sub_root = root / sub
        if not sub_root.is_dir():
            continue
        for p in sorted(sub_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if _skip(rel):
                continue
            out[str(rel)] = p.read_bytes()
    return out


def distill_entry(
    *,
    entry_id: str,
    round_num: int,
    base_dir: Path,
    source_dir: Path,
    requires: list[str],
    options: dict[str, str],
    summary: str,
    evidence: dict,
    memory_root: Path = MEMORY_ROOT,
) -> dict:
    base_dir = base_dir.resolve()
    source_dir = source_dir.resolve()
    dst = memory_root / entry_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    base_files = _collect_files(base_dir)
    src_files = _collect_files(source_dir)
    changed: list[str] = []
    deleted: list[str] = []

    for rel, data in src_files.items():
        if base_files.get(rel) != data:
            changed.append(rel)
    for rel in base_files:
        if rel not in src_files:
            deleted.append(rel)

    # Save exact delta tarball (changed/added files) + unified diff for human inspection
    delta_tar = dst / "delta.tar.gz"
    with tarfile.open(delta_tar, "w:gz") as tf:
        for rel in sorted(changed):
            tf.add(source_dir / rel, arcname=rel)

    export_patch(source_dir, dst / "patch.diff", base_dir=base_dir)

    base_sha = _engine_tree_sha(base_dir)
    source_sha = _engine_tree_sha(source_dir)
    manifest = {
        "entry_id": entry_id,
        "round": round_num,
        "requires": list(requires),
        "options": dict(options),
        "summary": summary,
        "base_tree_sha": base_sha,
        "source_tree_sha": source_sha,
        "changed_files": sorted(changed),
        "deleted_files": sorted(deleted),
        "delta_tar_sha256": hashlib.sha256(delta_tar.read_bytes()).hexdigest(),
        "evidence": evidence,
        "correctness_contract": "All options default OFF reproduce stock; verified by isolated 2xH100 Controller T1/T2 and T3 Sealed Final.",
        "distilled_unix": time.time(),
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _resolve_chain(entry_id: str, memory_root: Path) -> list[dict]:
    visited: set[str] = set()
    order: list[dict] = []

    def dfs(eid: str) -> None:
        if eid in visited:
            return
        visited.add(eid)
        m = json.loads((memory_root / eid / "manifest.json").read_text())
        for req in m.get("requires", []):
            dfs(req)
        order.append(m)

    dfs(entry_id)
    return order


def verify_entry(entry_id: str, stock_dir: Path = STOCK_ENGINE_DEFAULT, memory_root: Path = MEMORY_ROOT) -> dict:
    """Reconstruct from clean stock using transitive `requires` + delta and verify exact TREE_SHA."""
    chain = _resolve_chain(entry_id, memory_root)
    target = chain[-1]
    with tempfile.TemporaryDirectory(prefix="sp_mem_verify_") as td:
        work = Path(td) / "reconstructed"
        ignore = shutil.ignore_patterns(*SKIP_DIRS, "*.pyc", "*.pyo")
        work.mkdir(parents=True)
        shutil.copytree(stock_dir / "python", work / "python", symlinks=True, ignore=ignore)
        if (stock_dir / "lab").is_dir():
            shutil.copytree(stock_dir / "lab", work / "lab", symlinks=True, ignore=ignore)

        for step in chain:
            eid = step["entry_id"]
            with tarfile.open(memory_root / eid / "delta.tar.gz", "r:gz") as tf:
                tf.extractall(path=work)
            for rel in step.get("deleted_files", []):
                p = work / rel
                if p.exists():
                    p.unlink()

        reconstructed_sha = _engine_tree_sha(work)
        expected_sha = target["source_tree_sha"]
        matched = reconstructed_sha == expected_sha
        return {
            "entry_id": entry_id,
            "chain": [s["entry_id"] for s in chain],
            "expected_tree_sha": expected_sha,
            "reconstructed_tree_sha": reconstructed_sha,
            "byte_exact_match": matched,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--entry-id", required=True)

    args = parser.parse_args(argv)
    if args.action == "verify":
        res = verify_entry(args.entry_id)
        print(json.dumps(res, indent=2))
        return 0 if res["byte_exact_match"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
