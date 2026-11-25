#!/usr/bin/env python3
"""Isolated SGLang engine worktree manager for parallel Direction agents (pure Python, no git dependency)."""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
from pathlib import Path

from harness.build_engine import SKIP_DIRS, tree_sha

STOCK_ENGINE_DEFAULT = Path("/gpfs/users/jixinpeng/specpilot_data/work/engines/h100_c1_qwen38-27b_dflash2-stock")


def _skip(rel: Path) -> bool:
    return any(p in SKIP_DIRS or p.endswith((".pyc", ".pyo", ".egg-info")) for p in rel.parts)


def create_worktree(base_dir: Path, dst_dir: Path, force: bool = False) -> dict:
    base_dir = base_dir.resolve()
    dst_dir = dst_dir.resolve()
    if not (base_dir / "python").is_dir():
        raise FileNotFoundError(f"base engine missing python/: {base_dir}")
    if dst_dir.exists():
        if not force:
            raise FileExistsError(f"destination already exists: {dst_dir}")
        shutil.rmtree(dst_dir)

    ignore = shutil.ignore_patterns(*SKIP_DIRS, "*.pyc", "*.pyo")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_dir / "python", dst_dir / "python", symlinks=True, ignore=ignore)
    if (base_dir / "lab").is_dir():
        shutil.copytree(base_dir / "lab", dst_dir / "lab", symlinks=True, ignore=ignore)

    sha = tree_sha(dst_dir)
    (dst_dir / "TREE_SHA").write_text(sha + "\n")
    (dst_dir / ".worktree_meta.json").write_text(
        json.dumps({"base_dir": str(base_dir), "base_tree_sha": sha}, indent=2) + "\n"
    )
    return {"base_dir": str(base_dir), "worktree": str(dst_dir), "base_tree_sha": sha}


def refresh_tree_sha(worktree_dir: Path) -> str:
    # Exclude .worktree_meta.json from TREE_SHA comparison by computing on python/ and lab/
    meta = worktree_dir / ".worktree_meta.json"
    saved = meta.read_text() if meta.is_file() else None
    if meta.is_file():
        meta.unlink()
    try:
        sha = tree_sha(worktree_dir)
        (worktree_dir / "TREE_SHA").write_text(sha + "\n")
    finally:
        if saved is not None:
            meta.write_text(saved)
    return sha


def export_patch(worktree_dir: Path, out_patch: Path, base_dir: Path | None = None) -> dict:
    worktree_dir = worktree_dir.resolve()
    if base_dir is None:
        meta = json.loads((worktree_dir / ".worktree_meta.json").read_text())
        base_dir = Path(meta["base_dir"])
    base_dir = base_dir.resolve()

    def collect_files(root: Path) -> dict[str, Path]:
        out = {}
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
                out[str(rel)] = p
        return out

    base_files = collect_files(base_dir)
    curr_files = collect_files(worktree_dir)
    all_keys = sorted(set(base_files) | set(curr_files))

    diffs: list[str] = []
    dirty: list[str] = []
    for rel in all_keys:
        b = base_files.get(rel)
        c = curr_files.get(rel)
        b_bytes = b.read_bytes() if b else b""
        c_bytes = c.read_bytes() if c else b""
        if b_bytes == c_bytes:
            continue
        dirty.append(rel)
        b_lines = b_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
        c_lines = c_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
        diffs.extend(
            difflib.unified_diff(
                b_lines,
                c_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )

    diff_text = "".join(diffs)
    out_patch.parent.mkdir(parents=True, exist_ok=True)
    out_patch.write_text(diff_text)
    new_sha = refresh_tree_sha(worktree_dir)
    return {
        "worktree": str(worktree_dir),
        "patch_path": str(out_patch),
        "patch_bytes": len(diff_text.encode("utf-8")),
        "tree_sha": new_sha,
        "dirty_files": dirty,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    c = sub.add_parser("create")
    c.add_argument("--base", type=Path, default=STOCK_ENGINE_DEFAULT)
    c.add_argument("--dst", type=Path, required=True)
    c.add_argument("--force", action="store_true")

    e = sub.add_parser("export")
    e.add_argument("--worktree", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.action == "create":
        print(json.dumps(create_worktree(args.base, args.dst, force=args.force), indent=2))
    else:
        print(json.dumps(export_patch(args.worktree, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
