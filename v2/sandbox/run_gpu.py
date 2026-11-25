#!/usr/bin/env python3
"""Client CLI to submit commands to a persistent GPU exploration pool slot."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


def pool_status(pool_dir: Path) -> dict:
    meta_path = pool_dir / "pool_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    slots = {}
    slots_root = pool_dir / "slots"
    now = time.time()
    if slots_root.is_dir():
        for slot_dir in sorted(slots_root.glob("gpu*")):
            hb_path = slot_dir / "heartbeat.json"
            if hb_path.is_file():
                hb = json.loads(hb_path.read_text())
                hb["age_s"] = round(now - hb.get("updated_unix", 0), 1)
                hb["alive"] = hb["age_s"] < 30.0 and hb.get("state") != "stopped"
                slots[slot_dir.name] = hb
    return {"pool_dir": str(pool_dir), "meta": meta, "slots": slots}


def submit_and_wait(
    pool_dir: Path,
    slot: int,
    cmd: str,
    cwd: str | None = None,
    timeout_s: float = 1800.0,
    env: dict[str, str] | None = None,
    poll_interval: float = 0.25,
) -> dict:
    slot_dir = pool_dir / "slots" / f"gpu{slot}"
    inbox = slot_dir / "inbox"
    running = slot_dir / "running"
    done = slot_dir / "done"
    inbox.mkdir(parents=True, exist_ok=True)
    done.mkdir(parents=True, exist_ok=True)

    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "slot": slot,
        "cmd": cmd,
        "cwd": cwd or os.getcwd(),
        "timeout_s": timeout_s,
        "env": env or {},
        "submitted_unix": time.time(),
    }
    tmp = inbox / f"{job_id}.json.tmp"
    final = inbox / f"{job_id}.json"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, final)

    done_file = done / f"{job_id}.json"
    deadline = time.time() + timeout_s + 60.0
    while True:
        if done_file.is_file():
            return json.loads(done_file.read_text())
        if time.time() > deadline:
            (running / f"{job_id}.cancel").write_text("client timeout\n")
            raise TimeoutError(f"client timed out waiting for {job_id} on gpu{slot}")
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--status", action="store_true", help="Print pool and slot heartbeats")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--env", action="append", default=[], help="KEY=VAL")
    parser.add_argument("--cmd", type=str, default=None)
    args = parser.parse_args(argv)

    if args.status:
        print(json.dumps(pool_status(args.pool_dir), indent=2, ensure_ascii=False))
        return 0

    if not args.cmd:
        parser.error("--cmd is required unless --status is passed")

    extra_env = {}
    for item in args.env:
        if "=" not in item:
            parser.error(f"invalid --env {item!r}, expected KEY=VAL")
        k, v = item.split("=", 1)
        extra_env[k] = v

    result = submit_and_wait(
        pool_dir=args.pool_dir,
        slot=args.slot,
        cmd=args.cmd,
        cwd=args.cwd,
        timeout_s=args.timeout,
        env=extra_env,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
