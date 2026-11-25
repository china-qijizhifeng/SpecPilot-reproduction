#!/usr/bin/env python3
"""Persistent multi-GPU exploration pool worker (Scheme A).

Runs inside a single NorthJob pod (e.g. 8xH100). Spawns one dedicated worker
thread per GPU slot (0..num_gpus-1), each polling its GPFS inbox directory
every 0.25s. Zero cold-start between Direction microbenchmarks, unit tests,
and 1-GPU timing runs.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


STOP_EVENT = threading.Event()


def _chown_to_pool(path: Path, uid: int, gid: int) -> None:
    try:
        os.chown(path, uid, gid)
    except PermissionError:
        pass


def _atomic_write_json(path: Path, payload: dict, owner: tuple[int, int] | None = None) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    try:
        os.chmod(tmp, 0o664)
    except OSError:
        pass
    if owner is not None:
        _chown_to_pool(tmp, owner[0], owner[1])
    os.replace(tmp, path)


def _query_gpu_info(slot: int) -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={slot}",
                "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) == 5:
            return {
                "uuid": parts[0],
                "name": parts[1],
                "mem_used_mib": int(parts[2]),
                "mem_total_mib": int(parts[3]),
                "util_pct": int(parts[4]),
            }
    except Exception:
        pass
    return {"uuid": f"mock-gpu-{slot}", "name": "unknown", "mem_used_mib": 0, "mem_total_mib": 0, "util_pct": 0}


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _tail_text(path: Path, max_bytes: int = 8192) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def run_slot_loop(pool_dir: Path, slot: int, poll_interval: float = 0.25, mock_gpu: bool = False) -> None:
    st = pool_dir.stat()
    owner = (st.st_uid, st.st_gid)
    slot_dir = pool_dir / "slots" / f"gpu{slot}"
    inbox = slot_dir / "inbox"
    running = slot_dir / "running"
    done = slot_dir / "done"
    logs = slot_dir / "logs"
    for d in (pool_dir / "slots", slot_dir, inbox, running, done, logs):
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o775)
        except OSError:
            pass
        _chown_to_pool(d, owner[0], owner[1])

    hb_path = slot_dir / "heartbeat.json"
    gpu_info = {"uuid": f"mock-gpu-{slot}", "name": "mock", "mem_used_mib": 0, "mem_total_mib": 0, "util_pct": 0} if mock_gpu else _query_gpu_info(slot)

    def write_hb(state: str, job_id: str | None = None) -> None:
        info = gpu_info if mock_gpu else _query_gpu_info(slot)
        _atomic_write_json(
            hb_path,
            {
                "slot": slot,
                "pid": os.getpid(),
                "hostname": os.uname().nodename,
                "state": state,
                "active_job": job_id,
                "gpu": info,
                "updated_unix": time.time(),
            },
            owner=owner,
        )

    last_hb = 0.0
    write_hb("idle")

    while not STOP_EVENT.is_set():
        now = time.time()
        if now - last_hb >= 5.0:
            write_hb("idle")
            last_hb = now

        candidates = sorted(p for p in inbox.glob("*.json") if not p.name.endswith(".tmp"))
        if not candidates:
            time.sleep(poll_interval)
            continue

        job_file = candidates[0]
        running_file = running / job_file.name
        try:
            os.replace(job_file, running_file)
        except FileNotFoundError:
            continue

        job = json.loads(running_file.read_text())
        job_id = job["job_id"]
        cmd = job["cmd"]
        cwd = job.get("cwd") or str(pool_dir)
        timeout_s = float(job.get("timeout_s", 1800))
        extra_env = job.get("env") or {}
        log_path = logs / f"{job_id}.log"

        write_hb("busy", job_id)
        started = time.time()
        timed_out = False
        cancelled = False
        exit_code = -1

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in extra_env.items()})
        if not mock_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        env["SP_POOL_SLOT"] = str(slot)
        env["SP_JOB_ID"] = str(job_id)

        cancel_file = running / f"{job_id}.cancel"
        try:
            with log_path.open("wb") as log_fh:
                proc = subprocess.Popen(
                    ["bash", "-lc", cmd],
                    cwd=cwd,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while True:
                    ret = proc.poll()
                    if ret is not None:
                        exit_code = ret
                        break
                    if STOP_EVENT.is_set() or cancel_file.exists():
                        cancelled = True
                        _kill_process_group(proc)
                        exit_code = -15
                        break
                    if time.time() - started > timeout_s:
                        timed_out = True
                        _kill_process_group(proc)
                        exit_code = 124
                        break
                    if time.time() - last_hb >= 5.0:
                        write_hb("busy", job_id)
                        last_hb = time.time()
                    time.sleep(0.2)
        except Exception as exc:
            log_path.write_text(f"[pool_worker error] {exc!r}\n")
            exit_code = 127

        finished = time.time()
        result = {
            "job_id": job_id,
            "slot": slot,
            "cmd": cmd,
            "cwd": cwd,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_s": round(finished - started, 3),
            "log_path": str(log_path),
            "stdout_tail": _tail_text(log_path),
            "gpu_uuid": gpu_info["uuid"],
            "hostname": os.uname().nodename,
        }
        _atomic_write_json(done / f"{job_id}.json", result, owner=owner)
        _chown_to_pool(log_path, owner[0], owner[1])
        try:
            running_file.unlink(missing_ok=True)
            cancel_file.unlink(missing_ok=True)
        except OSError:
            pass
        write_hb("idle")
        last_hb = time.time()

    write_hb("stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--mock-gpu", action="store_true")
    args = parser.parse_args(argv)

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        args.pool_dir / "pool_meta.json",
        {
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "num_gpus": args.num_gpus,
            "started_unix": time.time(),
            "mock_gpu": args.mock_gpu,
        },
    )

    def _handle_sig(signum, _frame):
        STOP_EVENT.set()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    threads = [
        threading.Thread(
            target=run_slot_loop,
            args=(args.pool_dir, slot, args.poll_interval, args.mock_gpu),
            name=f"gpu-slot-{slot}",
            daemon=True,
        )
        for slot in range(args.num_gpus)
    ]
    for t in threads:
        t.start()
    print(f"[pool_worker] ready on {os.uname().nodename} with {args.num_gpus} slots at {args.pool_dir}", flush=True)
    while not STOP_EVENT.is_set():
        time.sleep(0.5)
    for t in threads:
        t.join(timeout=10)
    print("[pool_worker] clean shutdown", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
