#!/usr/bin/env python3
"""NorthJob launcher adapter for SpecPilot v2 persistent GPU pool and controller jobs.

Follows northjob-submit SKILL.md:
- Explicit absolute --workspace and confirmed --k8s-images
- Attaches read-only /gpfs/models PVC (share-readonly, subPath=models)
- Pre-submit deduplication via `northjob get --keyword <job-name> --page-size 100`
- Dry-run by default; requires --execute to actually call `northjob create`
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORTHJOB = os.environ.get("NORTHJOB_BIN", "/gpfs/users/jixinpeng/miniconda3/envs/dev/bin/northjob")
DEFAULT_IMAGE = "harbor.local.clusters/bp/oss-sglang@sha256:6e9415fcf90dbe66f14e994a7f7c51cb242db3805b1910ce7bd6d0b8e8db0798"
TOLERATION = "key=task-type,operator=Equal,value=infra-dev-weaver,effect=NoSchedule"


def attach_model_pvc(template: dict, gpus: int, deadline: int) -> dict:
    if not (1 <= gpus <= 8):
        raise ValueError(f"gpus must be 1..8, got {gpus}")
    if not (600 <= deadline <= 86400):
        raise ValueError(f"deadline must be 600..86400s, got {deadline}")
    result = copy.deepcopy(template)
    tasks = result["spec"]["tasks"]
    if len(tasks) != 1 or tasks[0]["replicas"] != 1:
        raise ValueError("single-node pod required")
    pod = tasks[0]["template"]["spec"]
    containers = pod["containers"]
    if len(containers) != 1:
        raise ValueError("single container required")
    actual_gpu = str(containers[0]["resources"]["limits"].get("nvidia.com/gpu"))
    if actual_gpu != str(gpus):
        raise ValueError(f"expected {gpus} GPUs in template, got {actual_gpu}")
    mounts = containers[0].setdefault("volumeMounts", [])
    volumes = pod.setdefault("volumes", [])
    if not any(m.get("mountPath") == "/gpfs/models" for m in mounts):
        volumes.append(
            {
                "name": "specpilot-models",
                "persistentVolumeClaim": {"claimName": "share-readonly", "readOnly": True},
            }
        )
        mounts.append(
            {
                "name": "specpilot-models",
                "mountPath": "/gpfs/models",
                "subPath": "models",
                "readOnly": True,
            }
        )
    pod["activeDeadlineSeconds"] = deadline
    result["spec"]["maxRetry"] = 0
    result["spec"]["policies"] = [{"events": ["PodEvicted", "PodFailed"], "action": "AbortJob"}]
    return result


def build_create_args(
    *,
    job_name: str,
    gpus: int,
    mem_gib: int,
    user_cmd: str,
    image: str = DEFAULT_IMAGE,
    priority: int = 3,
    workspace: Path = ROOT,
) -> list[str]:
    return [
        "create",
        "--job-name", job_name,
        "--project", "WeaverDev",
        "--workspace", str(workspace.resolve()),
        "--nnodes", "1",
        "--nproc-per-node", "1",
        "--gpu-per-node", str(gpus),
        "--gpu-resource", "h100",
        "--mem-per-pod", str(mem_gib),
        "--no-rdma",
        "--k8s-no-hostnetwork",
        "--k8s-namespace", "qiji",
        "--k8s-queue", "qiji",
        "--k8s-toleration", TOLERATION,
        "--k8s-pvc-name", "jixinpeng-pvc",
        "--k8s-pvc-mount-path", "/gpfs/users/jixinpeng",
        "--k8s-ttl", "86400",
        "--k8s-priority", str(priority),
        "--k8s-images", image,
        user_cmd,
    ]


def check_duplicate(job_name: str) -> list[dict]:
    proc = subprocess.run(
        [NORTHJOB, "get", "--keyword", job_name, "--page-size", "100"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"northjob get failed: {proc.stderr or proc.stdout}")
    text = proc.stdout
    start = text.find("{")
    if start < 0:
        return []
    end = text.rfind("}")
    data = json.loads(text[start : end + 1])
    rows = data.get("data") or []
    return [r for r in rows if r.get("user_def_name") == job_name]


def _run_patched_create(create_args: list[str], gpus: int, deadline: int) -> int:
    launcher = importlib.import_module("northmirror.launcher.create_job")
    cli = importlib.import_module("northmirror.launcher.northjob")
    original = launcher.build_job_yaml

    def patched(*args, **kwargs):
        return attach_model_pvc(original(*args, **kwargs), gpus=gpus, deadline=deadline)

    launcher.build_job_yaml = patched
    old_argv = sys.argv
    sys.argv = ["northjob"] + create_args
    try:
        return cli.main() or 0
    finally:
        launcher.build_job_yaml = original
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--mem-gib", type=int, default=512)
    parser.add_argument("--deadline", type=int, default=43200, help="activeDeadlineSeconds (default 12h)")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--priority", type=int, default=3)
    parser.add_argument("--user-cmd", required=True)
    parser.add_argument("--execute", action="store_true", help="Actually submit after duplicate check")
    args = parser.parse_args(argv)

    create_args = build_create_args(
        job_name=args.job_name,
        gpus=args.gpus,
        mem_gib=args.mem_gib,
        user_cmd=args.user_cmd,
        image=args.image,
        priority=args.priority,
    )
    summary = {
        "job_name": args.job_name,
        "mode": "bash (with /gpfs/models read-only PVC)",
        "workspace": str(ROOT),
        "nodes": 1,
        "gpu_per_node": args.gpus,
        "total_gpus": args.gpus,
        "gpu_type": "h100",
        "mem_gib": args.mem_gib,
        "deadline_s": args.deadline,
        "project": "WeaverDev",
        "namespace_queue": "qiji",
        "priority": args.priority,
        "image": args.image,
        "command": shlex.join(["northjob"] + create_args),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.execute:
        print("\n[dry-run] Pass --execute to submit after reviewing the summary above.")
        return 0

    dups = check_duplicate(args.job_name)
    active = [r for r in dups if r.get("status") in ("Running", "Pending", "Creating", "Starting")]
    if active:
        raise SystemExit(f"ABORT: active job with exact name {args.job_name!r} already exists: {active[0].get('name')}")
    if dups:
        raise SystemExit(f"ABORT: job name {args.job_name!r} already used previously ({len(dups)} matches); pick a fresh name.")

    return _run_patched_create(create_args, gpus=args.gpus, deadline=args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
