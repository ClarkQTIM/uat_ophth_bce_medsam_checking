"""
manifest.py

Writes run_manifest.json into every run folder: exactly what ran, on what,
with which code. This is the record the paper-era runs did not keep.
"""

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime


def _git(*args):
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", *args], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_state():
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "hostname": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    for mod in ("torch", "transformers", "monai", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    except Exception:
        pass
    return env


def write_manifest(run_dir, config, backbone_info=None, extra=None):
    m = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": environment(),
        "config": config,
        "backbone": backbone_info.as_dict() if backbone_info else None,
    }
    if extra:
        m.update(extra)
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=2, default=str)
    if m["git"]["dirty"]:
        print("[manifest] WARNING: git working tree is dirty; commit before production runs.")
    return path


def update_manifest(run_dir, **fields):
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path) as f:
        m = json.load(f)
    m.update(fields)
    with open(path, "w") as f:
        json.dump(m, f, indent=2, default=str)
