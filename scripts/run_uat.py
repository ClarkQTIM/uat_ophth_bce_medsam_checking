#!/usr/bin/env python
"""
Run ONE UAT training run.

Three ways to specify it:
  python scripts/run_uat.py --jobs jobs/parity_sam.jsonl --index 17
      (line 17, 0-based, of a job file made by scripts/make_jobs.py; this is what
       the SLURM array uses)
  python scripts/run_uat.py --config my_run.yaml
  python scripts/run_uat.py --config my_run.yaml --set uq_type=entropy gamma=5 n_warmup=5

Every resolved config is written to the run folder's run_manifest.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from uatcheck.train_uat import train_one_run  # noqa: E402
from uatcheck.train_multiclass import train_one_run as train_one_run_mc  # noqa: E402


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", help="JSONL job file from make_jobs.py")
    ap.add_argument("--index", type=int, help="0-based line in --jobs "
                    "(defaults to $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--config", help="YAML file with a single run config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    cfg = {}
    if args.jobs:
        idx = args.index
        if idx is None:
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        with open(args.jobs) as f:
            lines = [l for l in f if l.strip()]
        cfg = json.loads(lines[idx])
        print(f"[job] {args.jobs} line {idx} of {len(lines)}")
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    if not cfg:
        ap.error("give --jobs or --config")
    print("[config]", json.dumps(cfg, indent=2))
    if cfg.get("task") == "multiclass":       # 3-class UAT (DECISIONS D21)
        train_one_run_mc(cfg)
    else:
        train_one_run(cfg)


if __name__ == "__main__":
    main()
