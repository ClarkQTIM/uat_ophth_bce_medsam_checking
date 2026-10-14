#!/usr/bin/env python
"""
Expand an experiment manifest (configs/errata/*.yaml) into a JSONL job file:
one fully specified run config per line.

  python scripts/make_jobs.py configs/errata/parity_sam.yaml \
      --datasets configs/datasets.yaml --out jobs/parity_sam.jsonl

Manifest format (see the files in configs/errata/ for real examples):

  name: parity_sam
  backbones: [sam-vit-huge]
  runs: [1, 10]                 # inclusive range
  output_root: results
  common: {weight_form: paper}  # merged into every job
  arms:
    cup:
      - {uq_type: none}
      - {uq_type: entropy, gamma: 5, n_warmup: 5}
      - {uq_type: conformal, alpha: 0.20, gamma: 2, n_warmup: 1}

The job order interleaves arms within each run index (run 1 of every arm, then
run 2 of every arm, ...), so a partially finished array still gives a balanced
comparison and baseline/UQ arms run side by side in time and environment
(docs/DECISIONS.md, D11).
"""

import argparse
import json
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-dataset", nargs="*", default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = yaml.safe_load(f)
    with open(args.datasets) as f:
        dsets = yaml.safe_load(f)

    if "arms_file" in man:   # arms kept in one shared file (configs/errata/paper_arms.yaml)
        path = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), man["arms_file"])
        with open(path) as f:
            man["arms"] = yaml.safe_load(f)
    if man.get("datasets"):   # optional subset of datasets
        man["arms"] = {k: v for k, v in man["arms"].items() if k in man["datasets"]}
    if man.get("arm_filter"):  # optional: keep only arms matching all given key/values
        flt = man["arm_filter"]
        man["arms"] = {k: [a for a in v if all(a.get(fk) == fv for fk, fv in flt.items())]
                       for k, v in man["arms"].items()}

    lo, hi = man.get("runs", [1, 10])
    common = man.get("common", {}) or {}
    jobs = []
    for run in range(int(lo), int(hi) + 1):
        for backbone in man["backbones"]:
            for dname, arms in man["arms"].items():
                if args.only_dataset and dname not in args.only_dataset:
                    continue
                if dname not in dsets:
                    raise SystemExit(f"dataset '{dname}' not in {args.datasets}")
                d = dsets[dname]
                for arm in arms:
                    job = {
                        "experiment": man["name"],
                        "dataset": dname,
                        "csv": d["csv"],
                        "mask_max_value": d.get("mask_max_value", 255),
                        "backbone": backbone,
                        "output_root": man.get("output_root", "results"),
                        "run": run,
                        "gamma": 0, "n_warmup": 0, "alpha": None,
                        **common, **arm,
                    }
                    jobs.append(job)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    print(f"SLURM: sbatch --array=0-{len(jobs) - 1}%<max_concurrent> "
          f"scripts/slurm_array.sbatch {args.out}")


if __name__ == "__main__":
    main()
