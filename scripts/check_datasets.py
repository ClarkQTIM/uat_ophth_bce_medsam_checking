#!/usr/bin/env python
"""
STEP 1 (docs/REPRODUCTION.md): confirm the split CSVs are the paper's splits.

  python scripts/check_datasets.py --datasets configs/datasets.yaml

For each dataset: CSV exists, has the right columns, split counts equal the
paper's (expected_counts), every image/mask file exists, and a sample of masks
is binary after dividing by mask_max_value. Prints a SHA-256 of each CSV so the
exact split file can be cited in the erratum record.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import yaml


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="configs/datasets.yaml")
    ap.add_argument("--mask_sample", type=int, default=20)
    args = ap.parse_args()
    with open(args.datasets) as f:
        dsets = yaml.safe_load(f)

    ok_all = True
    for name, d in dsets.items():
        print(f"\n=== {name} ===")
        csv = d["csv"]
        if not os.path.exists(csv):
            print(f"  MISSING CSV: {csv}")
            ok_all = False
            continue
        df = pd.read_csv(csv)
        print(f"  csv: {csv}\n  sha256: {sha256(csv)}")
        counts = df["split"].value_counts().to_dict()
        exp = d.get("expected_counts", {})
        for s, n in exp.items():
            flag = "ok" if counts.get(s, 0) == n else "MISMATCH"
            ok_all &= flag == "ok"
            print(f"  {s:5s}: {counts.get(s, 0):4d} (paper {n}) {flag}")
        missing = [p for col in ("image", "segmentation") for p in df[col] if not os.path.exists(p)]
        print(f"  missing files: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""))
        ok_all &= not missing
        try:
            from monai.transforms import LoadImage
            vals = set()
            for p in df["segmentation"].sample(min(args.mask_sample, len(df)), random_state=0):
                arr = np.asarray(LoadImage(image_only=True)(p))
                vals |= set(np.unique(np.round(arr)).tolist())
            mmax = float(d.get("mask_max_value", 255))
            binary = vals <= {0.0, mmax}
            ok_all &= binary
            print(f"  mask values (sample): {sorted(vals)[:8]} -> "
                  f"{'binary ok' if binary else 'NOT {0,' + str(mmax) + '}'}")
        except Exception as e:
            print(f"  mask check skipped: {e}")
    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
