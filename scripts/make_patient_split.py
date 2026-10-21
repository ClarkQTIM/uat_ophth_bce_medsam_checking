#!/usr/bin/env python
"""
Re-split an existing split CSV so that every patient is in exactly one split
(DECISIONS D20; ERRATA E11). Same images, same target split sizes as the input.

  python scripts/make_patient_split.py --dataset ga  --in data/ga_split.csv  --out data/ga_patient_split.csv  --seed 1020
  python scripts/make_patient_split.py --dataset cup --in data/cup_split.csv --out data/cup_patient_split.csv --seed 1020

Patient ID from the image file name:
  ga  : "_Final_<site>_<patient>_..."  -> "<site>_<patient>"   (73 patients expected)
  cup : "RET<id>OD" / "RET<id>OS"      -> "RET<id>"             (244 patients expected)

Assignment (deterministic for a given seed): patients are shuffled with the seed,
then sorted by image count (largest first, stable), and each patient goes to the
split with the largest relative shortfall against the input's split sizes.
The output keeps every input column, adds `patient` and `split_original`, and
replaces `split`. Printed: achieved vs target counts and a leakage check.
The output contains in-house identifiers (GA): it stays in data/ (gitignored).
"""

import argparse
import hashlib
import os
import re

import numpy as np
import pandas as pd

PATTERNS = {
    "ga": (re.compile(r"_Final_([^_/]+)_([^_/]+)_"), lambda m: f"{m.group(1)}_{m.group(2)}", 73),
    "cup": (re.compile(r"(RET\d+)O[DS]", re.IGNORECASE), lambda m: m.group(1).upper(), 244),
}
ORDER = ["train", "test", "val", "cal"]


def patient_ids(paths, dataset):
    rx, key, _ = PATTERNS[dataset]
    out = []
    for p in paths:
        m = rx.search(os.path.basename(str(p)))
        if m is None:
            raise SystemExit(f"[{dataset}] no patient ID in a file name (row {len(out)}); "
                             f"check the pattern {rx.pattern!r} against your file names")
        out.append(key(m))
    return out


def assign(patients_sizes, targets, seed):
    rng = np.random.default_rng(seed)
    pats = list(patients_sizes)
    rng.shuffle(pats)
    pats.sort(key=lambda p: -patients_sizes[p])          # stable: ties keep shuffled order
    count = {s: 0 for s in targets}
    where = {}
    for p in pats:
        best = max(targets, key=lambda s: ((targets[s] - count[s]) / targets[s], -ORDER.index(s)))
        where[p] = best
        count[best] += patients_sizes[p]
    return where, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(PATTERNS))
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "split" not in df or "image" not in df:
        raise SystemExit("input CSV needs 'image' and 'split' columns")
    df["patient"] = patient_ids(df["image"], args.dataset)
    n_pat = df["patient"].nunique()
    expected = PATTERNS[args.dataset][2]
    print(f"[{args.dataset}] {len(df)} images, {n_pat} patients (expected {expected})")
    if n_pat != expected:
        raise SystemExit(f"patient count {n_pat} != {expected}: stop and check the ID pattern")

    targets = df["split"].value_counts().to_dict()
    leak_before = int((df.groupby("patient")["split"].nunique() > 1).sum())
    sizes = df.groupby("patient").size().to_dict()
    where, count = assign(sizes, targets, args.seed)

    df["split_original"] = df["split"]
    df["split"] = df["patient"].map(where)
    leak_after = int((df.groupby("patient")["split"].nunique() > 1).sum())

    print(f"patients in more than one split: before {leak_before}, after {leak_after}")
    print("split   target  achieved  patients")
    for s in ORDER:
        if s in targets:
            print(f"{s:6s} {targets[s]:7d} {count[s]:9d} {sum(v == s for v in where.values()):9d}")
    moved = int((df["split"] != df["split_original"]).sum())
    print(f"images whose split changed: {moved} of {len(df)}")
    if leak_after:
        raise SystemExit("leakage remains: bug")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out}  sha256 {h}")


if __name__ == "__main__":
    main()
