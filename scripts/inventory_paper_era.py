#!/usr/bin/env python
"""
Inventory of the surviving paper-era result tree (KNOWN_ISSUES K7).

  python scripts/inventory_paper_era.py <path>/training_essentials_paper --out docs/records/paper_era_runs.csv

One row per run folder with test_metrics.txt: dataset, gamma, W (recalc), arm, run,
test soft Dice ("Foreground Dice Score"), number of test images, "Final Gamma"
line, epoch rows in loss_parameters.csv (paper-era counting, ERRATA E8),
trained/stopped epochs, max validation uncertainty.
Baselines live at <dataset>/medsam/weighted_bce_v2/none/run_N (gamma/W = "-").
"""
import argparse, os, re, glob, sys
import pandas as pd, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--out", default="docs/records/paper_era_runs.csv")
args = ap.parse_args()
root = args.root
rows = []
for tm in glob.glob(os.path.join(root, "**", "test_metrics.txt"), recursive=True):
    rel = os.path.relpath(tm, root)
    parts = rel.split(os.sep)
    ds = parts[0]
    m = re.search(r"gamma_([^/]+)/recalc_([^/]+)/([^/]+)/run_(\d+)", rel)
    if m:
        g, w, arm, run = m.groups()
    else:
        m2 = re.search(r"/([^/]+)/run_(\d+)/test_metrics.txt", "/" + rel)
        g, w, arm, run = "-", "-", m2.group(1) if m2 else "?", (m2.group(2) if m2 else "?")
    txt = open(tm).read()
    def grab(key):
        r = re.search(key + r":\s*([-\d.eE]+)", txt)
        return float(r.group(1)) if r else np.nan
    rec = {"dataset": ds, "gamma": g, "W": w, "arm": arm, "run": int(run) if str(run).isdigit() else run,
           "dice": grab("Foreground Dice Score"), "n_test": grab("Number of test samples"),
           "final_gamma": grab("Final Gamma"), "path": os.path.dirname(rel)}
    lp = os.path.join(os.path.dirname(tm), "loss_parameters.csv")
    if os.path.exists(lp):
        d = pd.read_csv(lp)
        num = d[pd.to_numeric(d["epoch"], errors="coerce").notna()]
        rec["epochs_rows"] = len(num)
        t = d[d["epoch"].astype(str) == "test"]
        rec["trained_epochs"] = float(t["trained_epochs"].iloc[0]) if len(t) and "trained_epochs" in t else np.nan
        es = d[d["epoch"].astype(str) == "early_stop"]
        rec["stopped_at"] = float(es["stopped_at_epoch"].iloc[0]) if len(es) else np.nan
        rec["max_val_unc"] = num["val_uncertainty"].max() if "val_uncertainty" in num else np.nan
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"wrote {len(df)} runs -> {args.out}")
print(df.groupby("dataset").size())
print(df[df.gamma == "-"][["dataset", "arm", "path"]].drop_duplicates("arm").to_string())
print(df[df.gamma == "-"].groupby(["dataset", "arm"]).agg(n=("dice", "size"), mean=("dice", "mean"), sd=("dice", "std"),
      ep=("epochs_rows", "mean"), n_test=("n_test", "first")).to_string())
cells = df[df.gamma != "-"].groupby(["dataset", "arm", "gamma", "W"]).size()
print("\ncells per dataset/arm (count of (gamma,W) cells; runs per cell min/max):")
print(df[df.gamma != "-"].groupby(["dataset", "arm"]).apply(
    lambda x: pd.Series({"cells": x.groupby(["gamma", "W"]).ngroups,
                         "runs_min": x.groupby(["gamma", "W"]).size().min(),
                         "runs_max": x.groupby(["gamma", "W"]).size().max()})).to_string())
