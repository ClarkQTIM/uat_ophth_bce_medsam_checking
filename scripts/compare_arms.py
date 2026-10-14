#!/usr/bin/env python
"""
Summarize re-run results and apply the pre-registered criteria (DECISIONS D12).

  python scripts/compare_arms.py --root results/errata --backbone sam_vit_huge --mode parity
  python scripts/compare_arms.py --root results/errata --backbone medsam --mode primary

Reads every run folder under <root>/<dataset>/<backbone>/<loss_dir>/gamma_*/n_warmup_*/<arm>/run_*/
and writes, per dataset:
  summary_<backbone>_<loss_dir>.csv   one row per arm (n, mean, sd, median, epochs,
                                      Mann-Whitney P vs baseline, Holm-adjusted P,
                                      Cliff's delta, bootstrap 95% CI of the mean
                                      difference, Levene P, paper reference value)
  verdict_<backbone>_<loss_dir>.txt   PASS/FAIL per D12, in words
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def read_run(run_dir):
    lp = pd.read_csv(os.path.join(run_dir, "loss_parameters.csv"))
    test = lp[lp["epoch"].astype(str) == "test"].iloc[0]
    numeric = lp[pd.to_numeric(lp["epoch"], errors="coerce").notna()]
    return {"soft_dice": float(test["test_dice_foreground"]),
            "hard_dice": float(test.get("test_hard_dice", np.nan)),
            "val_dice_at_best": float(test.get("val_dice_at_best", np.nan)),
            "epochs_paper_count": len(numeric),
            "best_epoch": test.get("trained_epochs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/errata")
    ap.add_argument("--backbone", required=True, help="folder tag: sam_vit_huge | medsam")
    ap.add_argument("--loss_dir", default="weighted_bce_v2")
    ap.add_argument("--mode", choices=["parity", "primary", "plain"], default="plain")
    ap.add_argument("--reference", default="configs/errata/paper_reference.yaml")
    args = ap.parse_args()
    from scipy.stats import levene, mannwhitneyu

    ref = yaml.safe_load(open(args.reference))
    for dataset in sorted(os.listdir(args.root)):
        base = os.path.join(args.root, dataset, args.backbone, args.loss_dir)
        if not os.path.isdir(base):
            continue
        arms = {}
        for run_dir in glob.glob(os.path.join(base, "gamma_*", "n_warmup_*", "*", "run_*")):
            if not os.path.exists(os.path.join(run_dir, "test_metrics.txt")):
                continue
            g, w, arm = re.search(r"gamma_([^/]+)/n_warmup_([^/]+)/([^/]+)/run_", run_dir).groups()
            key = (arm, g, w)
            arms.setdefault(key, []).append(read_run(run_dir))
        base_key = next((k for k in arms if k[0] == "none"), None)
        if base_key is None:
            print(f"[{dataset}] no baseline runs yet")
            continue
        b = [r["soft_dice"] for r in arms[base_key]]
        rows = []
        for (arm, g, w), runs in sorted(arms.items()):
            x = [r["soft_dice"] for r in runs]
            row = {"dataset": dataset, "arm": arm, "gamma": g, "n_warmup": w, "n": len(x),
                   "mean": np.mean(x), "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
                   "median": np.median(x),
                   "mean_hard_dice": np.nanmean([r["hard_dice"] for r in runs]),
                   "mean_val_dice": np.nanmean([r["val_dice_at_best"] for r in runs]),
                   "avg_epochs_paper_count": np.mean([r["epochs_paper_count"] for r in runs]),
                   "paper_mean": ref.get(dataset, {}).get("mean_soft_dice", {}).get(arm),
                   "paper_avg_epochs": ref.get(dataset, {}).get("avg_epochs", {}).get(arm)}
            if arm != "none" and len(x) > 1 and len(b) > 1:
                row["mw_p"] = mannwhitneyu(x, b, alternative="two-sided").pvalue
                row["cliffs_delta"] = cliffs_delta(x, b)
                lo, hi = boot_ci(x, b)
                row["diff_ci_lo"], row["diff_ci_hi"] = lo, hi
                row["levene_p"] = levene(x, b).pvalue
            rows.append(row)
        df = pd.DataFrame(rows)
        uq = df["arm"] != "none"
        if "mw_p" in df and uq.any():
            df.loc[uq, "mw_p_holm"] = holm(df.loc[uq, "mw_p"].fillna(1.0).values)
        out_dir = os.path.join(args.root, dataset)
        df.to_csv(os.path.join(out_dir, f"summary_{args.backbone}_{args.loss_dir}.csv"), index=False)

        lines = [f"dataset={dataset} backbone={args.backbone} loss={args.loss_dir} mode={args.mode}"]
        bm = df[df.arm == "none"]["mean"].item()
        ent = df[df.arm == "entropy"]
        if args.mode == "parity" and len(ent):
            tol = ref[dataset]["parity_tolerance"]
            pm = ref[dataset]["mean_soft_dice"]
            ok_b = abs(bm - pm["none"]) <= tol
            ok_e = abs(ent["mean"].item() - pm["entropy"]) <= tol
            ok_s = ent["mean"].item() > bm and ent["mw_p"].item() < 0.05
            lines += [f"baseline mean {bm:.4f} vs paper {pm['none']:.3f} (tol {tol}): {'ok' if ok_b else 'FAIL'}",
                      f"entropy mean {ent['mean'].item():.4f} vs paper {pm['entropy']:.3f}: {'ok' if ok_e else 'FAIL'}",
                      f"entropy > baseline, P={ent['mw_p'].item():.4g}: {'ok' if ok_s else 'FAIL'}",
                      f"PARITY {'PASS' if ok_b and ok_e and ok_s else 'FAIL'}"]
        elif args.mode == "primary" and len(ent):
            up = ent["mean"].item() > bm and ent["mw_p"].item() < 0.05
            lines += [f"entropy {ent['mean'].item():.4f} vs baseline {bm:.4f}, P={ent['mw_p'].item():.4g}, "
                      f"Cliff's delta {ent['cliffs_delta'].item():.2f}",
                      f"MAIN CLAIM ON THIS BACKBONE: {'UPHELD' if up else 'NOT UPHELD'}"]
            for _, r in df[uq].iterrows():
                lines.append(f"  {r.arm:20s} g={r.gamma} W={r.n_warmup}: mean {r['mean']:.4f} "
                             f"(diff {r['mean'] - bm:+.4f}), P={r.get('mw_p', np.nan):.4g}, "
                             f"Holm P={r.get('mw_p_holm', np.nan):.4g}")
        txt = "\n".join(lines)
        print(txt + "\n")
        with open(os.path.join(out_dir, f"verdict_{args.backbone}_{args.loss_dir}.txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
