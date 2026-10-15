#!/usr/bin/env python
"""
Reproduce early stopping offline under each validation criterion (DECISIONS D16/D17).

  python scripts/select_offline.py --root results/errata_v2 --backbone sam_vit_huge
  python scripts/select_offline.py --root results/errata_v2 --backbone medsam

Needs runs made with track_all_criteria: true (epoch_log.csv has val_loss_unweighted,
val_loss_weighted, val_soft_dice and test_soft_dice for every epoch). Training does not
depend on when it stops, so for each criterion the result of "early stopping with
patience P on that criterion" is exactly: test Dice at the epoch with the best criterion
value before patience ran out.

Criteria
  unweighted  minimize validation BCE                (what the paper's Methods describe)
  weighted    minimize U-weighted validation BCE     (what the paper-era code did; ERRATA E14)
  dice        maximize validation soft Dice          (common to all arms)
  as_run      per cell, the criterion the paper-era runs of that cell actually used
              (weighted, except cells marked paper_val: unweighted in paper_arms.yaml)

Writes to <root>/<dataset>/:
  selections_<backbone>.csv     one row per run x criterion
  summary_<backbone>.csv        per criterion x arm: n, mean, sd, epochs, MW vs baseline
                                (same criterion), Holm, Cliff's delta, paper runs
  verdict_<backbone>.txt        D12 parity (as_run) and D17 questions, in words
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

CRITS = {"unweighted": ("val_loss_unweighted", 1), "weighted": ("val_loss_weighted", 1),
         "dice": ("val_soft_dice", -1)}


def simulate(ep, col, sign, patience):
    best, best_ep, bad, stop = np.inf, None, 0, None
    for _, r in ep.iterrows():
        v = sign * float(r[col])
        if v < best:
            best, best_ep, bad = v, int(r["epoch"]), 0
        else:
            bad += 1
        if bad >= patience:
            stop = int(r["epoch"])
            break
    return best_ep, stop


def norm(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/errata_v2")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--loss_dir", default="weighted_bce_v2")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--arms", default="configs/errata/paper_arms.yaml")
    ap.add_argument("--reference", default="configs/errata/paper_reference.yaml")
    ap.add_argument("--paper_runs", default="docs/records/paper_era_runs.csv")
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu

    arms_cfg = yaml.safe_load(open(args.arms))
    ref = yaml.safe_load(open(args.reference))
    paper = pd.read_csv(args.paper_runs, dtype={"gamma": str, "W": str}) \
        if os.path.exists(args.paper_runs) else None

    def as_run_crit(dataset, arm, g, w):
        for a in arms_cfg.get(dataset, []):
            name = a["uq_type"] if a["uq_type"] != "conformal" else \
                f"conformal_lac_{int(round(a['alpha'] * 100)):03d}"
            if name == arm and norm(a.get("gamma", 0)) == norm(g) and norm(a.get("n_warmup", 0)) == norm(w):
                return a.get("paper_val", "weighted")
        return "weighted"

    for dataset in sorted(os.listdir(args.root)):
        base = os.path.join(args.root, dataset, args.backbone, args.loss_dir)
        if not os.path.isdir(base):
            continue
        rows = []
        for run_dir in glob.glob(os.path.join(base, "gamma_*", "n_warmup_*", "*", "run_*")):
            f = os.path.join(run_dir, "epoch_log.csv")
            if not os.path.exists(os.path.join(run_dir, "test_metrics.txt")) or not os.path.exists(f):
                continue
            ep = pd.read_csv(f)
            if "test_soft_dice" not in ep or ep["test_soft_dice"].isna().all():
                continue      # run made without track_all_criteria
            g, w, arm, run = re.search(r"gamma_([^/]+)/n_warmup_([^/]+)/([^/]+)/run_(\d+)",
                                       run_dir.replace(os.sep, "/")).groups()
            sel = {}
            for c, (col, sign) in CRITS.items():
                be, stop = simulate(ep, col, sign, args.patience)
                sel[c] = (be, stop)
            sel["as_run"] = sel[as_run_crit(dataset, arm, g, w)]
            for c, (be, stop) in sel.items():
                r = ep[ep.epoch == be].iloc[0]
                rows.append({"dataset": dataset, "arm": arm, "gamma": norm(g), "n_warmup": norm(w),
                             "run": int(run), "criterion": c, "best_epoch": be, "stop_epoch": stop,
                             "epochs_paper_count": (stop - 1) if stop else len(ep),
                             "epochs_run": len(ep),
                             "test_soft_dice": float(r["test_soft_dice"]),
                             "test_hard_dice": float(r["test_hard_dice"]),
                             "val_soft_dice": float(r["val_soft_dice"])})
        if not rows:
            print(f"[{dataset}] no runs with per-epoch tracking")
            continue
        S = pd.DataFrame(rows)
        out_dir = os.path.join(args.root, dataset)
        S.to_csv(os.path.join(out_dir, f"selections_{args.backbone}.csv"), index=False)

        summ, lines = [], [f"dataset={dataset} backbone={args.backbone} patience={args.patience}"]
        for c in ["as_run", "unweighted", "weighted", "dice"]:
            x = S[S.criterion == c]
            b = x[x.arm == "none"]["test_soft_dice"].values
            crow = []
            for (arm, g, w), v in x.groupby(["arm", "gamma", "n_warmup"]):
                d = v["test_soft_dice"].values
                r = {"criterion": c, "arm": arm, "gamma": g, "n_warmup": w, "n": len(d),
                     "mean": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                     "mean_epochs_paper_count": v["epochs_paper_count"].mean(),
                     "mean_best_epoch": v["best_epoch"].mean()}
                if arm != "none" and len(d) > 1 and len(b) > 1:
                    r["mw_p"] = mannwhitneyu(d, b, alternative="two-sided").pvalue
                    r["cliffs_delta"] = cliffs_delta(d, b)
                    r["diff_vs_baseline"] = d.mean() - b.mean()
                if paper is not None and c == "as_run":
                    pr = paper[(paper.dataset == dataset) & (paper.arm == arm)] if arm == "none" else \
                        paper[(paper.dataset == dataset) & (paper.arm == arm) &
                              (paper.gamma.map(norm) == g) & (paper.W.map(norm) == w)]
                    r["paper_runs_n"] = len(pr)
                    if len(pr) > 1 and len(d) > 1:
                        r["paper_runs_mean"] = pr.dice.mean()
                        r["mw_p_vs_paper_runs"] = mannwhitneyu(d, pr.dice.values).pvalue
                crow.append(r)
            cdf = pd.DataFrame(crow)
            uq = (cdf.arm != "none") & cdf.get("mw_p", pd.Series(np.nan, index=cdf.index)).notna()
            if uq.any():
                cdf.loc[uq, "mw_p_holm"] = holm(cdf.loc[uq, "mw_p"].values)
            summ.append(cdf)

            if len(b) == 0:
                lines.append(f"[{c}] no baseline runs yet")
                continue
            lines.append(f"\n[{c}] baseline n={len(b)} mean {b.mean():.4f}")
            for _, r in cdf[cdf.arm != "none"].iterrows():
                lines.append(f"  {r.arm:18s} g={r.gamma} W={r.n_warmup}: n={r.n:2d} mean {r['mean']:.4f} "
                             f"({r.get('diff_vs_baseline', np.nan):+.4f}) P={r.get('mw_p', np.nan):.3g} "
                             f"Holm={r.get('mw_p_holm', np.nan):.3g} epochs={r.mean_epochs_paper_count:.1f}"
                             + (f" | paper runs {r.paper_runs_mean:.4f} P={r.mw_p_vs_paper_runs:.3g}"
                                if pd.notna(r.get("paper_runs_mean", np.nan)) else ""))
            if c == "as_run" and dataset in ref:
                tol, pm = ref[dataset]["parity_tolerance"], ref[dataset]["mean_soft_dice"]
                ent = cdf[cdf.arm == "entropy"]
                if len(ent) and pd.notna(ent["mw_p"].iloc[0] if "mw_p" in ent else np.nan):
                    ok_b = abs(b.mean() - pm["none"]) <= tol
                    ok_e = abs(ent["mean"].iloc[0] - pm["entropy"]) <= tol
                    ok_s = ent["mean"].iloc[0] > b.mean() and ent["mw_p"].iloc[0] < 0.05
                    lines.append(f"  D12 parity: baseline {'ok' if ok_b else 'FAIL'}, entropy "
                                 f"{'ok' if ok_e else 'FAIL'}, entropy>baseline {'ok' if ok_s else 'FAIL'}"
                                 f" -> {'PASS' if ok_b and ok_e and ok_s else 'FAIL'}")
        pd.concat(summ).to_csv(os.path.join(out_dir, f"summary_{args.backbone}.csv"), index=False)
        txt = "\n".join(lines)
        print(txt + "\n")
        with open(os.path.join(out_dir, f"verdict_{args.backbone}.txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
