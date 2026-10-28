"""
train_multiclass.py

One run of 3-class UAT on MedSAM (DECISIONS D21). Same structure as train_uat.py:
warmup (U = None while epoch_index < n_warmup), conformal recalibration when
(epoch_index + 1) % n_warmup == 0, U not detached, and every epoch logs all three
validation criteria plus test Dice so that scripts/select_offline.py can reproduce
early stopping under any of them.

Differences from the binary pipeline (all by design, D21):
  * softmax over the three MedSAM mask tokens, weighted cross-entropy;
  * PRIMARY early stopping = validation mean soft Dice (cup, disc), the common rule
    from the errata analysis (D17 Q3); `primary_criterion` can switch it;
  * logged Dice: cup, rim, disc and their mean; `val_soft_dice` / `test_soft_dice`
    hold the mean of cup and disc so select_offline.py works unchanged, and
    `test_cup_soft_dice` etc. hold the per-class values (select_offline --test_col).

Run folder: <output_root>/<dataset>/<backbone>/weighted_ce/gamma_X/n_warmup_W/<arm>/run_N
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .backbones import backbone_dir_tag, load_backbone
from .forward import prepare_inputs
from .manifest import update_manifest, write_manifest
from .metrics import AUROCAccumulator
from .multiclass import (WeightedCELoss, class_dice, forward_logits, make_mc_loaders,
                         recalibrate_mc, uncertainty_mc)
from .train_uat import _num, arm_dir_name, set_seed

LOSS_DIR_MC = {"paper": "weighted_ce", "gm1": "weighted_ce_gm1", "exp": "weighted_ce_exp"}

DEFAULTS = {
    "task": "multiclass",
    "dataset": None, "csv": None,
    "label_column": "combined_segmentation", "label_values": [0, 127, 255],
    "backbone": "medsam", "backbone_repo_id": None, "processor_id": None,
    "uq_type": "none", "alpha": None, "gamma": 0.0, "n_warmup": 0, "run": 1,
    "weight_form": "paper", "entropy_normalize": False, "detach_uncertainty": False,
    "clip_conformal_level": False,
    "primary_criterion": "dice",          # dice | unweighted | weighted
    "num_epochs": 40, "patience": 5, "batch_size": 1,
    "learning_rate": 1e-5, "weight_decay": 1e-5, "mask_size": [256, 256],
    "output_root": "results/multiclass_v1", "base_seed": 1000, "deterministic": False,
    "num_workers": 4, "limit_samples": None, "keep_checkpoint": False, "device": None,
}


def run_dir_for(cfg):
    return os.path.join(
        cfg["output_root"], cfg["dataset"], backbone_dir_tag(cfg["backbone"]),
        LOSS_DIR_MC[cfg["weight_form"]], f"gamma_{_num(cfg['gamma'])}",
        f"n_warmup_{int(cfg['n_warmup'])}", arm_dir_name(cfg["uq_type"], cfg["alpha"]),
        f"run_{int(cfg['run'])}")


def validate(cfg):
    for k in ("dataset", "csv"):
        if not cfg.get(k):
            raise ValueError(f"config key '{k}' is required")
    if cfg["uq_type"] not in ("none", "entropy", "conformal"):
        raise ValueError("multi-class arms: none | entropy | conformal (no variance arm, D21)")
    if cfg["uq_type"] == "conformal" and cfg["alpha"] is None:
        raise ValueError("conformal arm needs alpha")
    if cfg["uq_type"] != "none" and int(cfg["n_warmup"]) < 1:
        raise ValueError("UQ arms need n_warmup >= 1")
    if cfg["uq_type"] == "none" and (float(cfg["gamma"]) != 0 or int(cfg["n_warmup"]) != 0):
        raise ValueError("baseline must use gamma=0, n_warmup=0 (folder convention)")
    if cfg["primary_criterion"] not in ("dice", "unweighted", "weighted"):
        raise ValueError("primary_criterion: dice | unweighted | weighted")


def train_one_run(cfg_in):
    cfg = {**DEFAULTS, **cfg_in}
    validate(cfg)
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["base_seed"]) + int(cfg["run"])
    set_seed(seed, cfg["deterministic"])

    run_dir = run_dir_for(cfg)
    if os.path.exists(os.path.join(run_dir, "test_metrics.txt")):
        print(f"[skip] already complete: {run_dir}")
        return run_dir
    os.makedirs(run_dir, exist_ok=True)

    model, processor, binfo = load_backbone(cfg["backbone"], cfg["backbone_repo_id"],
                                            cfg["processor_id"], device)
    write_manifest(run_dir, {**cfg, "seed": seed, "device": device}, binfo)
    train_loader, val_loader, cal_loader, test_loader = make_mc_loaders(
        cfg["csv"], processor, cfg["batch_size"], tuple(cfg["mask_size"]), cfg["num_workers"],
        cfg["limit_samples"], cfg["label_column"], cfg["label_values"], seed)

    criterion = WeightedCELoss(cfg["gamma"], cfg["weight_form"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    uq, W, pat = cfg["uq_type"], int(cfg["n_warmup"]), int(cfg["patience"])
    prep = lambda inp: prepare_inputs(inp, device)

    def probs_of(inputs):
        logits = forward_logits(model, prep(inputs))
        return logits, logits.softmax(1)

    qhat, conformal_log, rows = None, [], []
    crit = {k: [math.inf, 0, 0] for k in ("unweighted", "weighted", "dice")}
    done = {k: False for k in crit}
    primary = cfg["primary_criterion"]
    qhat_primary = None
    ckpt = os.path.join(run_dir, "best_model.pth")

    for epoch in range(int(cfg["num_epochs"])):
        t0 = time.time()
        active = uq != "none" and epoch >= W
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.train()
        tr_loss, tr_u, n = 0.0, [], 0
        for inputs, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            logits, probs = probs_of(inputs)
            U = uncertainty_mc(uq, probs, qhat, cfg["entropy_normalize"]) if active else None
            if U is not None and cfg["detach_uncertainty"]:
                U = U.detach()
            loss = criterion(logits, y, U)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            n += 1
            if U is not None:
                tr_u.append(float(U.detach().mean()))
        tr_loss /= max(n, 1)

        model.eval()
        va_unw, va_w, m = 0.0, 0.0, 0
        vd = {"cup": [], "disc": [], "rim": [], "mean": []}
        va_u = []
        with torch.no_grad():
            for inputs, y in val_loader:
                y = y.to(device)
                logits, probs = probs_of(inputs)
                U = uncertainty_mc(uq, probs, qhat, cfg["entropy_normalize"])
                va_unw += criterion(logits, y, None).item()
                va_w += criterion(logits, y, U).item()     # U from epoch 1, as in the binary pipeline
                for k, v in class_dice(probs, y).items():
                    vd[k].extend(v.tolist())
                if U is not None and active:
                    va_u.append(float(U.mean()))
                m += 1
        va_unw, va_w = va_unw / max(m, 1), va_w / max(m, 1)

        ts = {f"{h}_{k}": [] for h in ("soft", "hard") for k in ("cup", "disc", "rim", "mean")}
        with torch.no_grad():
            for inputs, y in test_loader:
                y = y.to(device)
                _, probs = probs_of(inputs)
                for hard in (False, True):
                    for k, v in class_dice(probs, y, hard).items():
                        ts[f"{'hard' if hard else 'soft'}_{k}"].extend(v.tolist())
        mean = lambda a: float(np.mean(a)) if a else float("nan")

        row = {"epoch": epoch + 1, "loss_type": LOSS_DIR_MC[cfg["weight_form"]],
               "uncertainty_type": uq, "gamma": cfg["gamma"], "n_warmup": W, "uq_active": active,
               "train_loss": tr_loss,
               "train_uncertainty": mean(tr_u), "val_uncertainty": mean(va_u),
               "val_loss_unweighted": va_unw, "val_loss_weighted": va_w,
               "val_soft_dice": mean(vd["mean"]), "val_cup_soft_dice": mean(vd["cup"]),
               "val_disc_soft_dice": mean(vd["disc"]), "val_rim_soft_dice": mean(vd["rim"]),
               "test_soft_dice": mean(ts["soft_mean"]), "test_hard_dice": mean(ts["hard_mean"]),
               **{f"test_{k}_soft_dice": mean(ts[f"soft_{k}"]) for k in ("cup", "disc", "rim")},
               **{f"test_{k}_hard_dice": mean(ts[f"hard_{k}"]) for k in ("cup", "disc", "rim")},
               "seconds": round(time.time() - t0, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                              if torch.cuda.is_available() else float("nan")}

        for k, v in (("unweighted", va_unw), ("weighted", va_w), ("dice", -row["val_soft_dice"])):
            c = crit[k]
            if done[k]:
                continue
            if v < c[0]:
                c[0], c[1], c[2] = v, epoch + 1, 0
                if k == primary:
                    torch.save(model.state_dict(), ckpt)
                    qhat_primary = qhat
            else:
                c[2] += 1
            if c[2] >= pat:
                done[k] = True
            row[f"best_epoch_{k}"], row[f"patience_{k}"] = c[1], c[2]
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
        print(f"[epoch {epoch + 1}] train {tr_loss:.4f} | val unw {va_unw:.4f} w {va_w:.4f} | "
              f"val dice cup {row['val_cup_soft_dice']:.4f} disc {row['val_disc_soft_dice']:.4f} | "
              f"test cup {row['test_cup_soft_dice']:.4f} disc {row['test_disc_soft_dice']:.4f} | "
              f"U {row['val_uncertainty']:.4g} | {row['seconds']}s | peak GPU {row['peak_gpu_gb']} GB")
        if all(done.values()):
            break

        if uq == "conformal" and (epoch + 1) % W == 0:
            qhat, stats = recalibrate_mc(model, cal_loader, float(cfg["alpha"]), device, prep,
                                         clip_level=cfg["clip_conformal_level"])
            conformal_log.append({"after_epoch": epoch + 1, **stats})
            with open(os.path.join(run_dir, "conformal_log.json"), "w") as f:
                json.dump(conformal_log, f, indent=2)

    # ---- test with the primary-criterion checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    res = evaluate(model, test_loader, criterion, uq, qhat_primary, cfg, device, run_dir, prep)
    best = crit[primary][1]
    write_test_metrics(run_dir, res, len(test_loader.dataset), primary, best)
    update_manifest(run_dir, finished=True, best_epoch=best, epochs_run=len(rows),
                    primary_criterion=primary,
                    shadow_best_epochs={k: v[1] for k, v in crit.items()})
    if not cfg["keep_checkpoint"]:
        os.remove(ckpt)
    print(f"[done] {run_dir} | test cup {res['soft_cup']:.4f} disc {res['soft_disc']:.4f}")
    return run_dir


@torch.no_grad()
def evaluate(model, loader, criterion, uq, qhat, cfg, device, run_dir, prep):
    per = {f"{h}_{k}": [] for h in ("soft", "hard") for k in ("cup", "disc", "rim", "mean")}
    losses = []
    auroc = AUROCAccumulator(seed=0)
    for inputs, y in loader:
        y = y.to(device)
        logits = forward_logits(model, prep(inputs))
        probs = logits.softmax(1)
        losses.append(criterion(logits, y, None).item())
        for hard in (False, True):
            for k, v in class_dice(probs, y, hard).items():
                per[f"{'hard' if hard else 'soft'}_{k}"].extend(v.tolist())
        U = uncertainty_mc(uq, probs, qhat, cfg["entropy_normalize"])
        if U is not None:
            err = (probs.argmax(1) != y).float().unsqueeze(1)
            auroc.add(err, torch.zeros_like(err), U)      # error map vs 0 = the error itself
    pd.DataFrame({k: v for k, v in per.items()}).to_csv(
        os.path.join(run_dir, "test_per_image.csv"), index=False)
    out = {k: float(np.mean(v)) for k, v in per.items()}
    out["test_loss"] = float(np.mean(losses))
    out["uq_error_auroc"] = auroc.compute()
    return out


def write_test_metrics(run_dir, r, n, primary, best_epoch):
    with open(os.path.join(run_dir, "test_metrics.txt"), "w") as f:
        f.write("=== TEST METRICS (3-class: background / rim / cup) ===\n")
        f.write(f"Primary criterion: {primary}, selected epoch {best_epoch}\n")
        f.write(f"Test Loss: {r['test_loss']:.6f}\n")
        f.write(f"Foreground Dice Score: {r['soft_mean']:.6f}\n")   # mean of cup and disc (soft)
        f.write(f"Cup Dice Score: {r['soft_cup']:.6f}\n")
        f.write(f"Disc Dice Score: {r['soft_disc']:.6f}\n")
        f.write(f"Rim Dice Score: {r['soft_rim']:.6f}\n")
        f.write(f"Hard Dice cup / disc / rim: {r['hard_cup']:.6f} / {r['hard_disc']:.6f} / "
                f"{r['hard_rim']:.6f}\n")
        f.write(f"UQ-Error AUROC: {r['uq_error_auroc']:.6f}\n")
        f.write(f"Number of test samples: {n}\n")
