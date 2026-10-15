"""
train_uat.py

One training run of uncertainty-aware training (UAT), reproducing the paper-era
pipeline `train_segmentation_pipeline` (uat_old/src/train.py) except where
docs/DECISIONS.md says otherwise. Every deviation is listed in the docstring of
the relevant function and in DECISIONS.md.

Per epoch (paper-era order):
  1. train:  forward -> U -> (U = None during warmup: epoch_index < n_warmup)
             -> weighted BCE -> step
  2. val:    forward -> U (for logging only) -> UNWEIGHTED BCE
  3. early stopping on the unweighted val loss (patience 5), best checkpoint kept
  4. conformal arm: recalibrate qhat when (epoch_index + 1) % n_warmup == 0
After training: reload the best checkpoint, evaluate on test.

Validation criteria (DECISIONS D16, ERRATA E14). Every epoch logs three:
  val_loss_unweighted   BCE on the validation set (what the paper's Methods describe)
  val_loss_weighted     BCE weighted by U with the same gamma and formula, U computed
                        on validation from epoch 1 (no warmup gating) = what the
                        paper-era grid code actually used for early stopping
  val_soft_dice         validation soft Dice (a criterion common to all arms)
`val_weighting` ("none" | "paper_era") picks which loss drives the PRIMARY early
stopping and checkpoint (written to test_metrics.txt / loss_parameters.csv).
With `track_all_criteria: true` (default) training continues until EVERY criterion
has exhausted its patience (or num_epochs), and test soft/hard Dice are evaluated
every epoch. Stopping does not affect training, so scripts/select_offline.py can
then reproduce exactly what early stopping under each criterion would have selected.

Outputs (paper-compatible, so the analysis scripts in analysis/ read them):
  loss_parameters.csv   epoch rows + 'early_stop' row + 'test' row
                        (paper-era quirk kept: when early stopping triggers, the
                        triggering epoch is NOT written here; see D8)
  epoch_log.csv         every epoch, complete (new)
  test_metrics.txt      contains "Foreground Dice Score: x" (paper metric)
  run_manifest.json     what ran: backbone id + param count, git commit, config
"""

import csv
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from .backbones import backbone_dir_tag, load_backbone
from .conformal import recalibrate
from .data import make_loaders
from .forward import forward_probabilities, prepare_inputs
from .losses import WeightedBCELoss
from .manifest import update_manifest, write_manifest
from .metrics import (AUROCAccumulator, hard_dice_per_image, iou_per_image,
                      soft_dice_per_image, uq_error_dice)
from .uncertainty import uncertainty_map

LOSS_DIR = {"paper": "weighted_bce_v2", "gm1": "weighted_bce_v2_gm1",
            "exp": "weighted_bce_v2_exp"}

DEFAULTS = {
    # --- what to run ---
    "dataset": None,             # cup | ga | octa
    "csv": None,                 # split CSV (image, segmentation, split)
    "mask_max_value": 255.0,
    "backbone": "medsam",        # sam-vit-huge | medsam
    "backbone_repo_id": None,
    "processor_id": None,
    "uq_type": "none",           # none | entropy | sam_var | conformal
    "alpha": None,               # conformal only
    "gamma": 0.0,
    "n_warmup": 0,
    "run": 1,
    # --- fidelity switches (defaults = paper-era behavior; see DECISIONS.md) ---
    "weight_form": "paper",                 # D2
    "entropy_normalize": False,             # D3
    "variance_prediction_token": "multimask0",  # D4
    "detach_uncertainty": False,            # D5
    "clip_conformal_level": False,          # D6
    "val_weighting": "none",                # D16: "none" (paper text) | "paper_era" (as run)
    "track_all_criteria": True,             # D16: run until all criteria stop; test every epoch
    # --- training (paper Methods) ---
    "num_epochs": 40,
    "patience": 5,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "mask_size": [256, 256],
    # --- bookkeeping ---
    "output_root": "results",
    "base_seed": 1000,
    "deterministic": False,
    "num_workers": 4,
    "limit_samples": None,       # smoke tests only
    "keep_checkpoint": False,
    "save_test_tensors": True,
    "save_epoch_images": True,
    "device": None,
}


def arm_dir_name(uq_type, alpha):
    if uq_type in (None, "none"):
        return "none"
    if uq_type == "conformal":
        return f"conformal_lac_{int(round(alpha * 100)):03d}"
    return uq_type


def _num(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


def run_dir_for(cfg):
    return os.path.join(
        cfg["output_root"], cfg["dataset"], backbone_dir_tag(cfg["backbone"]),
        LOSS_DIR[cfg["weight_form"]], f"gamma_{_num(cfg['gamma'])}",
        f"n_warmup_{int(cfg['n_warmup'])}", arm_dir_name(cfg["uq_type"], cfg["alpha"]),
        f"run_{int(cfg['run'])}")


def validate_config(cfg):
    for k in ("dataset", "csv"):
        if not cfg.get(k):
            raise ValueError(f"config key '{k}' is required")
    if cfg["uq_type"] == "conformal" and cfg["alpha"] is None:
        raise ValueError("conformal arm needs alpha")
    if cfg["uq_type"] != "none" and int(cfg["n_warmup"]) < 1:
        raise ValueError("UQ arms need n_warmup >= 1 (paper grid: 1..5)")
    if cfg["val_weighting"] not in ("none", "paper_era"):
        raise ValueError("val_weighting must be 'none' or 'paper_era'")
    if cfg["uq_type"] == "none" and (float(cfg["gamma"]) != 0 or int(cfg["n_warmup"]) != 0):
        raise ValueError("baseline must use gamma=0, n_warmup=0 (folder convention)")


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _save_images(run_dir, epoch, phase, seg, probs, U):
    try:
        from torchvision.utils import save_image
        d = os.path.join(run_dir, "epoch_images", f"epoch_{epoch + 1}")
        os.makedirs(d, exist_ok=True)
        save_image(seg[:4], os.path.join(d, f"{phase}_segmentations.png"))
        save_image(probs[:4], os.path.join(d, f"{phase}_predicted_masks.png"))
        if U is not None:
            torch.save(U[:4].detach().cpu(), os.path.join(d, f"{phase}_uncertainty.pt"))
            save_image(U[:4].unsqueeze(1), os.path.join(d, f"{phase}_uncertainty_map.png"),
                       normalize=True, scale_each=True)
    except Exception as e:  # never let a figure kill a run
        print(f"[images] skipped: {e}")


def train_one_run(cfg_in):
    cfg = {**DEFAULTS, **cfg_in}
    validate_config(cfg)
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["base_seed"]) + int(cfg["run"])
    set_seed(seed, cfg["deterministic"])

    run_dir = run_dir_for(cfg)
    if os.path.exists(os.path.join(run_dir, "test_metrics.txt")):
        print(f"[skip] already complete: {run_dir}")
        return run_dir
    os.makedirs(run_dir, exist_ok=True)

    model, processor, binfo = load_backbone(
        cfg["backbone"], cfg["backbone_repo_id"], cfg["processor_id"], device)
    write_manifest(run_dir, {**cfg, "seed": seed, "device": device}, binfo)

    train_loader, val_loader, cal_loader, test_loader = make_loaders(
        cfg["csv"], processor, cfg["batch_size"], tuple(cfg["mask_size"]),
        cfg["num_workers"], cfg["limit_samples"], cfg["mask_max_value"], seed)

    criterion = WeightedBCELoss(cfg["gamma"], cfg["weight_form"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    uq = cfg["uq_type"]
    W = int(cfg["n_warmup"])

    def fwd(inputs):
        return forward_probabilities(model, prepare_inputs(inputs, device), uq,
                                     cfg["variance_prediction_token"])

    params_rows, epoch_rows, conformal_log = [], [], []
    best_val, best_epoch, patience_counter = math.inf, 0, 0
    qhat, early_stopped, stopped_at = None, False, None
    primary_done = False
    qhat_primary = None          # conformal threshold in force when the primary criterion stopped
    # shadow early stopping for every criterion (sign: lower is better)
    crit = {"unweighted": [math.inf, 0, 0], "weighted": [math.inf, 0, 0], "dice": [math.inf, 0, 0]}
    crit_done = {k: False for k in crit}
    ckpt = os.path.join(run_dir, "best_model.pth")
    loss_name = LOSS_DIR[cfg["weight_form"]]

    for epoch in range(int(cfg["num_epochs"])):
        t0 = time.time()
        active = uq != "none" and epoch >= W
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---------------- train ----------------
        model.train()
        tr_loss, tr_uqd, tr_u, n = 0.0, 0.0, [], 0
        for b, (inputs, seg) in enumerate(train_loader):
            seg = seg.to(device)
            optimizer.zero_grad()
            probs, var_map = fwd(inputs)
            U = uncertainty_map(uq, probs, var_map, qhat, cfg["entropy_normalize"])
            if not active:
                U = None
            if U is not None and cfg["detach_uncertainty"]:
                U = U.detach()
            tr_uqd += uq_error_dice(probs.detach(), seg, None if U is None else U.detach())
            loss = criterion(probs, seg, U)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            if U is not None:
                tr_u.append(float(U.detach().mean()))
            n += 1
        tr_loss /= max(n, 1)
        tr_uqd /= max(n, 1)

        # ---------------- val ----------------
        model.eval()
        va_unw, va_w, va_uqd, va_u, va_dice, m = 0.0, 0.0, 0.0, [], [], 0
        with torch.no_grad():
            for b, (inputs, seg) in enumerate(val_loader):
                seg = seg.to(device)
                probs, var_map = fwd(inputs)
                U = uncertainty_map(uq, probs, var_map, qhat, cfg["entropy_normalize"])
                va_uqd += uq_error_dice(probs, seg, U)
                va_unw += criterion(probs, seg, None).item()        # paper text
                va_w += criterion(probs, seg, U).item()             # paper-era code (D16)
                va_dice.extend(soft_dice_per_image(probs, seg)[:, 0].tolist())
                if U is not None and active:
                    va_u.append(float(U.mean()))
                if b == 0 and cfg["save_epoch_images"]:
                    _save_images(run_dir, epoch, "val", seg, probs, U)
                m += 1
        va_unw /= max(m, 1)
        va_w /= max(m, 1)
        va_uqd /= max(m, 1)
        va_loss = va_w if cfg["val_weighting"] == "paper_era" else va_unw
        va_dice_mean = float(np.mean(va_dice)) if va_dice else float("nan")

        # ---------------- test every epoch (offline selection) ----------------
        te_soft, te_hard = float("nan"), float("nan")
        if cfg["track_all_criteria"]:
            with torch.no_grad():
                ts, th = [], []
                for inputs, seg in test_loader:
                    probs, _ = fwd(inputs)
                    seg = seg.to(device)
                    ts.extend(soft_dice_per_image(probs, seg)[:, 0].tolist())
                    th.extend(hard_dice_per_image(probs, seg)[:, 0].tolist())
            te_soft, te_hard = float(np.mean(ts)), float(np.mean(th))

        row = {
            "epoch": epoch + 1, "loss_type": loss_name, "uncertainty_type": uq,
            "gamma": cfg["gamma"], "n_warmup": W, "uq_active": active,
            "train_loss": tr_loss, "train_uq_error_dice": tr_uqd,
            "train_uncertainty": float(np.mean(tr_u)) if tr_u else float("nan"),
            "val_loss": va_loss, "val_loss_unweighted": va_unw, "val_loss_weighted": va_w,
            "val_uq_error_dice": va_uqd,
            "val_uncertainty": float(np.mean(va_u)) if va_u else float("nan"),
            "val_soft_dice": va_dice_mean,
            "test_soft_dice": te_soft, "test_hard_dice": te_hard,
            "seconds": round(time.time() - t0, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                           if torch.cuda.is_available() else float("nan"),
        }

        # ---------------- early stopping (paper-era logic) ----------------
        if not primary_done:
            if va_loss < best_val:
                best_val, best_epoch, patience_counter = va_loss, epoch + 1, 0
                torch.save(model.state_dict(), ckpt)
            else:
                patience_counter += 1
        # shadow criteria
        for k, v in (("unweighted", va_unw), ("weighted", va_w), ("dice", -va_dice_mean)):
            c = crit[k]
            if crit_done[k]:
                continue
            if v < c[0]:
                c[0], c[1], c[2] = v, epoch + 1, 0
            else:
                c[2] += 1
            if c[2] >= int(cfg["patience"]):
                crit_done[k] = True
            row[f"best_epoch_{k}"] = c[1]
            row[f"patience_{k}"] = c[2]
        row["patience_counter"] = patience_counter
        row["best_epoch_so_far"] = best_epoch
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
        print(f"[epoch {epoch + 1}] train {tr_loss:.4f} | val unw {va_unw:.4f} w {va_w:.4f} | "
              f"val dice {va_dice_mean:.4f} | test dice {te_soft:.4f} | U {row['val_uncertainty']:.4g} | "
              f"patience {patience_counter}/{cfg['patience']} | {row['seconds']}s | "
              f"peak GPU {row['peak_gpu_gb']} GB")

        if not primary_done:
            if patience_counter >= int(cfg["patience"]):
                primary_done, early_stopped, stopped_at = True, True, epoch + 1
                qhat_primary = qhat
                params_rows.append({"epoch": "early_stop", "stopped_at_epoch": epoch + 1,
                                    "best_epoch": best_epoch, "best_val_loss": best_val,
                                    "patience_used": cfg["patience"]})
            else:
                params_rows.append(row)   # paper-era: the triggering epoch is not written (D8)
            pd.DataFrame(params_rows).to_csv(os.path.join(run_dir, "loss_parameters.csv"), index=False)

        if primary_done and (not cfg["track_all_criteria"] or all(crit_done.values())):
            break

        # ---------------- conformal recalibration ----------------
        if uq == "conformal" and (epoch + 1) % W == 0:
            qhat, stats = recalibrate(
                model, cal_loader, float(cfg["alpha"]), device,
                lambda inp: forward_probabilities(model, prepare_inputs(inp, device), "none")[0],
                clip_level=cfg["clip_conformal_level"])
            conformal_log.append({"after_epoch": epoch + 1, **stats})
            with open(os.path.join(run_dir, "conformal_log.json"), "w") as f:
                json.dump(conformal_log, f, indent=2)

    # ---------------- test with the best checkpoint ----------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    results = evaluate(model, test_loader, criterion, uq,
                       qhat_primary if primary_done else qhat, cfg, device, run_dir)

    # validation Dice of the selected checkpoint (for validation-based model selection)
    with torch.no_grad():
        vd = []
        for inputs, seg in val_loader:
            probs, _ = fwd(inputs)
            vd.extend(soft_dice_per_image(probs, seg.to(device))[:, 0].tolist())
    results["val_dice_at_best"] = float(np.mean(vd)) if vd else float("nan")

    test_row = {"epoch": "test", "loss_type": loss_name, "uncertainty_type": uq,
                "gamma": cfg["gamma"], "n_warmup": W,
                "test_loss": results["test_loss"],
                "test_dice_foreground": results["dice_foreground"],
                "test_dice_background": results["dice_background"],
                "test_dice_avg": results["dice_avg"],
                "test_hard_dice": results["hard_dice"], "test_iou": results["iou"],
                "test_uq_error_dice": results["uq_error_dice"],
                "test_uq_error_auroc": results["uq_error_auroc"],
                "val_dice_at_best": results["val_dice_at_best"],
                "trained_epochs": best_epoch, "stopped_at_epoch": stopped_at,
                "early_stopped": early_stopped, "val_weighting": cfg["val_weighting"],
                "epochs_run": len(epoch_rows)}
    params_rows.append(test_row)
    pd.DataFrame(params_rows).to_csv(os.path.join(run_dir, "loss_parameters.csv"), index=False)
    write_test_metrics(run_dir, results, len(test_loader.dataset))

    size_gb = os.path.getsize(ckpt) / 1024 ** 3
    update_manifest(run_dir, checkpoint_size_gb=round(size_gb, 3),
                    finished=True, best_epoch=best_epoch, stopped_at_epoch=stopped_at,
                    epochs_run=len(epoch_rows),
                    shadow_best_epochs={k: v[1] for k, v in crit.items()})
    if not cfg["keep_checkpoint"]:
        os.remove(ckpt)
    print(f"[done] {run_dir} | test soft Dice {results['dice_foreground']:.4f}")
    return run_dir


@torch.no_grad()
def evaluate(model, loader, criterion, uq, qhat, cfg, device, run_dir):
    """Paper-era evaluate_model_on_test, plus extra metrics. For sam_var the test
    uncertainty is the variance map (the paper-era function would have raised
    here; see ERRATA_LOG E2 note)."""
    losses, soft, hard, iou, uqd = [], [], [], [], []
    auroc = AUROCAccumulator(seed=0)
    preds, gts, us = [], [], []
    for inputs, seg in loader:
        seg = seg.to(device)
        probs, var_map = forward_probabilities(model, prepare_inputs(inputs, device), uq,
                                               cfg["variance_prediction_token"])
        U = uncertainty_map(uq, probs, var_map, qhat, cfg["entropy_normalize"])
        losses.append(criterion(probs, seg, None).item())
        soft.extend(soft_dice_per_image(probs, seg)[:, 0].tolist())
        hard.extend(hard_dice_per_image(probs, seg)[:, 0].tolist())
        iou.extend(iou_per_image(probs, seg)[:, 0].tolist())
        uqd.append(uq_error_dice(probs, seg, U))
        auroc.add(probs, seg, U)
        if cfg["save_test_tensors"]:
            preds.append(probs.cpu())
            gts.append(seg.cpu())
            if U is not None:
                us.append(U.cpu())
    if cfg["save_test_tensors"]:
        torch.save(torch.cat(preds), os.path.join(run_dir, "test_predictions_tensor.pt"))
        torch.save(torch.cat(gts), os.path.join(run_dir, "test_ground_truth_tensor.pt"))
        if us:
            torch.save(torch.cat(us), os.path.join(run_dir, "test_uncertainty_tensor.pt"))
    pd.DataFrame({"soft_dice": soft, "hard_dice": hard, "iou": iou}).to_csv(
        os.path.join(run_dir, "test_per_image.csv"), index=False)
    fg = float(np.mean(soft))
    return {
        "test_loss": float(np.mean(losses)),
        # single-channel output: paper-era code reported foreground for both
        "dice_foreground": fg, "dice_background": fg, "dice_avg": fg,
        "hard_dice": float(np.mean(hard)), "iou": float(np.mean(iou)),
        "uq_error_dice": float(np.mean(uqd)) if uq != "none" else 0.0,
        "uq_error_auroc": auroc.compute(),
    }


def write_test_metrics(run_dir, r, n):
    with open(os.path.join(run_dir, "test_metrics.txt"), "w") as f:
        f.write("=== TEST METRICS ===\n")
        f.write(f"Test Loss: {r['test_loss']:.6f}\n")
        f.write(f"Average Dice Score: {r['dice_avg']:.6f}\n")
        f.write(f"Foreground Dice Score: {r['dice_foreground']:.6f}\n")
        f.write(f"Background Dice Score: {r['dice_background']:.6f}\n")
        f.write(f"UQ-Error Dice Score: {r['uq_error_dice']:.6f}\n")
        f.write(f"Number of test samples: {n}\n")
        f.write("--- additional (not in paper) ---\n")
        f.write(f"Hard Dice (threshold 0.5): {r['hard_dice']:.6f}\n")
        f.write(f"IoU (threshold 0.5): {r['iou']:.6f}\n")
        f.write(f"UQ-Error AUROC: {r['uq_error_auroc']:.6f}\n")
        f.write(f"Validation soft Dice at selected checkpoint: {r['val_dice_at_best']:.6f}\n")
