"""
Training for the evidential track (project outline v2, section 4). UNWEIGHTED:
no uncertainty-aware weighting here.

  model "A": STAPLE hard labels, softmax + CE
  model "B": two-rater soft target q, softmax + soft CE
  model "C": two-rater soft target q, Dirichlet (alpha = softplus(z) + offset) + EDL loss

All three use the same MedSAM forward pass and the same three output channels
(the three multimask tokens, read as BG / rim / cup), so they differ only in
activation and loss.

Early stopping: A/B on validation loss; C on the lambda-free "fit" part of the
EDL loss (the annealed KL makes the total rise mechanically).

STATUS: set up, not yet run. Run only after the literature review; see README.md.
"""

import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..backbones import backbone_dir_tag, load_backbone
from ..forward import forward_multimask_logits, prepare_inputs
from ..manifest import update_manifest, write_manifest
from .data import TwoRaterPapila
from .losses import edl_loss, kl_schedule, soft_ce

DEFAULTS = {
    "csv": None, "model": "C", "backbone": "medsam", "backbone_repo_id": None,
    "processor_id": None, "mask_max_value": 255.0, "mask_size": [256, 256],
    "lambda0": 1e-4, "kl_scale_by_K": True, "alpha_offset": 0.0, "alpha_eps": 1e-6,
    "num_epochs": 40, "patience": 5, "batch_size": 1, "learning_rate": 1e-5,
    "weight_decay": 1e-5, "run": 1, "base_seed": 2000, "output_root": "results/evidential",
    "num_workers": 4, "limit_samples": None, "keep_checkpoint": False, "device": None,
}


def to_alpha(logits, cfg):
    return F.softplus(logits) + cfg["alpha_offset"] + cfg["alpha_eps"]


def probs_from(logits, cfg):
    if cfg["model"] == "C":
        a = to_alpha(logits, cfg)
        return a / a.sum(1, keepdim=True), a
    return F.softmax(logits, 1), None


def soft_dice(p, t, eps=1e-5):
    dims = (1, 2)
    return ((2 * (p * t).sum(dims) + eps) / ((p + t).sum(dims) + eps)).tolist()


def _auroc(score, label):
    from sklearn.metrics import roc_auc_score
    if label.min() == label.max():
        return float("nan")
    return float(roc_auc_score(label, score))


@torch.no_grad()
def evaluate(model, loader, cfg, device, epoch_for_lambda):
    model.eval()
    tot, fit_tot, n = 0.0, 0.0, 0
    disc_d, cup_d = [], []
    S_agree, S_dis, H_all, S_all, A_all = [], [], [], [], []
    rng = np.random.default_rng(0)
    for batch in loader:
        x = prepare_inputs(batch["processed"], device)
        q = batch["q"].to(device)
        logits = forward_multimask_logits(model, x)
        p, alpha = probs_from(logits, cfg)
        if cfg["model"] == "C":
            lam = kl_schedule(epoch_for_lambda, cfg["num_epochs"], cfg["lambda0"], 3, cfg["kl_scale_by_K"])
            out = edl_loss(q, alpha, lam)
            fit_tot += float(out["fit"])
        else:
            tgt = F.one_hot(batch["hard"].to(device), 3).permute(0, 3, 1, 2).float() \
                if cfg["model"] == "A" else q
            out = soft_ce(tgt, logits)
        tot += float(out["loss"])
        n += 1
        gt = F.one_hot(batch["hard"].to(device), 3).permute(0, 3, 1, 2).float()
        disc_d += soft_dice(p[:, 1] + p[:, 2], gt[:, 1] + gt[:, 2])
        cup_d += soft_dice(p[:, 2], gt[:, 2])
        H = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(1)       # [B, H, W]
        agree = batch["agree"].to(device)
        idx = rng.choice(agree.numel(), min(20000, agree.numel()), replace=False)
        H_all.append(H.flatten()[idx].cpu().numpy())
        A_all.append((~agree).flatten()[idx].cpu().numpy())
        if alpha is not None:
            S = alpha.sum(1)
            S_agree.append(float(S[agree].mean()) if agree.any() else float("nan"))
            S_dis.append(float(S[~agree].mean()) if (~agree).any() else float("nan"))
            S_all.append(S.flatten()[idx].cpu().numpy())
    res = {"loss": tot / max(n, 1), "disc_soft_dice": float(np.mean(disc_d)),
           "cup_soft_dice": float(np.mean(cup_d))}
    A = np.concatenate(A_all).astype(int)
    res["auroc_entropy_vs_disagreement"] = _auroc(np.concatenate(H_all), A)
    if cfg["model"] == "C":
        res["fit"] = fit_tot / max(n, 1)
        res["S_mean_agree"] = float(np.nanmean(S_agree))
        res["S_mean_disagree"] = float(np.nanmean(S_dis))
        res["auroc_lowS_vs_disagreement"] = _auroc(-np.concatenate(S_all), A)
    return res


def train_one_run(cfg_in):
    cfg = {**DEFAULTS, **cfg_in}
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["base_seed"] + int(cfg["run"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tag = f"model_{cfg['model']}" + (f"_lambda0_{cfg['lambda0']:g}" if cfg["model"] == "C" else "")
    run_dir = os.path.join(cfg["output_root"], "papila", backbone_dir_tag(cfg["backbone"]),
                           tag, f"run_{cfg['run']}")
    os.makedirs(run_dir, exist_ok=True)
    model, processor, info = load_backbone(cfg["backbone"], cfg["backbone_repo_id"],
                                           cfg["processor_id"], device)
    write_manifest(run_dir, {**cfg, "seed": seed, "device": device}, info)

    def loader(split, shuffle):
        ds = TwoRaterPapila(cfg["csv"], split, processor, cfg["mask_size"],
                            cfg["mask_max_value"], cfg["limit_samples"])
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                          num_workers=cfg["num_workers"])
    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    ckpt = os.path.join(run_dir, "best_model.pth")
    best, best_ep, bad, log = math.inf, 0, 0, []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        lam = kl_schedule(epoch, cfg["num_epochs"], cfg["lambda0"], 3, cfg["kl_scale_by_K"])
        tl = 0.0
        for batch in tr:
            x = prepare_inputs(batch["processed"], device)
            q = batch["q"].to(device)
            logits = forward_multimask_logits(model, x)
            if cfg["model"] == "C":
                out = edl_loss(q, to_alpha(logits, cfg), lam)
            elif cfg["model"] == "B":
                out = soft_ce(q, logits)
            else:
                out = soft_ce(F.one_hot(batch["hard"].to(device), 3).permute(0, 3, 1, 2).float(), logits)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            tl += float(out["loss"])
        v = evaluate(model, va, cfg, device, epoch)
        crit = v["fit"] if cfg["model"] == "C" else v["loss"]
        log.append({"epoch": epoch + 1, "lambda": lam, "train_loss": tl / max(len(tr), 1),
                    **{f"val_{k}": x for k, x in v.items()}})
        pd.DataFrame(log).to_csv(os.path.join(run_dir, "epoch_log.csv"), index=False)
        if crit < best:
            best, best_ep, bad = crit, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, te, cfg, device, best_ep - 1)
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump({"best_epoch": best_ep, **test}, f, indent=2)
    update_manifest(run_dir, finished=True, best_epoch=best_ep,
                    checkpoint_size_gb=round(os.path.getsize(ckpt) / 1024 ** 3, 3))
    if not cfg["keep_checkpoint"]:
        os.remove(ckpt)
    return run_dir
