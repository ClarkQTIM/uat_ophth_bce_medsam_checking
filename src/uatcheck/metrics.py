"""
metrics.py

PAPER METRIC (reproduced exactly):
  soft_dice_per_image: the paper-era StandardDiceLoss(loss=False) applied to
  PROBABILITIES (not thresholded masks), per image and channel,
  smooth = 1e-5 in numerator and denominator. The paper's "Foreground Dice Score"
  is the mean of this over test images.

ADDITIONAL METRICS (new; reported alongside, never replacing the paper metric):
  hard_dice_per_image / iou: on masks thresholded at 0.5
  uq_error_dice: paper-era definition (soft Dice between the binary error map and
     the min-max normalized uncertainty map of the batch)
  uq_error_auroc: threshold-free: how well U ranks error pixels above correct
     pixels (pooled over the split). Added because the paper-era UQ-error Dice
     depends on the scale of U (docs/KNOWN_ISSUES.md).
"""

import numpy as np
import torch


def soft_dice_per_image(y_pred, y_true, smooth_nr=1e-5, smooth_dr=1e-5):
    dims = list(range(2, y_pred.dim()))
    inter = torch.sum(y_true * y_pred, dim=dims)
    denom = torch.sum(y_true + y_pred, dim=dims)
    return (2.0 * inter + smooth_nr) / (denom + smooth_dr)       # [B, C]


def hard_dice_per_image(y_pred, y_true, threshold=0.5, smooth=1e-6):
    return soft_dice_per_image((y_pred > threshold).float(), y_true, smooth, smooth)


def iou_per_image(y_pred, y_true, threshold=0.5, smooth=1e-6):
    p = (y_pred > threshold).float()
    dims = list(range(2, p.dim()))
    inter = torch.sum(p * y_true, dim=dims)
    union = torch.sum(p, dim=dims) + torch.sum(y_true, dim=dims) - inter
    return (inter + smooth) / (union + smooth)


def uq_error_dice(predictions, ground_truth, uncertainty_map, threshold=0.5):
    """Paper-era calculate_uq_error_dice (uat_old/src/train.py), verbatim logic."""
    if uncertainty_map is None:
        return 0.0
    pred_binary = (predictions > threshold).float()
    error_map = (pred_binary != ground_truth).float()
    u = uncertainty_map
    if u.shape != error_map.shape:
        if u.dim() == 3:
            u = u.unsqueeze(1)
        u = u.expand_as(error_map)
    lo, hi = u.min(), u.max()
    u_n = (u - lo) / (hi - lo) if hi > lo else u
    inter = (error_map * u_n).sum()
    union = error_map.sum() + u_n.sum()
    return float((2.0 * inter + 1e-5) / (union + 1e-5))


class AUROCAccumulator:
    """Pooled pixel AUROC of uncertainty vs error, with optional subsampling."""

    def __init__(self, max_pixels_per_batch=20000, seed=0):
        self.u, self.e = [], []
        self.k = max_pixels_per_batch
        self.rng = np.random.default_rng(seed)

    def add(self, predictions, ground_truth, uncertainty_map, threshold=0.5):
        if uncertainty_map is None:
            return
        err = ((predictions > threshold).float() != ground_truth).float().flatten()
        u = uncertainty_map
        if u.dim() == 3:
            u = u.unsqueeze(1)
        u = u.expand_as(predictions).flatten()
        err, u = err.detach().cpu().numpy(), u.detach().cpu().numpy()
        if len(err) > self.k:
            idx = self.rng.choice(len(err), self.k, replace=False)
            err, u = err[idx], u[idx]
        self.e.append(err)
        self.u.append(u)

    def compute(self):
        if not self.e:
            return float("nan")
        e, u = np.concatenate(self.e), np.concatenate(self.u)
        if e.min() == e.max():
            return float("nan")
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(e, u))
