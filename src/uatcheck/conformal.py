"""
conformal.py

LAC conformal calibration for binary segmentation, reproducing the paper-era
functions in uat_old/src/conformal.py:

  get_calibration_scores_lac:  s = 1 - p(true class), per pixel
  find_quantile_segmentation:  qhat = per-pixel quantile ACROSS calibration
                               images at level ceil((n+1)(1-alpha))/n

Note on small calibration sets (docs/ERRATA_LOG.md, E6): the quantile level
ceil((n+1)(1-alpha))/n exceeds 1 when n is small. For GA (n_cal = 10) and
alpha = 0.05 the level is 11/10 = 1.1 and numpy raises. This is consistent with
the paper having no GA Conf-0.05 arm in Figure 5. We reproduce the paper-era
behavior (raise, with a clear message) by default; `clip_level=True` exists
only for exploratory runs and is NOT used for reproduction.
"""

import math

import numpy as np
import torch


def quantile_level(n_cal, alpha):
    return math.ceil((n_cal + 1) * (1 - alpha)) / n_cal


def calibration_scores_lac(y_cal, cal_probs):
    """
    y_cal: [N, 1, H, W] or [N, H, W] binary labels (float or int)
    cal_probs: [N, 1, H, W] foreground probabilities (or [N, C, H, W])
    Returns scores [N, H, W] (or [N, 1, H, W] following y_cal's shape).
    """
    if cal_probs.shape[1] == 1:
        cal_probs = torch.cat([1 - cal_probs, cal_probs], dim=1)
    y = y_cal.long()
    flat_probs = cal_probs.contiguous().view(cal_probs.shape[0], cal_probs.shape[1], -1)
    flat_y = y.contiguous().view(y.shape[0], -1)
    p_true = flat_probs.gather(1, flat_y.unsqueeze(1))
    return 1 - p_true.view_as(y).float()


def find_quantile_segmentation(cal_scores, alpha, clip_level=False):
    n_cal = cal_scores.shape[0]
    s = cal_scores.detach().cpu().numpy()
    if s.ndim == 4:
        s = s.squeeze(1)
    flat = s.reshape(n_cal, -1)
    level = quantile_level(n_cal, alpha)
    if level > 1.0:
        if clip_level:
            level = 1.0
        else:
            raise ValueError(
                f"Conformal quantile level ceil((n+1)(1-alpha))/n = {level:.3f} > 1 "
                f"(n_cal={n_cal}, alpha={alpha}). This configuration cannot be "
                f"calibrated; the paper-era code failed here too (ERRATA_LOG E6)."
            )
    qhat = np.quantile(flat, level, axis=0, method="linear")
    qhat = qhat.reshape(1, *s.shape[1:])
    return torch.tensor(qhat, dtype=torch.float32), level


@torch.no_grad()
def recalibrate(model, cal_loader, alpha, device, forward_fn, clip_level=False):
    """Run the model on the calibration split and return (qhat, stats)."""
    model.eval()
    probs, labels = [], []
    for inputs, seg in cal_loader:
        p = forward_fn(inputs)
        probs.append(p.cpu())
        labels.append(seg.cpu())
    cal_probs = torch.cat(probs, dim=0)
    cal_labels = torch.cat(labels, dim=0)
    scores = calibration_scores_lac(cal_labels, cal_probs)
    qhat, level = find_quantile_segmentation(scores, alpha, clip_level=clip_level)
    stats = {
        "qhat_level": level, "n_cal": int(cal_probs.shape[0]),
        "qhat_min": float(qhat.min()), "qhat_max": float(qhat.max()),
        "qhat_mean": float(qhat.mean()),
    }
    print(f"[conformal] recalibrated: level={level:.4f} n_cal={stats['n_cal']} "
          f"qhat mean={stats['qhat_mean']:.4f} (min {stats['qhat_min']:.4f}, "
          f"max {stats['qhat_max']:.4f})")
    return qhat, stats
