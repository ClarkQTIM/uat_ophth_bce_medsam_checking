"""
losses.py

Uncertainty-weighted binary cross-entropy ("weighted_bce_v2" in the paper's
results tree), with the weight formula made EXPLICIT and configurable because
three different formulas exist across the code history (docs/ERRATA_LOG.md, E3):

  weight_form = "paper"   W = 1 + gamma * U            (paper Eq.; gamma=0 -> BCE)
  weight_form = "gm1"     W = 1 + (gamma - 1) * U      (refactor "V2"; gamma=1 -> BCE)
  weight_form = "exp"     W = gamma ** U               (uat_old; gamma=1 -> BCE)

DEFAULT is "paper": we assume the published equation is what ran
(docs/DECISIONS.md, D2). The gamma=1 formula check (configs/errata/
gamma1_formula_check.yaml) tests that assumption empirically.

The loss is a plain mean over all pixels, as in the paper-era code.
Validation/test always call it with weight_map=None (unweighted), as in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_FORMS = ("paper", "gm1", "exp")


def weight_from_uncertainty(U, gamma, weight_form):
    if weight_form == "paper":
        return 1.0 + gamma * U
    if weight_form == "gm1":
        return 1.0 + (gamma - 1.0) * U
    if weight_form == "exp":
        return torch.as_tensor(gamma, dtype=U.dtype, device=U.device) ** U
    raise ValueError(f"weight_form must be one of {WEIGHT_FORMS}")


class WeightedBCELoss(nn.Module):
    def __init__(self, gamma, weight_form="paper"):
        super().__init__()
        if weight_form not in WEIGHT_FORMS:
            raise ValueError(f"weight_form must be one of {WEIGHT_FORMS}")
        self.gamma = float(gamma)
        self.weight_form = weight_form

    def forward(self, y_pred, y_true, weight_map=None):
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        if weight_map is None:
            return bce.mean()
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)
        if weight_map.shape != y_pred.shape:
            weight_map = weight_map.expand_as(y_pred)
        w = weight_from_uncertainty(weight_map, self.gamma, self.weight_form)
        return (bce * w).mean()
