"""
uncertainty.py

Per-pixel uncertainty maps, reproducing the paper-era `get_uncertainty_map`
(uat_old/src/train.py). All maps are returned with shape [B, H, W].

  entropy:   U = -p log p - (1-p) log(1-p), natural log, NOT normalized
             (max ln 2 ~ 0.693). This matches the paper's equation and the
             paper-era code. The later refactor divided by ln 2; that is
             available as entropy_normalize=True but is NOT the default.
             (docs/DECISIONS.md, D3)

  sam_var:   produced in the forward pass (forward.forward_probabilities).

  conformal: LAC prediction sets with a PER-PIXEL threshold qhat [1, H, W]
             (see conformal.py). Set = {k : p_k >= 1 - qhat}; U = 1 where the
             set contains both classes, else 0. Before the first calibration
             (qhat is None) U = 0, i.e. no weighting. (paper-era behavior)

None of these maps are detached here; whether gradients flow through U is
decided in the training loop (`detach_uncertainty`, default False = paper-era).
"""

import math

import torch

UQ_TYPES = ("none", "entropy", "sam_var", "conformal")


def entropy_map(probabilities, epsilon=1e-6, normalize=False):
    p = torch.clamp(probabilities, min=epsilon, max=1 - epsilon)
    h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    if normalize:
        h = h / math.log(2.0)
    return h.squeeze(1)


def conformal_map(probabilities, qhat):
    if qhat is None:
        return torch.zeros_like(probabilities[:, 0])
    if probabilities.shape[1] == 1:
        probabilities = torch.cat([1 - probabilities, probabilities], dim=1)
    q = qhat.to(probabilities.device)
    prediction_sets = probabilities >= (1 - q)          # q broadcasts [1,H,W] over [B,2,H,W]
    set_sizes = prediction_sets.sum(dim=1)              # [B, H, W]
    return (set_sizes == 2).float()


def uncertainty_map(uq_type, probabilities, variance_map=None, qhat=None,
                    entropy_normalize=False):
    """Dispatch. Returns [B, H, W] or None for the baseline."""
    if uq_type in (None, "none"):
        return None
    if uq_type == "entropy":
        return entropy_map(probabilities, normalize=entropy_normalize)
    if uq_type == "sam_var":
        if variance_map is None:
            raise RuntimeError(
                "sam_var arm without a variance map. The forward pass must be "
                "run with uq_type='sam_var' (multimask_output=True). "
                "This is exactly the refactor bug documented in ERRATA_LOG E2."
            )
        return variance_map
    if uq_type == "conformal":
        return conformal_map(probabilities, qhat)
    raise ValueError(f"Unknown uq_type {uq_type}; choose from {UQ_TYPES}")
