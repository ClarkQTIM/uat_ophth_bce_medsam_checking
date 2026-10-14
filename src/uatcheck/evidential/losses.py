"""
Losses for the evidential track. Shapes: q [B, K, H, W], alpha/logits [B, K, H, W].

edl_loss: the infectious-keratitis `dirichlet_loss` (Sensoy et al. 2018 MSE form
with the alpha-tilde KL that removes target evidence), applied per pixel and
averaged over pixels:

  L = sum_k (q_k - p_k)^2 + p_k (1 - p_k) / (S + 1)
      + lambda * KL[Dir(alpha_tilde) || Dir(1)],   alpha_tilde = q + (1 - q) * alpha

IMPORTANT (loss-surface study, 2026-10-14; scripts/edl_loss_surface.py): at a
disagreement target such as (0.5, 0.5, 0) the error+variance term REWARDS large S;
only the KL term lowers S there, and only for lambda ~ 1e-3 to 1e-2 (30-300x the
ICMLA schedule maximum for K=3). Treat lambda0 as a primary ablation.

The returned dict separates the lambda-free part ("fit") so early stopping can
select on it: with an annealed lambda the total validation loss rises
mechanically over epochs.
"""

import torch
import torch.nn.functional as F


def kl_dirichlet_uniform(alpha):
    S = alpha.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    return (torch.lgamma(S.squeeze(1))
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.lgamma(alpha).sum(dim=1)
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1))


def edl_loss(q, alpha, lam):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    fit = ((q - p) ** 2 + p * (1 - p) / (S + 1)).sum(dim=1)          # [B, H, W]
    alpha_t = q + (1 - q) * alpha
    kl = kl_dirichlet_uniform(alpha_t)                                # [B, H, W]
    total = fit + lam * kl
    return {"loss": total.mean(), "fit": fit.mean().detach(), "kl": kl.mean().detach()}


def kl_schedule(epoch, total_epochs, lambda0, K, scale_by_K=True):
    """ICMLA eq. 7: lambda = lambda0 / K * epoch / N  (epoch 0-based here)."""
    base = lambda0 / K if scale_by_K else lambda0
    return base * (epoch / max(total_epochs, 1))


def soft_ce(q, logits):
    """Cross-entropy with soft targets (Model B); equals hard CE for one-hot q (Model A)."""
    return {"loss": -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()}
