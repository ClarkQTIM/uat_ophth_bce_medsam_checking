"""Evidential-track building blocks (no training)."""

import torch

from uatcheck.evidential.data import BG, CUP, RIM, categorical, interface_map
from uatcheck.evidential.losses import edl_loss, kl_dirichlet_uniform, kl_schedule, soft_ce


def test_categorical_nesting():
    disc = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    cup = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)     # last pixel: cup outside disc
    y, v = categorical(disc, cup)
    assert y.tolist() == [[BG, RIM, CUP, CUP]] and v == 1


def test_interface_codes():
    y1 = torch.tensor([BG, RIM, RIM, CUP, BG])
    y2 = torch.tensor([BG, BG, CUP, CUP, CUP])
    assert interface_map(y1, y2).tolist() == [0, 1, 2, 0, 3]


def test_kl_zero_at_uniform_prior():
    a = torch.ones(1, 3, 2, 2)
    assert torch.allclose(kl_dirichlet_uniform(a), torch.zeros(1, 2, 2), atol=1e-5)


def test_edl_fit_rewards_high_evidence_at_disagreement():
    """The loss-surface finding: at q=(.5,.5,0), lambda=0, larger S lowers the loss."""
    q = torch.tensor([0.5, 0.5, 0.0]).view(1, 3, 1, 1)
    lo = edl_loss(q, torch.tensor([5.0, 5.0, 1e-3]).view(1, 3, 1, 1), 0.0)["loss"]
    hi = edl_loss(q, torch.tensor([500.0, 500.0, 1e-3]).view(1, 3, 1, 1), 0.0)["loss"]
    assert hi < lo


def test_kl_schedule():
    assert kl_schedule(0, 40, 1e-4, 3) == 0.0
    assert abs(kl_schedule(20, 40, 1e-4, 3) - 1e-4 / 3 * 0.5) < 1e-12


def test_soft_ce_equals_ce_for_onehot():
    logits = torch.randn(2, 3, 4, 4)
    y = torch.randint(0, 3, (2, 4, 4))
    q = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float()
    assert torch.allclose(soft_ce(q, logits)["loss"], torch.nn.functional.cross_entropy(logits, y))
