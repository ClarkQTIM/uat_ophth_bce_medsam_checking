"""Function-level checks of the loss and uncertainty definitions
(docs/DECISIONS.md D2, D3; docs/ERRATA_LOG.md E2, E3)."""

import math

import pytest
import torch
import torch.nn.functional as F

from uatcheck.losses import WeightedBCELoss, weight_from_uncertainty
from uatcheck.uncertainty import conformal_map, entropy_map, uncertainty_map


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(2, 1, 8, 8, generator=g) * 0.98 + 0.01
    y = (torch.rand(2, 1, 8, 8, generator=g) > 0.5).float()
    U = torch.rand(2, 8, 8, generator=g)
    return p, y, U


def test_baselines_of_each_formula_equal_plain_bce():
    p, y, U = _pair()
    bce = F.binary_cross_entropy(p, y)
    assert torch.allclose(WeightedBCELoss(0, "paper")(p, y, U), bce)
    assert torch.allclose(WeightedBCELoss(1, "gm1")(p, y, U), bce)
    assert torch.allclose(WeightedBCELoss(1, "exp")(p, y, U), bce)


def test_gamma1_differs_only_under_paper_formula():
    """The heart of KNOWN_ISSUES K1: gamma=1 is weighted only under the paper's formula."""
    p, y, U = _pair()
    bce = F.binary_cross_entropy(p, y)
    assert not torch.allclose(WeightedBCELoss(1, "paper")(p, y, U), bce)


def test_weight_values():
    U = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(weight_from_uncertainty(U, 5, "paper"), torch.tensor([1.0, 3.5, 6.0]))
    assert torch.allclose(weight_from_uncertainty(U, 5, "gm1"), torch.tensor([1.0, 3.0, 5.0]))
    assert torch.allclose(weight_from_uncertainty(U, 5, "exp"),
                          torch.tensor([1.0, math.sqrt(5), 5.0]))


def test_weighted_loss_is_manual_formula():
    p, y, U = _pair(1)
    bce = F.binary_cross_entropy(p, y, reduction="none")
    manual = (bce * (1 + 3 * U.unsqueeze(1))).mean()
    assert torch.allclose(WeightedBCELoss(3, "paper")(p, y, U), manual)


def test_unweighted_when_no_map():
    p, y, _ = _pair()
    assert torch.allclose(WeightedBCELoss(5, "paper")(p, y, None), F.binary_cross_entropy(p, y))


def test_entropy_unnormalized_by_default():
    p = torch.full((1, 1, 2, 2), 0.5)
    assert torch.allclose(entropy_map(p), torch.full((1, 2, 2), math.log(2)), atol=1e-6)
    assert torch.allclose(entropy_map(p, normalize=True), torch.ones(1, 2, 2), atol=1e-6)
    assert entropy_map(p).shape == (1, 2, 2)


def test_conformal_map_set_size_two():
    # qhat = 0.6 -> include class k if p_k >= 0.4. p=0.5 -> both; p=0.9 -> only fg; p=0.1 -> only bg
    p = torch.tensor([0.5, 0.9, 0.1, 0.45]).view(1, 1, 2, 2)
    qhat = torch.full((1, 2, 2), 0.6)
    assert conformal_map(p, qhat).tolist() == [[[1.0, 0.0], [0.0, 1.0]]]


def test_conformal_before_calibration_is_zero():
    p = torch.rand(2, 1, 4, 4)
    assert torch.equal(conformal_map(p, None), torch.zeros(2, 4, 4))


def test_sam_var_requires_variance_map():
    """Guard against the refactor bug (ERRATA E2): sam_var silently became 1-p."""
    with pytest.raises(RuntimeError):
        uncertainty_map("sam_var", torch.rand(1, 1, 4, 4), variance_map=None)


def test_sam_var_returns_given_map():
    v = torch.rand(1, 4, 4)
    assert torch.equal(uncertainty_map("sam_var", torch.rand(1, 1, 4, 4), v), v)
