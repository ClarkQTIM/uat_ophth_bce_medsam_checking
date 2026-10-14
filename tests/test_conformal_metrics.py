"""Conformal calibration and metric definitions (ERRATA E6; DECISIONS D9)."""

import numpy as np
import pytest
import torch

from uatcheck.conformal import (calibration_scores_lac, find_quantile_segmentation,
                                quantile_level)
from uatcheck.metrics import soft_dice_per_image, uq_error_dice


def test_quantile_level_and_ga_alpha005_failure():
    assert quantile_level(10, 0.05) == pytest.approx(1.1)      # GA: cannot calibrate
    assert quantile_level(10, 0.10) == pytest.approx(1.0)      # GA: smallest usable alpha
    assert quantile_level(30, 0.05) == pytest.approx(1.0)      # FAZ: just works
    scores = torch.rand(10, 1, 4, 4)
    with pytest.raises(ValueError):
        find_quantile_segmentation(scores, 0.05)
    q, lvl = find_quantile_segmentation(scores, 0.05, clip_level=True)
    assert lvl == 1.0 and q.shape == (1, 4, 4)


def test_lac_scores():
    probs = torch.tensor([0.9, 0.2]).view(1, 1, 1, 2)    # fg prob
    y = torch.tensor([1.0, 1.0]).view(1, 1, 1, 2)
    s = calibration_scores_lac(y, probs)
    assert torch.allclose(s.flatten(), torch.tensor([0.1, 0.8]))


def test_per_pixel_quantile_matches_numpy():
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(51, 1, 3, 3, generator=g)
    q, lvl = find_quantile_segmentation(scores, 0.2)
    ref = np.quantile(scores.squeeze(1).numpy().reshape(51, -1), lvl, axis=0).reshape(1, 3, 3)
    assert np.allclose(q.numpy(), ref)


def test_soft_dice_is_on_probabilities():
    p = torch.full((1, 1, 2, 2), 0.5)
    y = torch.ones(1, 1, 2, 2)
    # 2*sum(0.5*1) / (sum(1) + sum(0.5)) = 4/6
    assert soft_dice_per_image(p, y).item() == pytest.approx(4 / 6, rel=1e-5)


def test_uq_error_dice_none():
    assert uq_error_dice(torch.rand(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), None) == 0.0
