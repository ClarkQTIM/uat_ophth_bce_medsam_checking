"""Uncertainty-quality tracking (DECISIONS D22): calibration accumulator and per-epoch columns."""

import os

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import tiny_sam
from uatcheck.metrics import CalibrationAccumulator


def test_calibration_perfect_and_calibrated():
    y = (torch.rand(1, 1, 32, 32) > 0.5).float()
    c = CalibrationAccumulator(); c.add(y, y)
    r = c.compute()
    assert r["ece"] < 1e-9 and r["brier"] < 1e-12
    # p = 0.8 everywhere, 80% of pixels foreground -> calibrated: ECE ~ 0, Brier = 0.16
    y2 = torch.zeros(1, 1, 10, 10); y2.view(-1)[:80] = 1
    c2 = CalibrationAccumulator(); c2.add(torch.full_like(y2, 0.8), y2)
    r2 = c2.compute()
    assert r2["ece"] < 1e-6 and abs(r2["brier"] - 0.16) < 1e-6


@pytest.mark.parametrize("arm", [{"uq_type": "none"}, {"uq_type": "entropy", "gamma": 5, "n_warmup": 1},
                                 {"uq_type": "conformal", "alpha": 0.2, "gamma": 2, "n_warmup": 1}])
def test_uq_quality_columns(tmp_path, monkeypatch, arm):
    try:
        model = tiny_sam()
    except Exception as e:
        pytest.skip(f"tiny SAM unavailable: {e}")
    import uatcheck.train_uat as tu
    from uatcheck.backbones import BackboneInfo
    from test_forward_and_loop import _Synth
    info = BackboneInfo("medsam", "tiny", "tiny", "tiny", 1, 1e-6, (0, 1))
    monkeypatch.setattr(tu, "load_backbone", lambda *a, **k: (model, None, info))
    monkeypatch.setattr(tu, "make_loaders", lambda *a, **k: tuple(
        DataLoader(_Synth(n, s), batch_size=1, shuffle=False)
        for n, s in ((4, 0), (3, 1), (5, 2), (3, 3))))
    cfg = {"dataset": "cup", "csv": "unused.csv", "backbone": "medsam", "output_root": str(tmp_path),
           "num_epochs": 3, "patience": 2, "device": "cpu", "learning_rate": 1e-3,
           "track_uq_quality": True, "save_test_tensors": False, **arm}
    run_dir = tu.train_one_run(cfg)
    ep = pd.read_csv(os.path.join(run_dir, "epoch_log.csv"))
    for col in ("test_err_auroc_entropy", "test_ece", "test_brier", "test_nll"):
        assert col in ep and ep[col].notna().all()
    assert ep.test_err_auroc_entropy.between(0, 1).all() and ep.test_ece.between(0, 1).all()
    assert "test_err_auroc_own_u" in ep
