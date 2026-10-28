"""3-class UAT (DECISIONS D21): uncertainty maps, loss, metrics, label decoding, and a
full run per arm on a tiny random SAM with synthetic 3-class labels (CPU)."""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from conftest import fake_batch, tiny_sam
from uatcheck.multiclass import (WeightedCELoss, calibration_scores_mc, class_dice,
                                 conformal_mc, decode_labels, entropy_mc)


def test_entropy_max_is_ln3_and_zero_when_certain():
    u = torch.full((1, 3, 2, 2), 1 / 3)
    assert torch.allclose(entropy_mc(u), torch.full((1, 2, 2), math.log(3)), atol=1e-6)
    c = torch.zeros(1, 3, 2, 2); c[:, 1] = 1
    assert entropy_mc(c).max() < 1e-5
    assert torch.allclose(entropy_mc(u, normalize=True), torch.ones(1, 2, 2), atol=1e-6)


def test_conformal_sets_and_scores():
    p = torch.tensor([[0.5, 0.45, 0.05], [0.9, 0.05, 0.05]]).T.reshape(1, 3, 1, 2)
    q = torch.full((1, 1, 2), 0.6)            # threshold 1 - q = 0.4
    U = conformal_mc(p, q)
    assert U.tolist() == [[[1.0, 0.0]]]        # pixel 0: {0,1} -> 1; pixel 1: {0} -> 0
    assert conformal_mc(p, None).sum() == 0    # before the first calibration: no weighting
    y = torch.tensor([[[1, 0]]])
    s = calibration_scores_mc(y, p)
    assert torch.allclose(s, torch.tensor([[[0.55, 0.10]]]), atol=1e-6)


def test_weighted_ce():
    logits = torch.randn(2, 3, 4, 4)
    y = torch.randint(0, 3, (2, 4, 4))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(WeightedCELoss(5)(logits, y, None), ce)
    assert torch.allclose(WeightedCELoss(0)(logits, y, torch.rand(2, 4, 4)), ce)
    U = torch.ones(2, 4, 4)
    assert torch.allclose(WeightedCELoss(2)(logits, y, U), 3 * ce, atol=1e-6)


def test_dice_perfect_and_disc_is_rim_plus_cup():
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    y[0, 2:6, 2:6] = 1
    y[0, 3:5, 3:5] = 2
    p = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float()
    for hard in (False, True):
        d = class_dice(p, y, hard)
        for k in ("cup", "rim", "disc", "mean"):
            assert abs(float(d[k][0]) - 1.0) < 1e-4
    # predicting everything as rim inside the disc: disc perfect, cup 0
    p2 = p.clone(); p2[:, 1] += p2[:, 2]; p2[:, 2] = 0
    d = class_dice(p2, y)
    assert abs(float(d["disc"][0]) - 1) < 1e-4 and float(d["cup"][0]) < 1e-3


def test_decode_labels():
    a = torch.tensor([[0.0, 127.0], [255.0, 0.0]])
    assert decode_labels(a, [0.0, 127.0, 255.0]).tolist() == [[0, 1], [2, 0]]
    with pytest.raises(ValueError):
        decode_labels(torch.tensor([[0.0, 128.0]]), [0.0, 127.0, 255.0])


def test_dataset_reads_combined_map(tmp_path):
    pytest.importorskip("monai")
    from PIL import Image
    from uatcheck.multiclass import MultiClassDataset
    img = (np.random.rand(40, 40, 3) * 255).astype("uint8")
    lab = np.zeros((40, 40), "uint8"); lab[10:30, 10:30] = 127; lab[15:25, 15:25] = 255
    Image.fromarray(img).save(tmp_path / "i.png"); Image.fromarray(lab).save(tmp_path / "l.png")
    csv = tmp_path / "s.csv"
    pd.DataFrame({"image": [str(tmp_path / "i.png")], "combined_segmentation": [str(tmp_path / "l.png")],
                  "split": ["train"]}).to_csv(csv, index=False)
    proc = lambda **k: {"pixel_values": torch.zeros(1, 3, 8, 8), "input_boxes": torch.zeros(1, 1, 4)}
    ds = MultiClassDataset(str(csv), "train", proc, mask_size=(20, 20))
    _, y = ds[0]
    assert y.shape == (20, 20) and set(torch.unique(y).tolist()) == {0, 1, 2}
    assert int((y == 2).sum()) == 25 and int((y >= 1).sum()) == 100   # 10x10 cup, 20x20 disc at half size


class _Synth3(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = [fake_batch(1) for _ in range(n)]
        self.y = [torch.randint(0, 3, (64, 64), generator=g) for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {k: v[0] for k, v in self.x[i].items()}, self.y[i]


@pytest.mark.parametrize("arm", [
    {"uq_type": "none"},
    {"uq_type": "entropy", "gamma": 3, "n_warmup": 1},
    {"uq_type": "conformal", "alpha": 0.2, "gamma": 2, "n_warmup": 1},
])
def test_full_multiclass_run(tmp_path, monkeypatch, arm):
    try:
        model = tiny_sam()
    except Exception as e:
        pytest.skip(f"tiny SAM unavailable: {e}")
    import uatcheck.train_multiclass as tm
    from uatcheck.backbones import BackboneInfo
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "select_offline", os.path.join(os.path.dirname(__file__), "..", "scripts", "select_offline.py"))
    so = importlib.util.module_from_spec(spec); spec.loader.exec_module(so)

    info = BackboneInfo("medsam", "tiny", "tiny", "tiny", 1, 1e-6, (0, 1))
    monkeypatch.setattr(tm, "load_backbone", lambda *a, **k: (model, None, info))
    monkeypatch.setattr(tm, "make_mc_loaders", lambda *a, **k: tuple(
        DataLoader(_Synth3(n, s), batch_size=1, shuffle=False)
        for n, s in ((4, 0), (3, 1), (5, 2), (3, 3))))
    cfg = {"dataset": "papila3", "csv": "unused.csv", "output_root": str(tmp_path),
           "num_epochs": 5, "patience": 2, "device": "cpu", "learning_rate": 1e-3, **arm}
    run_dir = tm.train_one_run(cfg)

    txt = open(os.path.join(run_dir, "test_metrics.txt")).read()
    assert "Foreground Dice Score:" in txt and "Cup Dice Score:" in txt
    ep = pd.read_csv(os.path.join(run_dir, "epoch_log.csv"))
    for col in ("val_loss_unweighted", "val_loss_weighted", "val_soft_dice", "test_soft_dice",
                "test_cup_soft_dice", "test_disc_soft_dice", "test_hard_dice"):
        assert col in ep and ep[col].notna().all()
    if arm["uq_type"] == "none":
        assert (ep.val_loss_unweighted - ep.val_loss_weighted).abs().max() < 1e-9
    if arm["uq_type"] == "entropy":
        assert ep.val_loss_weighted.iloc[0] > ep.val_loss_unweighted.iloc[0]
        assert not bool(ep.uq_active.iloc[0]) and bool(ep.uq_active.iloc[1:].all())
    # primary = validation Dice; offline simulation reproduces the saved checkpoint and its test Dice
    best_ep, _ = so.simulate(ep, "val_soft_dice", -1, 2)
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    assert best_ep == man["best_epoch"]
    fg = float(txt.split("Foreground Dice Score:")[1].split()[0])
    assert abs(fg - float(ep[ep.epoch == best_ep].test_soft_dice.iloc[0])) < 1e-5
    assert "/papila3/medsam/weighted_ce/" in run_dir.replace(os.sep, "/")
    with pytest.raises(ValueError):
        tm.train_one_run({**cfg, "uq_type": "sam_var", "gamma": 1, "n_warmup": 1})
