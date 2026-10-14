"""Forward-pass semantics on a tiny random SAM, and a full train_one_run per arm
with the backbone/data loaders swapped for synthetic ones. Runs on CPU in ~1 min."""

import json
import os

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from conftest import fake_batch, tiny_sam
from uatcheck.forward import forward_probabilities, prepare_inputs


def test_sam_var_forward_is_variance_of_multimask(tiny_model):
    x = prepare_inputs(fake_batch(2), "cpu")
    with torch.no_grad():
        probs, var = forward_probabilities(tiny_model, x, "sam_var")
        multi = tiny_model(**x, multimask_output=True).pred_masks.sigmoid()
    assert probs.shape == (2, 1, 64, 64) and var.shape == (2, 64, 64)
    assert torch.allclose(probs, multi[:, :, 0])                 # prediction = token 0
    assert torch.allclose(var, multi.var(dim=2)[:, 0])           # unbiased variance of 3


def test_other_arms_use_single_mask(tiny_model):
    x = prepare_inputs(fake_batch(1), "cpu")
    with torch.no_grad():
        probs, var = forward_probabilities(tiny_model, x, "entropy")
        single = tiny_model(**x, multimask_output=False).pred_masks[:, :, 0].sigmoid()
    assert var is None and torch.allclose(probs, single)


class _Synth(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = [fake_batch(1) for _ in range(n)]
        self.y = [(torch.rand(1, 64, 64, generator=g) > 0.7).float() for _ in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {k: v[0] for k, v in self.x[i].items()}, self.y[i]

    def get_dataframe(self):
        return pd.DataFrame({"i": range(len(self))})


@pytest.mark.parametrize("arm", [
    {"uq_type": "none"},
    {"uq_type": "entropy", "gamma": 5, "n_warmup": 1},
    {"uq_type": "sam_var", "gamma": 3, "n_warmup": 1},
    {"uq_type": "conformal", "alpha": 0.2, "gamma": 2, "n_warmup": 1},
])
def test_full_run_writes_paper_compatible_outputs(tmp_path, monkeypatch, arm):
    try:
        model = tiny_sam()
    except Exception as e:
        pytest.skip(f"tiny SAM unavailable: {e}")
    import uatcheck.train_uat as tu
    from uatcheck.backbones import BackboneInfo

    info = BackboneInfo("medsam", "tiny", "tiny", "tiny", 1, 1e-6, (0, 1))
    monkeypatch.setattr(tu, "load_backbone", lambda *a, **k: (model, None, info))
    monkeypatch.setattr(tu, "make_loaders", lambda *a, **k: tuple(
        DataLoader(_Synth(n, s), batch_size=1, shuffle=False)
        for n, s in ((4, 0), (3, 1), (5, 2), (3, 3))))

    cfg = {"dataset": "cup", "csv": "unused.csv", "backbone": "medsam",
           "output_root": str(tmp_path), "num_epochs": 3, "patience": 5,
           "device": "cpu", "learning_rate": 1e-3, **arm}
    run_dir = tu.train_one_run(cfg)

    with open(os.path.join(run_dir, "test_metrics.txt")) as f:
        txt = f.read()
    assert "Foreground Dice Score:" in txt                       # read by analysis scripts
    lp = pd.read_csv(os.path.join(run_dir, "loss_parameters.csv"))
    assert "test" in lp["epoch"].astype(str).tolist()
    for col in ("val_loss", "val_uncertainty", "val_uq_error_dice"):
        assert col in lp.columns                                 # read by curves analysis
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    assert man["config"]["weight_form"] == "paper" and man["finished"]
    assert not os.path.exists(os.path.join(run_dir, "best_model.pth"))
    if arm["uq_type"] == "conformal":
        assert os.path.exists(os.path.join(run_dir, "conformal_log.json"))
    if arm["uq_type"] != "none":
        ep = pd.read_csv(os.path.join(run_dir, "epoch_log.csv"))
        assert not bool(ep.loc[ep.epoch == 1, "uq_active"].item())   # warmup, W=1
        assert bool(ep.loc[ep.epoch >= 2, "uq_active"].all())
    # folder naming the analysis scripts expect
    assert "/medsam/weighted_bce_v2/" in run_dir.replace(os.sep, "/")
