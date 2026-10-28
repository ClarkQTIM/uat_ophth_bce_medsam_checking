"""
multiclass.py

Building blocks for 3-class UAT on PAPILA (DECISIONS D21): background / rim / cup,
with disc = rim + cup. New research, not a reproduction of the paper (which was
binary cup only), so nothing here has a paper-era counterpart to match; where a
binary choice exists it is carried over unchanged (weight formula, warmup,
conformal LAC per-pixel quantile, U not detached).

  data:        MultiClassDataset reads one label map per image (default column
               `combined_segmentation`, STAPLE fused, values 0/127/255) and maps
               the listed values to class indices 0, 1, 2. Any other value stops.
  forward:     the three MedSAM multimask tokens are read as the three class
               logits (multimask_output=True), softmax over them.
  uncertainty: entropy  U = -sum_k p_k log p_k (natural log, max ln 3 ~ 1.099)
               conformal LAC: score 1 - p(true class); per-pixel qhat across
               calibration images (same quantile rule as the binary arm);
               set = {k : p_k >= 1 - qhat}; U = 1 where the set has >= 2 classes.
               There is no variance arm: the three tokens are now the classes.
  loss:        cross-entropy per pixel times W = 1 + gamma * U, mean over pixels.
  metrics:     soft Dice (on probabilities, the paper's metric family) for
               cup (class 2), rim (class 1) and disc (classes 1 + 2); hard Dice
               from the argmax. "Mean Dice" = mean of cup and disc.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .conformal import find_quantile_segmentation
from .data import SPLITS, _load
from .losses import WEIGHT_FORMS, weight_from_uncertainty
from .metrics import soft_dice_per_image

N_CLASSES = 3
BG, RIM, CUP = 0, 1, 2


# ---------------------------------------------------------------- data
def decode_labels(arr, values):
    """Map stored label values (e.g. [0, 127, 255]) to class indices 0..K-1.
    Raises on any value not listed (e.g. anti-aliasing)."""
    out = torch.full(arr.shape, -1, dtype=torch.long)
    for k, v in enumerate(values):
        out[arr == v] = k
    if (out < 0).any():
        bad = torch.unique(arr[out < 0]).tolist()[:6]
        raise ValueError(f"label map has values {bad} not in {values}")
    return out


class MultiClassDataset(Dataset):
    def __init__(self, csv_file, split, processor, mask_size=(256, 256),
                 label_column="combined_segmentation", label_values=(0, 127, 255), limit=None):
        import pandas as pd
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        df = pd.read_csv(csv_file)
        missing = {"image", label_column, "split"} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing columns {missing}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        self.processor = processor
        self.mask_size = tuple(mask_size)
        self.label_column = label_column
        self.values = list(label_values)
        if len(self.values) != N_CLASSES:
            raise ValueError("label_values must list exactly 3 values (bg, rim, cup)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = np.asarray(_load(row["image"])).astype("uint8")
        if arr.ndim == 2:                      # same rule as data.py (KNOWN_ISSUES K15)
            arr = np.stack([arr, arr, arr], axis=-1)
        image = Image.fromarray(arr)
        box = [[0.0, 0.0, image.size[0] - 1, image.size[1] - 1]]
        processed = self.processor(images=image, input_boxes=[box], return_tensors="pt")
        lab = torch.as_tensor(np.asarray(_load(row[self.label_column]))).float()
        if lab.ndim == 3:                      # an RGB-saved label map: first channel
            lab = lab[..., 0] if lab.shape[-1] in (3, 4) else lab[0]
        lab = TF.resize(lab.unsqueeze(0), size=self.mask_size,
                        interpolation=TF.InterpolationMode.NEAREST)[0].round()
        return processed, decode_labels(lab, [float(v) for v in self.values])


def make_mc_loaders(csv_file, processor, batch_size, mask_size, num_workers, limit,
                    label_column, label_values, seed=None):
    from torch.utils.data import DataLoader
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    def ds(s):
        return MultiClassDataset(csv_file, s, processor, mask_size, label_column,
                                 label_values, limit)
    return (DataLoader(ds("train"), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, generator=gen),
            DataLoader(ds("val"), batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(ds("cal"), batch_size=1, shuffle=False, num_workers=num_workers),
            DataLoader(ds("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers))


# ---------------------------------------------------------------- forward
def forward_logits(model, inputs):
    """[B, 3, H, W] logits: the three multimask tokens read as class channels."""
    return model(**inputs, multimask_output=True).pred_masks[:, 0]


# ---------------------------------------------------------------- uncertainty
def entropy_mc(probs, eps=1e-8, normalize=False):
    p = probs.clamp(min=eps)
    h = -(p * p.log()).sum(dim=1)
    return h / math.log(probs.shape[1]) if normalize else h


def calibration_scores_mc(labels, probs):
    """labels [N, H, W] long; probs [N, K, H, W] -> scores [N, H, W] = 1 - p(true)."""
    return 1.0 - probs.gather(1, labels.unsqueeze(1)).squeeze(1)


def conformal_mc(probs, qhat):
    if qhat is None:
        return torch.zeros_like(probs[:, 0])
    q = qhat.to(probs.device).unsqueeze(1)          # [1, 1, H, W]
    sizes = (probs >= 1.0 - q).sum(dim=1)            # [B, H, W]
    return (sizes >= 2).float()


def uncertainty_mc(uq_type, probs, qhat=None, entropy_normalize=False):
    if uq_type in (None, "none"):
        return None
    if uq_type == "entropy":
        return entropy_mc(probs, normalize=entropy_normalize)
    if uq_type == "conformal":
        return conformal_mc(probs, qhat)
    raise ValueError(f"multi-class arms: none | entropy | conformal (got {uq_type})")


@torch.no_grad()
def recalibrate_mc(model, cal_loader, alpha, device, prep, clip_level=False):
    model.eval()
    P, Y = [], []
    for inputs, y in cal_loader:
        P.append(forward_logits(model, prep(inputs)).softmax(1).cpu())
        Y.append(y)
    probs, labels = torch.cat(P), torch.cat(Y)
    qhat, level = find_quantile_segmentation(calibration_scores_mc(labels, probs), alpha,
                                             clip_level=clip_level)
    stats = {"qhat_level": level, "n_cal": int(probs.shape[0]),
             "qhat_mean": float(qhat.mean()), "qhat_min": float(qhat.min()),
             "qhat_max": float(qhat.max())}
    print(f"[conformal-mc] level={level:.4f} n_cal={stats['n_cal']} qhat mean={stats['qhat_mean']:.4f}")
    return qhat, stats


# ---------------------------------------------------------------- loss
class WeightedCELoss(nn.Module):
    def __init__(self, gamma, weight_form="paper"):
        super().__init__()
        if weight_form not in WEIGHT_FORMS:
            raise ValueError(f"weight_form must be one of {WEIGHT_FORMS}")
        self.gamma, self.weight_form = float(gamma), weight_form

    def forward(self, logits, labels, weight_map=None):
        ce = F.cross_entropy(logits, labels, reduction="none")      # [B, H, W]
        if weight_map is None:
            return ce.mean()
        return (ce * weight_from_uncertainty(weight_map, self.gamma, self.weight_form)).mean()


# ---------------------------------------------------------------- metrics
def _onehot(labels):
    return F.one_hot(labels, N_CLASSES).permute(0, 3, 1, 2).float()


def class_dice(probs, labels, hard=False):
    """Per-image Dice dict {cup, rim, disc, mean}; each a [B] tensor.
    Soft: on probabilities (paper metric family). Hard: on the argmax."""
    p = _onehot(probs.argmax(1)) if hard else probs
    y = _onehot(labels)
    sd = lambda a, b: soft_dice_per_image(a, b)[:, 0]
    cup = sd(p[:, CUP:CUP + 1], y[:, CUP:CUP + 1])
    rim = sd(p[:, RIM:RIM + 1], y[:, RIM:RIM + 1])
    disc = sd(p[:, RIM:RIM + 1] + p[:, CUP:CUP + 1], y[:, RIM:RIM + 1] + y[:, CUP:CUP + 1])
    return {"cup": cup, "rim": rim, "disc": disc, "mean": (cup + disc) / 2}
