"""
Two-rater PAPILA data for the evidential track (project outline v2, sections 1, 3, 6).

CSV columns:
  image, r1_disc, r1_cup, r2_disc, r2_cup, split   (required)
  staple_disc, staple_cup                           (optional; Model A target)

Per rater r, the nested disc/cup masks become one categorical map:
  Y_r = 0 background (disc=0), 1 rim (disc=1, cup=0), 2 cup (cup=1)
A cup pixel outside the disc (nesting violation) is labelled cup and counted.

Returned per item:
  processed : processor output (image + whole-image box), as in the UAT track
  q         : [3, H, W] soft target = mean of the two raters' one-hots
  hard      : [H, W] long, STAPLE-derived labels if available, else argmax(q)
              with ties broken toward the larger class index (cup > rim > bg)
  agree     : [H, W] bool, Y_1 == Y_2
  interface : [H, W] long, 0 agree, 1 outer (BG<->rim), 2 inner (rim<->cup),
              3 BG<->cup (should be rare)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

BG, RIM, CUP = 0, 1, 2


def _load(path):
    from monai.transforms import LoadImage
    return np.asarray(LoadImage(image_only=True)(path))


def _binary(path, size, max_value):
    m = torch.as_tensor(_load(path)).float()
    if m.ndim == 3:
        m = m[..., 0] if m.shape[-1] in (3, 4) else m[0]
    m = TF.resize(m.unsqueeze(0), size=size, interpolation=TF.InterpolationMode.NEAREST)[0]
    return (torch.round(m) / max_value) > 0.5


def categorical(disc, cup):
    y = torch.zeros(disc.shape, dtype=torch.long)
    y[disc] = RIM
    y[cup] = CUP
    violations = int((cup & ~disc).sum())
    return y, violations


def interface_map(y1, y2):
    out = torch.zeros(y1.shape, dtype=torch.long)
    pair = {frozenset((BG, RIM)): 1, frozenset((RIM, CUP)): 2, frozenset((BG, CUP)): 3}
    dis = y1 != y2
    for k, v in pair.items():
        a, b = tuple(k)
        out[dis & (((y1 == a) & (y2 == b)) | ((y1 == b) & (y2 == a)))] = v
    return out


class TwoRaterPapila(Dataset):
    def __init__(self, csv_file, split, processor, mask_size=(256, 256),
                 mask_max_value=255.0, limit=None):
        df = pd.read_csv(csv_file)
        need = {"image", "r1_disc", "r1_cup", "r2_disc", "r2_cup", "split"}
        if need - set(df.columns):
            raise ValueError(f"{csv_file} missing columns {need - set(df.columns)}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if limit:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        self.processor = processor
        self.size = tuple(mask_size)
        self.mmax = float(mask_max_value)
        self.has_staple = {"staple_disc", "staple_cup"} <= set(df.columns)
        self.nesting_violations = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        image = Image.fromarray(_load(r["image"]).astype("uint8"))
        box = [[0.0, 0.0, image.size[0] - 1, image.size[1] - 1]]
        processed = self.processor(images=image, input_boxes=[box], return_tensors="pt")

        y1, v1 = categorical(_binary(r["r1_disc"], self.size, self.mmax),
                             _binary(r["r1_cup"], self.size, self.mmax))
        y2, v2 = categorical(_binary(r["r2_disc"], self.size, self.mmax),
                             _binary(r["r2_cup"], self.size, self.mmax))
        self.nesting_violations += v1 + v2
        q = (F.one_hot(y1, 3) + F.one_hot(y2, 3)).permute(2, 0, 1).float() / 2.0
        if self.has_staple:
            hard, _ = categorical(_binary(r["staple_disc"], self.size, self.mmax),
                                  _binary(r["staple_cup"], self.size, self.mmax))
        else:
            hard = torch.maximum(y1, y2)
        return {"processed": processed, "q": q, "hard": hard,
                "agree": y1 == y2, "interface": interface_map(y1, y2)}
