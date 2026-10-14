"""
data.py

Binary segmentation dataset, reproducing the paper-era pipeline
(uat_old/src/data.py, SegmentationDataset with model_type="medsam"):

  * CSV with columns: image, segmentation, split  (split in train/val/cal/test)
  * image and mask read with MONAI LoadImage (same reader as the paper-era code,
    so any axis convention it applies is applied identically to both)
  * prompt = a box covering the whole image
  * image preprocessed by the SAM/MedSAM processor
  * mask resized with NEAREST to mask_size (paper-era: hard-coded 256x256, which
    is the native low-resolution mask size of the SAM mask decoder)
  * mask = round(mask) / 255

Guardrail added (not in the paper-era code): after normalization the mask must
contain only {0, 1}. If the source masks are stored as 0/1 instead of 0/255,
the paper-era "/255" would silently turn them into 0/0.0039; we stop instead.
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

SPLITS = ("train", "val", "cal", "test")


def _load(path):
    from monai.transforms import LoadImage
    return LoadImage(image_only=True)(path)


class SegmentationDataset(Dataset):
    def __init__(self, csv_file, split, processor, mask_size=(256, 256),
                 limit=None, mask_max_value=255.0):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split}")
        df = pd.read_csv(csv_file)
        missing = {"image", "segmentation", "split"} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing columns {missing}")
        self.split_data = df[df["split"] == split].reset_index(drop=True)
        if limit is not None:
            self.split_data = self.split_data.iloc[:limit].reset_index(drop=True)
        self.processor = processor
        self.mask_size = tuple(mask_size)
        self.mask_max_value = float(mask_max_value)
        self.split = split

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        row = self.split_data.iloc[idx]
        image = _load(row["image"])
        segmentation = _load(row["segmentation"])

        image = Image.fromarray(np.asarray(image).astype("uint8"))
        input_box = [[0.0, 0.0, image.size[0] - 1, image.size[1] - 1]]
        processed = self.processor(images=image, input_boxes=[input_box],
                                   return_tensors="pt")

        seg = torch.as_tensor(np.asarray(segmentation)).clone().detach().float()
        if seg.ndim == 3:  # e.g. an RGB-saved mask: keep first channel
            seg = seg[..., 0] if seg.shape[-1] in (3, 4) else seg[0]
        seg = seg.unsqueeze(0)
        seg = TF.resize(seg, size=self.mask_size,
                        interpolation=TF.InterpolationMode.NEAREST)
        seg = torch.round(seg) / self.mask_max_value

        uniq = torch.unique(seg)
        if not all(float(u) in (0.0, 1.0) for u in uniq):
            raise ValueError(
                f"Mask {row['segmentation']} is not binary after /"
                f"{self.mask_max_value}: values {uniq.tolist()[:6]}. "
                f"If masks are stored as 0/1, set mask_max_value: 1 in the dataset config."
            )
        return processed, seg

    def get_dataframe(self):
        return self.split_data.copy()


def make_loaders(csv_file, processor, batch_size, mask_size=(256, 256),
                 num_workers=4, limit=None, mask_max_value=255.0, seed=None):
    """Paper-era loader settings: train shuffled; val/test/cal not; cal batch 1."""
    from torch.utils.data import DataLoader

    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    def ds(split):
        return SegmentationDataset(csv_file, split, processor, mask_size,
                                   limit=limit, mask_max_value=mask_max_value)

    train = DataLoader(ds("train"), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, generator=gen)
    val = DataLoader(ds("val"), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers)
    cal = DataLoader(ds("cal"), batch_size=1, shuffle=False,
                     num_workers=num_workers)
    test = DataLoader(ds("test"), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
    return train, val, cal, test
