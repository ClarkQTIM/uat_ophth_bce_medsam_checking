#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
import re
import random
from collections import Counter
from PIL import Image

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
EXT_ORDER = {ext: i for i, ext in enumerate(IMG_EXTS)}

def natural_key(s: str):
    tokens = re.findall(r"\d+|\D+", s)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in tokens]


def collect_files_by_stem(root: Path):
    """Return dict: stem -> list[Path] (so we can pick one if multiples exist)."""
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXT_ORDER:
            out.setdefault(p.stem, []).append(p)  # Remove .resolve() to keep original paths
    return out

def choose_one_path(paths):
    """Deterministically pick one file among duplicates (prefer ext order, then natural filename)."""
    def key(p: Path):
        return (EXT_ORDER.get(p.suffix.lower(), 999), natural_key(p.name))
    return sorted(paths, key=key)[0]

def size_of(p: Path):
    """Return (W,H) or None if unreadable."""
    try:
        with Image.open(p) as im:
            return im.size  # (W,H)
    except Exception:
        return None

def assign_splits(data, train_ratio, test_ratio, val_ratio, cal_ratio, seed=None):
    """Randomly assign train/test/val/cal splits to the data."""
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the data randomly
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    # Cal gets the remainder to account for rounding
    cal_count = total - train_count - test_count - val_count
    
    print(f"[DEBUG] Total samples: {total}")
    print(f"[DEBUG] Split counts - train: {train_count}, test: {test_count}, val: {val_count}, cal: {cal_count}")
    
    # Assign splits
    result = []
    for i, (stem, ip, sp) in enumerate(shuffled_data):
        if i < train_count:
            split = 'train'
        elif i < train_count + test_count:
            split = 'test'
        elif i < train_count + test_count + val_count:
            split = 'val'
        else:
            split = 'cal'
        result.append((stem, ip, sp, split))
    
    return result

def main():
    ap = argparse.ArgumentParser(description="Build CSV of image/seg pairs: intersection of names, both 400x400, with train/test/val/cal splits.")
    ap.add_argument("--images_dir", required=True, type=Path, help="Directory with input images")
    ap.add_argument("--masks_dir",  required=True, type=Path, help="Directory with segmentation masks (ground truth)")
    ap.add_argument("--out_csv",    required=True, type=Path, help="Output CSV path")
    ap.add_argument("--target_w", type=int, default=400, help="Target width (default: 400)")
    ap.add_argument("--target_h", type=int, default=400, help="Target height (default: 400)")
    ap.add_argument("--train_ratio", type=float, default=0.5, help="Training set ratio (default: 0.5)")
    ap.add_argument("--test_ratio", type=float, default=0.3, help="Test set ratio (default: 0.3)")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    ap.add_argument("--cal_ratio", type=float, default=0.1, help="Calibration set ratio (default: 0.1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible splits")
    args = ap.parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio + args.cal_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    assert args.images_dir.is_dir(), f"Images dir not found: {args.images_dir}"
    assert args.masks_dir.is_dir(),  f"Masks dir not found: {args.masks_dir}"

    imgs_by_stem = collect_files_by_stem(args.images_dir)
    segs_by_stem = collect_files_by_stem(args.masks_dir)

    # Resolve to one file per stem (in case of duplicates)
    imgs = {stem: (choose_one_path(paths) if len(paths) > 1 else paths[0])
            for stem, paths in imgs_by_stem.items()}
    segs = {stem: (choose_one_path(paths) if len(paths) > 1 else paths[0])
            for stem, paths in segs_by_stem.items()}

    img_ids = set(imgs.keys())
    seg_ids = set(segs.keys())
    both = sorted(img_ids & seg_ids, key=natural_key)

    print(f"[INFO] images: {len(img_ids)} unique names, masks: {len(seg_ids)} unique names")
    print(f"[INFO] intersection (name-matched pairs): {len(both)}")

    TARGET = (args.target_w, args.target_h)

    kept, dropped_size, dropped_unreadable = [], [], []
    img_shape_hist, seg_shape_hist = Counter(), Counter()

    for stem in both:
        ip, sp = imgs[stem], segs[stem]
        isz = size_of(ip)
        ssz = size_of(sp)
        if isz is None or ssz is None:
            dropped_unreadable.append(stem)
            continue
        img_shape_hist[isz] += 1
        seg_shape_hist[ssz] += 1
        if isz == TARGET and ssz == TARGET:
            kept.append((stem, ip, sp))
        else:
            dropped_size.append((stem, isz, ssz))

    # Shape summary
    def print_hist(h, label):
        total = sum(h.values())
        print(f"[SHAPES] {label}: {total} readable")
        for (w,h), c in h.most_common():
            print(f"  {w}x{h}: {c}")

    print_hist(img_shape_hist, "images")
    print_hist(seg_shape_hist, "masks")

    if dropped_unreadable:
        print(f"[WARN] Unreadable files for {len(dropped_unreadable)} pair(s) (showing up to 10): {dropped_unreadable[:10]}")
    if dropped_size:
        print(f"[WARN] Dropped {len(dropped_size)} pair(s) for size mismatch != {TARGET} (showing up to 10):")
        for stem, isz, ssz in dropped_size[:10]:
            print(f"  {stem}: image={isz}, mask={ssz}")

    # Assign train/test/val/cal splits
    data_with_splits = assign_splits(kept, args.train_ratio, args.test_ratio, args.val_ratio, args.cal_ratio, args.seed)
    
    # Count splits for summary
    split_counts = Counter(split for _, _, _, split in data_with_splits)

    # Write CSV with matched-and-400x400 pairs plus split column
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "segmentation", "split"])
        for stem, ip, sp, split in data_with_splits:
            w.writerow([stem, str(ip), str(sp), split])

    print(f"[OK] Wrote {len(kept)} pairs (both 400x400) to {args.out_csv}")
    print(f"[SPLITS] train: {split_counts['train']}, test: {split_counts['test']}, val: {split_counts['val']}, cal: {split_counts['cal']}")
    if len(kept) > 0:
        print(f"[SPLITS] Actual ratios - train: {split_counts['train']/len(kept):.3f}, test: {split_counts['test']/len(kept):.3f}, val: {split_counts['val']/len(kept):.3f}, cal: {split_counts['cal']/len(kept):.3f}")

if __name__ == "__main__":
    main()