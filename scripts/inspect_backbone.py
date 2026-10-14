#!/usr/bin/env python
"""
Pre-flight check for a backbone, BEFORE spending GPU hours (docs/REPRODUCTION.md, step 2).

  python scripts/inspect_backbone.py --backbone medsam --csv <split.csv> --n 8
  python scripts/inspect_backbone.py --backbone sam-vit-huge --csv <split.csv> --n 8

Reports:
  * repo id, parameter count, family (and fails loudly if they do not match)
  * the processor's preprocessing settings (resize, normalization)
  * on n training images, with the pretrained (not fine-tuned) weights:
      - mean foreground probability of the single-mask output and of the three
        multimask outputs
      - mean per-pixel variance across the three multimask outputs
        (= the paper's 'variance' uncertainty at initialization)
      - mean absolute difference between multimask tokens
    If MedSAM's multimask tokens were never trained on medical data they may be
    degenerate (near-identical or near-constant); this shows it up front
    (docs/KNOWN_ISSUES.md, K4).
Writes the report to inspect_<backbone>.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from uatcheck.backbones import load_backbone  # noqa: E402
from uatcheck.data import SegmentationDataset  # noqa: E402
from uatcheck.forward import prepare_inputs  # noqa: E402
from uatcheck.metrics import soft_dice_per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mask_max_value", type=float, default=255)
    ap.add_argument("--processor_id", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, info = load_backbone(args.backbone, processor_id=args.processor_id,
                                           device=device)
    model.eval()
    ds = SegmentationDataset(args.csv, "train", processor, limit=args.n,
                             mask_max_value=args.mask_max_value)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            inputs, seg = ds[i]
            batch = {k: v.unsqueeze(0) for k, v in inputs.items()}
            x = prepare_inputs(batch, device)
            seg = seg.unsqueeze(0).to(device)
            single = model(**x, multimask_output=False).pred_masks[:, :, 0].sigmoid()
            multi = model(**x, multimask_output=True).pred_masks.sigmoid()  # [1,1,3,H,W]
            m = multi[:, 0]                                                  # [1,3,H,W]
            rows.append({
                "pixel_values_mean": float(x["pixel_values"].mean()),
                "pixel_values_std": float(x["pixel_values"].std()),
                "single_mean_p": float(single.mean()),
                "multi_mean_p": [float(m[:, k].mean()) for k in range(3)],
                "multi_var_mean": float(m.var(dim=1).mean()),
                "multi_absdiff_01": float((m[:, 0] - m[:, 1]).abs().mean()),
                "multi_absdiff_02": float((m[:, 0] - m[:, 2]).abs().mean()),
                "single_soft_dice_init": float(soft_dice_per_image(single, seg)[0, 0]),
                "multi0_soft_dice_init": float(soft_dice_per_image(m[:, 0:1], seg)[0, 0]),
                "mask_fraction": float(seg.mean()),
            })

    def avg(k):
        return sum(r[k] for r in rows) / len(rows)

    summary = {k: avg(k) for k in rows[0] if not isinstance(rows[0][k], list)}
    report = {"backbone": info.as_dict(), "n_images": len(rows),
              "summary": summary, "per_image": rows}
    out = f"inspect_{args.backbone}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"backbone": info.as_dict(), "summary": summary}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
