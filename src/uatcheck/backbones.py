"""
backbones.py

Single place where a segmentation backbone is created. Every run goes through
`load_backbone`, which:

  1. resolves a short backbone name ("sam-vit-huge", "medsam") to a Hugging Face
     repo id (overridable),
  2. loads the model AND the processor from that repo (unless a processor is
     explicitly overridden),
  3. counts parameters and REFUSES to continue if the count does not match the
     architecture the name promises (guardrail: see docs/ERRATA_LOG.md, E1),
  4. returns a metadata dict that is written into every run folder.

Why this exists: in every surviving version of the paper-era code the "medsam"
model type loaded `facebook/sam-vit-huge`, and nothing recorded what was loaded.
"""

from dataclasses import dataclass, field

import torch

# Short name -> (default HF repo id, expected parameter range in millions, family)
BACKBONES = {
    # Original SAM, ViT-H image encoder. ~641M parameters.
    "sam-vit-huge": ("facebook/sam-vit-huge", (600.0, 680.0), "SAM ViT-H"),
    # MedSAM (Ma et al., Nat Commun 2024), ViT-B image encoder. ~94M parameters.
    # Official Hugging Face port published by the MedSAM authors' lab.
    "medsam": ("wanglab/medsam-vit-base", (85.0, 100.0), "MedSAM ViT-B"),
}


@dataclass
class BackboneInfo:
    name: str
    repo_id: str
    processor_id: str
    family: str
    n_params: int
    n_params_millions: float
    expected_range_millions: tuple
    processor_config: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "backbone_name": self.name,
            "backbone_repo_id": self.repo_id,
            "processor_repo_id": self.processor_id,
            "backbone_family": self.family,
            "n_params": self.n_params,
            "n_params_millions": round(self.n_params_millions, 3),
            "expected_range_millions": list(self.expected_range_millions),
            "processor_config": self.processor_config,
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _processor_summary(processor):
    """Pull the preprocessing settings that matter for reproducibility."""
    ip = getattr(processor, "image_processor", processor)
    keys = [
        "do_resize", "size", "do_rescale", "rescale_factor", "do_normalize",
        "image_mean", "image_std", "do_pad", "pad_size", "do_convert_rgb",
    ]
    out = {}
    for k in keys:
        if hasattr(ip, k):
            v = getattr(ip, k)
            try:
                v = dict(v) if hasattr(v, "keys") else v
            except Exception:
                v = str(v)
            out[k] = v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)
    return out


def load_backbone(name, repo_id=None, processor_id=None, device="cpu",
                  allow_unverified=False):
    """
    Load model + processor and verify the parameter count.

    Args:
        name: key of BACKBONES ("sam-vit-huge" or "medsam").
        repo_id: optional override of the HF repo id (e.g. a local path).
        processor_id: optional override of the processor source. Default is the
            same repo as the model. (Use "facebook/sam-vit-huge" here to mimic the
            paper-era preprocessing with MedSAM weights, if that is ever needed.)
        device: torch device.
        allow_unverified: skip the parameter-count assertion (tests only).

    Returns:
        model, processor, BackboneInfo
    """
    from transformers import SamModel, SamProcessor

    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONES)}")
    default_repo, expected, family = BACKBONES[name]
    repo_id = repo_id or default_repo
    processor_id = processor_id or repo_id

    model = SamModel.from_pretrained(repo_id)
    processor = SamProcessor.from_pretrained(processor_id)

    n = count_parameters(model)
    n_m = n / 1e6
    lo, hi = expected
    if not (lo <= n_m <= hi) and not allow_unverified:
        raise RuntimeError(
            f"Backbone check FAILED: '{name}' should have {lo}-{hi}M parameters "
            f"({family}), but {repo_id} has {n_m:.1f}M. Refusing to train on the "
            f"wrong model. (See docs/ERRATA_LOG.md, E1.)"
        )

    info = BackboneInfo(
        name=name, repo_id=repo_id, processor_id=processor_id, family=family,
        n_params=n, n_params_millions=n_m, expected_range_millions=expected,
        processor_config=_processor_summary(processor),
    )
    print(f"[backbone] {name}: {repo_id} | {n_m:.1f}M params ({family}) | "
          f"processor: {processor_id}")
    return model.to(device), processor, info


def backbone_dir_tag(name):
    """Folder name used in the output tree, e.g. results/cup/<tag>/..."""
    return name.replace("-", "_")
