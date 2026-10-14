import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def tiny_sam():
    """A randomly initialized, very small SamModel (CPU-friendly).
    image 256x256, patch 16 -> 16x16 embeddings -> 64x64 masks."""
    from transformers import (SamConfig, SamMaskDecoderConfig, SamModel,
                              SamPromptEncoderConfig, SamVisionConfig)
    hidden = 32
    vision = SamVisionConfig(hidden_size=hidden, output_channels=hidden, num_hidden_layers=2,
                             num_attention_heads=2, image_size=256, patch_size=16,
                             mlp_dim=64, global_attn_indexes=[1], window_size=4)
    prompt = SamPromptEncoderConfig(hidden_size=hidden, image_size=256, patch_size=16,
                                    mask_input_channels=8)
    decoder = SamMaskDecoderConfig(hidden_size=hidden, num_attention_heads=2, mlp_dim=64,
                                   iou_head_hidden_dim=32)
    cfg = SamConfig(vision_config=vision.to_dict(), prompt_encoder_config=prompt.to_dict(),
                    mask_decoder_config=decoder.to_dict())
    return SamModel(cfg).eval()


@pytest.fixture
def tiny_model():
    pytest.importorskip("transformers")
    try:
        return tiny_sam()
    except Exception as e:  # config fields differ across transformers versions
        pytest.skip(f"could not build tiny SAM with this transformers version: {e}")


def fake_batch(b=1, size=256):
    import torch
    # mimic processor output + DataLoader batch dim: [B, 1, 3, H, W], [B, 1, 1, 4]
    return {"pixel_values": torch.randn(b, 1, 3, size, size),
            "input_boxes": torch.tensor([[[[0.0, 0.0, size - 1.0, size - 1.0]]]]).repeat(b, 1, 1, 1)}
