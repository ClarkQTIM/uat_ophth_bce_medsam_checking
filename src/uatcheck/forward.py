"""
forward.py

Model forward pass, reproducing the paper-era `get_probabilities`
(uat_old/src/train.py) with one deliberate change: only the arguments that
SamModel.forward actually uses are passed (pixel_values, input_boxes,
multimask_output). The paper-era code splatted the whole processor dict,
including original_sizes / reshaped_input_sizes, which SamModel ignores.

Paper-era behavior kept exactly:
  * arm "sam_var":  multimask_output=True; the prediction is mask token 0 of the
    three multimask outputs; the uncertainty map is the (unbiased) variance of
    the three sigmoid outputs.  (docs/ERRATA_LOG.md, E2: the later refactor
    discarded this variance map. The paper-era code used it.)
  * every other arm: multimask_output=False; prediction = the single mask.

Option `variance_prediction_token` lets the sam_var arm predict from the
single-mask token instead (NOT the paper's behavior; for sensitivity checks only).
"""

import torch


def prepare_inputs(batch_inputs, device):
    """DataLoader adds a batch dim on top of the processor's; remove it."""
    pv = batch_inputs["pixel_values"]
    boxes = batch_inputs["input_boxes"]
    if pv.dim() == 5:
        pv = pv.squeeze(1)          # [B, 3, 1024, 1024]
    if boxes.dim() == 4:
        boxes = boxes.squeeze(1)    # [B, nb_boxes, 4]
    return {"pixel_values": pv.to(device), "input_boxes": boxes.to(device)}


def forward_probabilities(model, inputs, uq_type, variance_prediction_token="multimask0"):
    """
    Returns:
        probabilities: [B, 1, H, W] foreground probability
        variance_map:  [B, H, W] for uq_type == "sam_var", else None
    """
    if uq_type == "sam_var":
        out = model(**inputs, multimask_output=True)
        all_masks = out.pred_masks.sigmoid()            # [B, 1, 3, H, W]
        variance_map = all_masks.var(dim=2)[:, 0]       # [B, H, W] (unbiased, as paper-era)
        if variance_prediction_token == "multimask0":
            probabilities = all_masks[:, :, 0]          # [B, 1, H, W]
        elif variance_prediction_token == "single":
            single = model(**inputs, multimask_output=False)
            probabilities = single.pred_masks[:, :, 0].sigmoid()
        else:
            raise ValueError(variance_prediction_token)
        return probabilities, variance_map

    out = model(**inputs, multimask_output=False)
    probabilities = out.pred_masks[:, :, 0].sigmoid()   # [B, 1, H, W]
    return probabilities, None


def forward_multimask_logits(model, inputs):
    """Raw logits of the three multimask tokens: [B, 3, H, W]. Used by the
    evidential track as three per-class evidence channels."""
    out = model(**inputs, multimask_output=True)
    return out.pred_masks[:, 0]                         # [B, 3, H, W]
