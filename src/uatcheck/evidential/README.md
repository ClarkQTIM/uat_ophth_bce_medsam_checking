# Evidential MedSAM (PAPILA, 3-class) — set up, NOT yet run

Separate from the errata track. Starts after the literature review.

**Design (project outline v2):** one 3-class Dirichlet per pixel over
background / neuroretinal rim / cup (mutually exclusive, so a joint Dirichlet, not
independent Betas as in the infectious-keratitis project). Disc = rim ∪ cup.
Two-rater soft targets q; disagreement maps A (outer BG↔rim, inner rim↔cup).

| Model | Target | Output | Loss |
|---|---|---|---|
| A | STAPLE hard labels | softmax | CE |
| B | two-rater q | softmax | soft CE |
| C | two-rater q | Dirichlet α = softplus(z) + offset | EDL (MSE + annealed KL with α̃) |

All three read the three MedSAM multimask tokens as the three class channels, so
they differ only in activation and loss. No uncertainty-aware weighting (UAT)
in this track.

**Before running, settle:**
1. Data: a CSV with `image, r1_disc, r1_cup, r2_disc, r2_cup, split`
   (+ optional `staple_disc, staple_cup`) from PAPILA's two raters.
2. λ0. The loss-surface study (scripts/edl_loss_surface.py) shows the EDL fit
   term rewards *high* evidence at 50/50 disagreement pixels; only the KL term can
   lower it, and only for λ ≈ 1e-3–1e-2. Default λ0 = 1e-4 (ICMLA) is almost
   certainly too small for H2; plan λ0 ∈ {1e-4, 1e-3, 1e-2} as the main ablation.
3. α offset: the IK code uses α = softplus(z) (offset 0); Sensoy et al. use
   evidence + 1. Default 0 (+1e-6 for stability), configurable.
4. Whether MedSAM's multimask tokens are a sensible starting point (KNOWN_ISSUES
   K4); the alternative is a 1×1 conv head on the decoder's upscaled embedding.

**Run (when ready):**
```bash
python scripts/run_evidential.py --config configs/evidential/papila_edl.yaml --set model=C lambda0=1e-3 run=1
```
