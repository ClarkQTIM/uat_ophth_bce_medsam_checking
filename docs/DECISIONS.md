# Decisions

Each decision: what we chose, why, and what would change it. Defaults in
`src/uatcheck/train_uat.py` (`DEFAULTS`) implement these. "Paper-era" means the
behavior of `uat_old` unless stated otherwise (see the version table in
ERRATA_LOG.md).

**Guiding rule.** Where the paper text and surviving code agree, follow both.
Where they disagree and the grid code is lost, follow the **paper text** and
schedule a test (KNOWN_ISSUES.md). Where the paper is silent, follow
**paper-era code**. Never silently "fix" a behavior that ran in the paper: make
it a switch, default to what ran, and document it.

---

### D1. Code base
New package (`src/uatcheck`) with the refactor's modular structure and
`uat_old`'s semantics, ported function by function and tested
(`tests/`). Neither old version is run directly: `uat_old` lacks the fixed-γ grid,
and the refactor carries the E2 bug and other drift (E3, E4).

### D2. Weight formula: W = 1 + γU (paper)  — `weight_form: paper`
Why: matches the published equation; the paper's own data (γ = 1 in the grid,
γ = 1 arms differing from baseline) favour it (ERRATA E3).
Changes if: the γ = 1 check (K1) reproduces the paper only under `gm1`, or
paper-era artifacts show otherwise. Alternatives `gm1`, `exp` are implemented.

### D3. Entropy not normalized — `entropy_normalize: false`
Why: paper equation and `uat_old` (ERRATA E4). Changes if: K2 says otherwise.

### D4. Variance arm — `variance_prediction_token: multimask0`
Prediction = mask token 0 of the three multimask outputs; U = unbiased variance
(`torch.var` default) across the three sigmoid outputs. Exactly paper-era. The
other arms use the single-mask output (`multimask_output=False`). Consequence:
the variance arm's segmentation comes from a different output token than every
other arm; documented, not changed. `single` is available for sensitivity runs.

### D5. Uncertainty not detached — `detach_uncertainty: false`
Paper-era behavior (ERRATA E5).

### D6. Conformal (LAC)
- Scores on the calibration split: s = 1 − p(true class), per pixel.
- q̂ = per-pixel quantile across calibration images at level ⌈(n+1)(1−α)⌉/n.
- Prediction set {k : p_k ≥ 1 − q̂}; U = 1 where the set has both classes, else 0.
- Before the first calibration U = 0 (no weighting).
- Recalibrate at the end of every epoch with (epoch + 1) mod W = 0.
- Level > 1 raises (paper-era); `clip_conformal_level` exists but is never used
  for reproduction (ERRATA E6).
- Calibration forward pass uses the single-mask output (paper-era).

### D7. Warmup
The loss is unweighted while the 0-based epoch index < W; weighting starts at
epoch W + 1 (1-based). Paper-era.

### D8. Early stopping and logging
Unweighted validation BCE, patience 5, maximum 40 epochs, best checkpoint kept,
test with the best checkpoint. `loss_parameters.csv` keeps the paper-era quirk
of omitting the epoch that triggered early stopping (so the analysis scripts
count epochs the way the paper did; ERRATA E8); `epoch_log.csv` is complete.

### D9. Metrics
Primary = the paper's: soft Dice on probabilities, per image, smooth 1e-5,
averaged over test images ("Foreground Dice Score" in `test_metrics.txt`).
Additional, never replacing it: hard Dice and IoU at 0.5, UQ-error Dice
(paper-era definition), pixel AUROC of U for detecting errors, validation Dice of
the selected checkpoint.

### D10. Data and preprocessing
Same CSVs and splits as the paper; MONAI `LoadImage`; whole-image box prompt;
masks resized NEAREST to 256 × 256, then round and divide by `mask_max_value`
(255). New: refuse masks that are not binary after normalization.
**Processor:** each backbone's own processor (`processor_id` defaults to the
model repo). For SAM ViT-H this is what the paper-era code used. For MedSAM this
is the correct MedSAM preprocessing; `scripts/inspect_backbone.py` records the
settings so any difference from SAM's preprocessing is documented.

### D11. Runs, seeds, scheduling
10 runs per arm, as in the paper. Seed = 1000 + run index (the paper-era code was
unseeded; seeding makes our runs repeatable on the same hardware, not identical
to the paper's). Job files interleave arms within each run index, so baseline and
UAT arms run in the same arrays, environment and period.
GPU non-determinism means distributions, not numbers, are compared (K8).

### D12. Pre-registered criteria (fixed 2026-10-14, before any re-run result)
**Parity (SAM ViT-H, `parity_sam`):** the pipeline is accepted as faithful if,
for each dataset, (a) the baseline and entropy-arm mean test soft Dice are each
within a tolerance of the paper's figure values (docs/PAPER_CONFIGS.md):
0.02 for cup and GA, 0.01 for FAZ (whose entropy-vs-baseline gap is only ≈0.018),
and (b) entropy beats baseline with Mann-Whitney P < 0.05 as in the paper. If parity
fails, stop and diagnose before running MedSAM.
**MedSAM primary (`medsam_primary`):** the paper's main claim is upheld on MedSAM
for a dataset if the entropy arm's mean test soft Dice exceeds the baseline's
with two-sided Mann-Whitney P < 0.05 (n = 10 vs 10). Variance and conformal arms
are reported with the same test plus a Holm correction across arms within each
dataset (the paper did not correct). Effect sizes (Cliff's δ) and bootstrap
confidence intervals are reported for every arm. No re-tuning of γ or W.

### D13. Model selection for anything new
Parity and MedSAM runs use the paper's reported cells, fixed in advance. Any new
grid selects the best cell on validation Dice (`val_dice_at_best`), never on test
(ERRATA E7).

### D14. Backbone guardrail
Every run asserts the parameter count of the requested backbone and writes
repo id, parameter count, processor settings, git commit and full config to
`run_manifest.json`.

### D15. Evidential track
Separate, MedSAM only, unweighted (no UAT), set up but not run until the
literature review is done. See `src/uatcheck/evidential/README.md`.

### D16. Validation criteria: log all, stop on the paper's text, select offline (2026-10-15)
After ERRATA E14. Every run logs, every epoch: unweighted validation BCE, U-weighted
validation BCE (same γ and formula, U computed on validation from epoch 1, as the
paper-era code did), validation soft Dice, and test soft/hard Dice
(`track_all_criteria: true`). The primary early stopping (what `test_metrics.txt`
reports) uses `val_weighting: none` (the paper's text); training continues until all
three criteria have exhausted patience 5 (max 40 epochs). Because stopping does not
change training, `scripts/select_offline.py` reproduces exactly what early stopping
under each criterion would have selected, from one set of runs. A test checks that
the offline simulation of the primary criterion reproduces the saved checkpoint and
its test Dice. Runs go to `results/errata_v2`; the 42 runs in `results/errata`
(previous pipeline, unweighted criterion only) are kept as interim records.

### D17. Pre-registered analysis of the re-runs (DRAFT, fixed 2026-10-15; to be confirmed by C.W.C. before more results are seen)
Disclosure: written after seeing (a) the paper-era natural experiment in E14 and
(b) 42 interim runs of the previous pipeline (unweighted criterion only; cup entropy
n = 4 vs baseline n = 3; GA n = 2 vs 2). No `results/errata_v2` result had been seen.

All criteria use patience 5, test soft Dice, 10 runs per arm, two-sided Mann-Whitney
vs the baseline **under the same criterion**, Holm across UQ arms within a dataset,
Cliff's δ and bootstrap CIs reported.

- **Q1 Parity (SAM ViT-H): is the new pipeline faithful?** Criterion `as_run` (per
  cell, the validation loss the paper-era runs used). D12 tolerances for baseline and
  entropy means; entropy > baseline P < 0.05. Supplementary: each cell vs the paper's
  own 10 runs (Mann-Whitney).
- **Q2 As written:** criterion `unweighted` for every arm (the paper's Methods).
- **Q3 Common, outcome-aligned:** criterion `dice` (validation soft Dice) for every arm.
- **Main claim ("UAT improves Dice") is upheld for a backbone and dataset only if
  entropy > baseline with P < 0.05 under both Q2 and Q3.** If it holds only under the
  weighted criterion, the conclusion is that the reported gains are not demonstrated to
  come from the weighted training loss rather than from the stopping criterion.
- Results under all four criteria are reported for every arm, whatever they show.
- Same analysis for MedSAM (`medsam_primary`), with Q2/Q3 as the primary questions.
