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
its test Dice. **Test Dice is logged per epoch only so that the result of each
stopping rule can be read off afterwards (instead of storing a checkpoint per epoch);
it is never used to choose anything: every criterion selects its epoch from
validation values alone.** Runs go to `results/errata_v2`; the 42 runs in `results/errata`
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

### D18. Supplementary paired-by-seed comparison (added 2026-10-16, after interim data)
Runs with the same index share a seed and are identical through the end of warmup, so
each UQ arm can be compared with the baseline seed by seed (mean paired difference,
Wilcoxon signed-rank). Reported alongside D17's unpaired tests, never instead of them;
added after seeing the 22-run interim data (docs/records/interim_2026-10-16.md).

### D19. FAZ: new split, seed fixed in advance (2026-10-19, before any FAZ re-run)
The paper's FAZ split file is lost (ERRATA E15). Decided by C.W.C.: proceed with a new
split rather than wait for Alpine.
- Script: the paper-era `create_octa_faz_csv.py`, copied **unchanged** to
  `scripts/create_octa_faz_csv.py` (SHA-256 05bd4c27…93de, identical to
  `uat_code/src/misc/`), default ratios 0.5/0.3/0.1/0.1 → 150/90/30/30 of the 300
  OCTA-500 6 mm pairs (400 × 400), i.e. the paper's sizes.
- **Seed: 2026**, fixed here before the split was generated or any FAZ run made. Never
  changed after seeing results; no second split is drawn.
- OCTA-500 has one image per subject, so the per-image split is also per subject (no
  leakage of the kind in ERRATA E11).
- Cells unchanged from the paper (`faz_core_sam.yaml`, `faz_core_medsam.yaml`: baseline,
  entropy γ4 W2, variance γ4 W3, Conf-0.05 γ1 W1). Conf-0.05 calibrates on 30 images
  (level ⌈31·0.95⌉/30 = 1.0, allowed).
- Analysis as D17. **Consequence:** FAZ comparisons with the paper's figures and the
  paper-era FAZ runs are confounded by the change of test set (the paper's FAZ
  entropy-vs-baseline gap is only ≈ 0.018), so the D12 FAZ parity check is reported but
  not decisive; within-split comparisons (Q2, Q3) are valid.
- If the original split is later found, the FAZ cells are re-run on it and both are
  reported.

### D20. Patient-level split sensitivity run (fixed 2026-10-20, before the splits were generated)
ERRATA E11: the paper's splits are per image (GA: 23 of 73 patients in more than one
split; cup: 140 of 244). Question: do the results, above all the GA entropy gain, hold
when no patient is in more than one split?
- **Splits:** `scripts/make_patient_split.py --seed 1020`, applied to the paper's own split
  CSVs (`data/ga_split.csv`, `data/cup_split.csv`). Same images; patient ID from the file
  name (GA `_Final_<site>_<patient>_…`, 73 patients; PAPILA `RET<id>`, 244); target split
  sizes = the paper's. Written to `data/{ga,cup}_patient_split.csv` (gitignored); counts
  and SHA-256 recorded in docs/records when generated. **Seed 1020 is fixed here and never
  changed; no second split is drawn.**
- **Runs:** the same 8 cells as `medsam_core` / `parity_sam_core` (no re-tuning), 10 runs
  each, seeds 1001–1010. MedSAM first (`configs/patient_split/medsam_patient.yaml`, 80 runs);
  SAM ViT-H (`sam_patient.yaml`, 80 runs) when GPU time allows.
- **Output:** a separate root, `results/patient_split_v1/`, with the same layout as
  `results/errata_v2/`, so each arm has a per-image-split twin.
- **Analysis:** `scripts/select_offline.py --root results/patient_split_v1`, exactly as D17
  (as_run, Q2, Q3; Holm; paired by seed as supplementary). Primary question: **does entropy
  beat the baseline under Q3 (validation-Dice stopping) on the patient-level split**, per
  dataset and backbone. Secondary: the Δ (arm − baseline) under each split type, side by
  side. Absolute Dice is expected to be lower on the patient split (leakage removed) and is
  not itself the question; the test sets differ, so arms are compared within a split only.
