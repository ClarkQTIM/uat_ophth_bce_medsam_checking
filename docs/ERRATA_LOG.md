# Errata log

**Paper:** Clark CW, Kinder S, Nebbia G, et al. *Uncertainty-Aware Training for
Ophthalmic Segmentation Using MedSAM.* Transl Vis Sci Technol. 2026;15(2):19.
doi:10.1167/tvst.15.2.19

This file is the dated record of every discrepancy found between the paper and
the code that produced it, with the evidence for each. It is written so that a
co-author or editor can follow it without any other context.

**Code versions referred to below**

| Name | Location | What it is |
|---|---|---|
| `uat_old` | `/scratch90/chris_/uat_dice/uat_old/src` | Oldest surviving training code (`train.py`, ~1,500 lines). Learnable γ. |
| grid code | lost (possibly on Alpine) | The code that ran the paper's fixed γ × W grid (`weighted_bce_v2`). |
| refactor | `/scratch90/chris_/uat_dice/src` | Later modular rewrite (made with a coding assistant). Fixed γ. |
| this repo | `paper_crisis` | Re-implementation for verification and re-runs. |

**Status key:** CONFIRMED (shown directly by code or data) · LIKELY (strong
indirect evidence; confirmation pending) · OPEN (unresolved) · NOT AN ERROR
(documented for completeness).

---

## E1. The backbone was SAM ViT-H, not MedSAM — LIKELY (confirmation pending)

*Found 2026-10-14.*

**Paper says:** "For the foundational model to fine-tune, we used the Medical
Segment Anything Model (MedSAM)" (Methods, Model); title "…Using MedSAM";
"Preprocessing consists of applying the accompanying MedSAM preprocessing
function" (Methods, Experiment).

**Code shows:** every surviving version loads the original SAM, ViT-H:

- `uat_old/src/train.py`, `setup_model`: `model = create_sam_model()`;
  `processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")`
- `uat_old/src/model.py`: `def create_sam_model(pretrained_weights="facebook/sam-vit-huge", ...)`
- `uat_old/src/train_before_080825.py`: identical.
- refactor `src/setup.py`: `SamModel.from_pretrained("facebook/sam-vit-huge")`.
- `--model_checkpoint` is accepted by `main.py` but never used by the training
  function; both run scripts set `MODEL_CHECKPOINT="blank"`.
- The only MedSAM reference, `preprocess_medsam_inputs` in `uat_old/src/data.py`
  (loads `flaviagiammarino/medsam-vit-base`), is defined but never called.
- Circumstantial: the paper uses batch size 1 "due to memory constraints",
  which fits a 641M-parameter ViT-H far better than a 94M-parameter ViT-B.

**Still to confirm:** the grid code is lost, so the final confirmation must come
from a paper-era artifact: a checkpoint (≈2.5 GB / ≈641M parameters = SAM ViT-H;
≈0.36 GB / ≈94M = MedSAM), a SLURM log with the "Deleted best model checkpoint
(X GB)" line, or a Hugging Face download line in a log.

**Impact:** title; Methods (model description, reference 18 as the model used,
preprocessing). The paper's comparisons (UAT arm vs. baseline) are within-model:
every arm used the same backbone, splits and settings, so the comparisons
themselves remain internally valid — for SAM ViT-H.

**Action:** erratum; re-run the paper's reported configurations with real MedSAM
weights (`configs/errata/medsam_primary.yaml`), after a parity re-run with SAM
ViT-H (`parity_sam.yaml`) shows the new pipeline is faithful.

**Guardrail added:** `src/uatcheck/backbones.py` asserts the parameter count of
the loaded model matches the requested backbone and records repo id, parameter
count, and processor settings in every run's `run_manifest.json`.

---

## E2. The refactor's "variance" arm does not use the variance — CONFIRMED (refactor only)

*Found 2026-10-14.*

**Paper says:** variance = per-pixel variance of MedSAM's three outputs.

**Refactor shows:** `utils.get_probabilities` computes the variance map and
returns it as `weight_map_sam`; `epoch.py` never uses it. Instead
`uncertainty.get_uncertainty_map("sam_var")` returns
`compute_confidence_map(probabilities, invert=True)` = `1 − max_c p` over a
single channel = `1 − p`, i.e. the background probability.

**`uat_old` shows the paper-era behavior was correct:** `train_epoch`,
`validate_epoch` and `test_segmentation_pipeline` only call `get_uncertainty_map`
when `uncertainty_type != "sam_var"`; for `sam_var` the variance map from the
forward pass is used.

**The published figures agree with real variance, not with `1 − p`:**
`1 − p` would give a mean validation uncertainty of ~0.7–0.9 (most pixels are
confident background) and bright backgrounds in uncertainty maps. Figure 7a shows
the variance arm at ~0.002 decaying to ~0; Figure 8's variance maps have dark
backgrounds with structure around the lesion.

**Impact:** none on the paper (as far as can be determined); any results produced
with the refactor's `sam_var` arm are invalid.

**Note (paper-era oddity, see K3):** `uat_old`'s `evaluate_model_on_test` calls
`get_uncertainty_map` without the `sam_var` guard, and `get_uncertainty_map`
raises for `sam_var`. So `uat_old` as it stands would crash at test time for the
variance arm; the grid code must have differed here.

**Guardrail added:** `uncertainty_map("sam_var", …)` raises if no variance map is
supplied; `tests/test_forward_and_loop.py` checks the variance equals the variance
of the three multimask outputs.

---

## E3. Three different weight formulas in the code history — RESOLVED 2026-10-14: the grid weighted at γ = 1 (paper formula consistent; refactor changed it)

*Found 2026-10-14.*

| Source | Weight | No weighting when |
|---|---|---|
| Paper, Methods (Loss) | W = 1 + γU, γ ≥ 0 | γ = 0 |
| refactor `WeightedBCELoss` ("Weighted BCE Loss V2 (Fixed Gamma)") | W = 1 + (γ − 1)U | γ = 1 |
| `uat_old` `WeightedBCELoss` (learnable γ ∈ [0, 5]) | W = γ^U | γ = 1 |
| commented-out older code at top of `uat_old/src/loss.py` | W = base^U | base = 1 |

The grid code (`weighted_bce_v2`, fixed γ ∈ {1..5}) is lost. The refactor
labels its class "V2" and its analysis script says "Paper used weighted_bce_v2",
but the refactor has already been shown unfaithful once (E2).

**Evidence the grid used the paper's formula:** the grid includes γ = 1; under
either code formula every γ = 1 arm would be an exact duplicate of the baseline,
which would be a strange grid to design. Several reported γ = 1 arms differ
significantly from baseline (Table: FAZ Conf-0.05 (γ = 1, W = 1) converges in
9.4 vs 17.9 epochs, P < 0.001), which is expected if they were weighted
(W = 1 + U) and hard to explain if they were not.

**Decision:** assume the paper's formula (DECISIONS D2). **Test:** K1,
`configs/errata/gamma1_formula_check.yaml`.

**Resolution (2026-10-14, from the paper's own runs; docs/records/paper_era_inventory.md):**
the 35 γ = 1 entropy/conformal cells differ strongly from one another (cup
Kruskal–Wallis P = 1.6 × 10⁻¹⁵; FAZ P = 4.3 × 10⁻³) with a dose–response in the
amount of flagged uncertainty. Under a formula with no weighting at γ = 1 they
would be replicates. So the grid weighted at γ = 1: not the refactor's
1 + (γ−1)U, not `uat_old`'s γ^U; consistent with the paper's W = 1 + γU. The
paper's equation stands. The refactor's formula change means any later
experiments run with the refactor (e.g. the UAT-Dice/Tversky work) used a
different γ scale and are not directly comparable with the paper.

**Impact if the paper formula is wrong:** the equation in Methods needs
correcting and every reported γ maps to a different effective weight; γ = 1 arms
would be baseline replicates, and their reported differences from baseline would
need another explanation (see E7).

---

## E4. Entropy normalization differs between versions — OPEN (assume paper)

*Found 2026-10-14.*

Paper equation and `uat_old`: natural-log binary entropy, not normalized
(maximum ln 2 ≈ 0.693). Refactor: divided by ln 2 (maximum 1). The grid code is
lost. Normalizing is equivalent to multiplying γ by 1/ln 2 ≈ 1.44 for the entropy
arm only. **Decision:** paper equation (D3). Optional test: K2.

---

## E5. Uncertainty maps are not detached — NOT AN ERROR (undocumented property)

*Found 2026-10-14.*

In every version, U is computed from the model's own outputs and multiplied into
the loss without `.detach()`, so gradients flow through U as well as through the
BCE term. For entropy this adds a term that rewards confident (low-entropy)
predictions; for variance it rewards agreement between the three mask tokens.
The paper does not mention this. It is part of the method as run; the erratum
or any follow-up should describe it. Reproduced by default
(`detach_uncertainty: false`, D5).

---

## E6. GA Conf-0.05 cannot be calibrated; the Table's GA Conf-0.05 row is suspect — LIKELY

*Found 2026-10-14.*

The conformal threshold is a per-pixel quantile at level ⌈(n + 1)(1 − α)⌉ / n.
GA has n_cal = 10: for α = 0.05 the level is 11/10 = 1.1 > 1 and `np.quantile`
raises. The first calibration happens after the warmup epochs, so such a run
trains for W epochs and then crashes.

Consistent with this: Figure 5 (GA conformal ablation) has no α = 0.05 arm, and
the Discussion mentions small datasets "that do not allow for all uncertainty
methods". **But the Table lists "GA Conf-0.05: 5.0 epochs (−58.3%), P = 0.150".**
That row most likely comes from runs that crashed at the first calibration, and
should be removed. **To confirm** from the paper-era run folders (K7).

---

## E7. Best configurations were selected on the test set — CONFIRMED (methodological; also confirmed from the paper's runs)

*Found 2026-10-14.*

`analysis/1_run_boxplot_analysis.py` reads the test Dice of every (γ, W) cell,
picks the best per method, and then tests that best cell against the baseline on
the same test data. Choosing the maximum of up to 25 cells and then testing it
on the data used to choose it biases the comparison in favour of UAT. The size of
the bias grows as the true differences between cells shrink.

**Impact:** small for entropy-UAT, whose gains (e.g. OC ≈ 0.78 vs 0.70) are far
larger than cell-to-cell noise; potentially material for the smaller conformal
and variance effects. **Action:** disclose; the re-runs use the paper's reported
cells fixed in advance (no re-selection); any new grid selects on validation
Dice (`val_dice_at_best` is now recorded for every run; D13).

---

## E8. "Avg. Epochs" in the Table undercounts early-stopped runs by one — CONFIRMED (minor; 4,888/4,888 early-stopped paper-era runs)

*Found 2026-10-14.*

Paper-era `train_segmentation_pipeline` breaks out of the epoch loop before
appending the row of the epoch that triggered early stopping; the convergence
analysis counts rows. Early-stopped runs are therefore counted as one epoch
shorter than they ran. This applies to every arm, so differences between arms
are essentially unaffected. Reproduced in `loss_parameters.csv` for
comparability; `epoch_log.csv` has the complete record (D8).

---

## E9. PAPILA image count: 486, not 438 — CONFIRMED (text)

*Found 2026-10-14 (dataset check, docs/records/datasets_2026-10-14.txt).*

Paper: "438 full-color fundus images in a 238/146/51/51 train/test/calibration/
validation split". 238 + 146 + 51 + 51 = 486, and the split CSV used has 486 rows
with exactly those counts. **Action:** correct 438 → 486 in the erratum.

---

## E10. GA split sizes: 40 train / 50 test in the data, "50/40" in the text — CONFIRMED (text)

*Found 2026-10-14.*

Paper: "110 fundus autofluorescence (FAF) images from 73 patients … in a
50/40/10/10 train/test/calibration/validation split". The split CSV used
(`ga_dataset_train_val_test_cal.csv`, identical splits in the Alpine copy) has
**train 40, test 50**, cal 10, val 10. The data are what ran; the text swaps train
and test. **Action:** correct the sentence. The re-runs use the CSV as is.

---

## E11. Splits were made per image, not per patient — CONFIRMED (limitation)

*Found 2026-10-14.*

- **GA:** 110 images from 73 patients (patient ID read from the file name,
  `_Final_<site>_<patient>_…`; the count matches the paper's 73).
  **23 of 73 patients** have images in more than one split (e.g. one patient in
  train, cal and test).
- **Cup (PAPILA):** 486 images from 244 patients (`RET<id>OD/OS`).
  **140 of 244 patients** have their two eyes in different splits.

Images of the same patient (fellow eyes, repeat visits) are correlated, so test
scores are optimistic, and calibration/test images share patients with training,
which weakens the exchangeability conformal prediction relies on. The paper does
not claim patient-level splits, so this is a limitation to disclose, not a
misstatement. Baseline and UAT arms share the same splits, so the comparison
between them is not biased by it in an obvious direction.
**Action:** disclose in the erratum. The re-runs keep the original splits (to
reproduce what ran); a patient-level re-split is a planned sensitivity analysis.
FAZ (OCTA-500) to be checked once its split file is found.

---

## E12. "Final Gamma: 5.000000" in baseline results — NOT AN ERROR (artifact)

*Found 2026-10-14.* Paper-era baseline `test_metrics.txt` files end with
"Final Gamma: 5.000000". The writer prints the configured γ of the loss object;
the baseline never passes an uncertainty map, so no weight is applied. Recorded so
nobody reads it as a weighted baseline.

---

## E13. The Table's convergence numbers cannot be reproduced from the runs behind the figures — CONFIRMED inconsistency; cause unknown

*Found 2026-10-14 (docs/records/paper_era_inventory.md). Wording revised the same
day to claim no more than the evidence supports.*

**What we can show.** The surviving tree is the data behind Figures 1–6 (every
reported cell is the best test-Dice cell; means match to ±0.001). The Table reports
average epochs for those same best cells. For 21 reported UQ cells, all 10 runs of
the cell survive, and we count epochs the way the convergence script does (epoch
rows in `loss_parameters.csv`). Only one Table value is reproduced (GA Conf-0.25,
21.4). E.g. cup entropy: Table 9.8, the cell's 10 runs 15.5; FAZ Conf-0.05:
Table 9.4, the cell's 10 runs 16.5. No alternative measure reproduces the Table
(epochs to 90/95/99 % of the loss drop, best epoch, stop epoch).

**What we cannot show.**
- Whether the Table is correct for the runs it was computed from. Those runs do
  not survive; the Table may be accurate for an earlier run set that was later
  overwritten or deleted.
- The Table's baseline values, % differences and P values: no cup or GA baseline
  runs survive, and only 2 of 10 FAZ baseline runs.

**Conclusion.** The Figures and the Table appear to describe different sets of
runs, and only the figures' runs survive. A reader would reasonably assume they
describe the same experiments, so the inconsistency should be corrected, not the
Table declared wrong. **Action:** report the convergence analysis from the parity
re-runs (provenance recorded for every run) and state why in the erratum. The Dice
results (Figures 1–6) are not affected.

---

## E14. Early stopping used the uncertainty-weighted validation loss, not the unweighted loss the Methods state — CONFIRMED from training logs (code lost); impact tested on SAM ViT-H 2026-10-19 (cup gain not supported under a common criterion; GA gain holds)

*Found 2026-10-15 (docs/records/interim_2026-10-15.md).*

**Paper says:** "Training is guided by early stopping using unweighted loss with a
patience of 5 epochs" (Methods).

**Paper-era logs show** (`loss_parameters.csv` of the 4,890 surviving runs) that the
validation loss driving early stopping and checkpoint selection was weighted by the
uncertainty map, with the same γ and formula as training, from epoch 1:
1. At epoch 1 (warmup: training is unweighted for every γ) the *training* loss is
   the same for every γ (cup 0.2245–0.2302), but the entropy arm's *validation*
   loss rises linearly with γ (cup 0.176, 0.230, 0.280, 0.347 for γ = 1–4; FAZ
   0.016 → 0.030 for γ = 1–5), extrapolating at γ = 0 to the unweighted value of the
   conformal arms (cup ≈ 0.12, FAZ ≈ 0.012). The variance arm shows the same γ trend.
2. Conformal arms (U = 0 before the first calibration) have unweighted-looking
   validation loss at epoch 1, then a jump right after the first calibration that
   grows with γ (cup ×1.10, 1.22, 1.29, 1.45 for γ = 1–4).
3. The γ = 1 cells are already weighted (ratio ≈ 1.2–1.4), an independent
   confirmation of W = 1 + γU (E3).

**Exception:** cup γ = 5, W = 1–4 (all arms) ran with **unweighted** validation
(epoch-1 entropy ratio ≈ 1.0; cup γ = 5, W = 5 is weighted, ratio 3.1): a batch run
with a different code version. Of the reported cells, only cup Conf-0.10 (γ5 W1) is
affected; `configs/errata/paper_arms.yaml` marks it `paper_val: unweighted`.

**Why it matters.** The baseline has no uncertainty map, so its stopping criterion
is plain BCE; the UAT arms stopped on a different criterion. In the paper's own runs
(cup, entropy arm), the cells with weighted validation average 0.733–0.781 test Dice
and 9.7–15.5 epochs, while the cells with unweighted validation (γ5 W1–4, n = 40)
average **0.672** and **6.7 epochs**, below the paper's cup baseline (0.705, figure
reading). Caveat: those cells also differ in γ/W and possibly in other code details.
Our interim re-runs with unweighted validation agree in direction (small n): in 2 of
4 cup entropy runs and 2 of 2 GA entropy runs, early stopping selected a checkpoint
from inside the warmup, i.e. a model trained identically to the baseline of the same
seed (possibly stopped at a different epoch). *(Wording corrected 2026-10-19: the
earlier text said "identical to the baseline", which holds only when both stop at the
same epoch.)*

**Tested 2026-10-19 (SAM ViT-H, cup and GA, n = 10 per arm;
docs/records/core_sam_2026-10-19/RESULTS.md).**
- Under the as-run (weighted) criterion the re-runs match the paper's own runs in all
  six UQ cells (P = 0.31–0.97) and reproduce the reported gains (cup entropy +0.059,
  GA entropy +0.093 over baseline).
- **Cup:** with validation-Dice stopping for every arm, the baseline reaches 0.781 and
  entropy-UAT 0.785 (+0.004, P = 0.31). The reported cup gain is explained by the
  stopping criterion (baseline stopped at epoch ≈ 6 under BCE vs ≈ 19 under Dice), not
  by uncertainty-aware training. Variance +0.010 (P = 0.017, Holm 0.052).
- **GA:** with validation-Dice stopping, entropy-UAT 0.813 vs baseline 0.749 (+0.065,
  P = 0.005, Holm 0.014). A real gain, smaller than reported (+0.097).
- **The Methods' stated rule could not have produced the reported entropy results:**
  with unweighted-BCE stopping, the entropy arm selects a checkpoint from inside warmup
  (never trained with UAT) in 10/10 cup and 4/10 GA runs.
- Pre-registered D17 verdict (entropy > baseline under both stated and common
  criteria): not upheld for either dataset on SAM ViT-H; GA upheld under the common
  criterion alone.
- Still to test: MedSAM (`medsam_core`), FAZ.

**Action:** correct the Methods statement; report results under a common stopping
criterion (D17) alongside the as-run criterion; the cup result must be described as
not supported under a common stopping criterion (pending MedSAM, which is the model
the paper names).
