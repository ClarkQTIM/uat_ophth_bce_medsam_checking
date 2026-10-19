# Known issues and open questions

Each item: the question, why it matters, and how it will be resolved.
Update the status line when something changes; never delete an item, mark it
RESOLVED with the date and evidence.

### K1. Which weight formula did the grid use? — RESOLVED 2026-10-14
γ = 1 was weighted (paper-era γ = 1 cells differ strongly, with a dose–response;
docs/records/paper_era_inventory.md). The paper's W = 1 + γU stands; ERRATA E3.
`gamma1_formula_check.yaml` is no longer needed to decide this; if run, it simply
shows unweighted training on the new pipeline (optional).

Original note:
ERRATA E3. **Resolve by:** `configs/errata/gamma1_formula_check.yaml` (20 runs):
re-run the reported γ = 1 arms (FAZ Conf-0.05 γ1 W1; OC Conf-0.15 γ1 W1) with
the no-weighting formula and compare with the same arms under the paper formula
(in `parity_sam`) and with the paper (FAZ Conf-0.05: 9.4 epochs vs 17.9 baseline,
P < 0.001). Whichever reproduces the paper is the one that ran. Also settled
directly by any surviving grid code or `args.json` (K7).

### K2. Was entropy normalized by ln 2 in the grid? — OPEN, low priority
ERRATA E4. Changes the effective γ of the entropy arm by ×1.44. **Resolve by:**
optional 20-run check (entropy arm, cup and octa, `entropy_normalize: true`)
compared with parity, or by grid code (K7).

### K3. How did the paper get test metrics for the variance arm? — OPEN
`uat_old`'s `evaluate_model_on_test` would raise for `sam_var` (ERRATA E2 note).
The grid code must have differed. Our evaluation uses the variance map at test
time. **Resolve by:** grid code (K7). Low impact on the main claims.

### K4. Are MedSAM's three multimask tokens meaningful? — RESOLVED 2026-10-14: yes
`docs/records/inspect_medsam.json` (8 cup training images, pretrained weights):
mean variance across the three tokens 0.055 (SAM ViT-H: 0.082), mean absolute
difference between tokens 0.15 (0↔1) and 0.33 (0↔2). The outputs are distinct, so
the variance arm is meaningful on MedSAM.

Original note:
As far as we know, MedSAM was fine-tuned with single-mask output, so its three
multimask tokens may be essentially SAM's weights. The variance arm on MedSAM
may then behave very differently (near-zero or degenerate variance).
**Resolve by:** `scripts/inspect_backbone.py --backbone medsam` before the MedSAM
runs (REPRODUCTION step 2). If degenerate, the variance arm on MedSAM is still
run and reported, with this explanation.

### K5. GA Conf-0.05 in the Table — LIKELY ERROR; not confirmable from surviving files (no GA α=0.05 folder survives)
ERRATA E6. **Resolve by:** paper-era GA conformal_lac_005 run folders (do they
end in a crash at the first calibration?).

### K6. Are the split CSVs the paper's? — RESOLVED for cup and GA (2026-10-14); FAZ: original lost, new seeded split (2026-10-19, ERRATA E15, DECISIONS D19)
Cup and GA: path-corrected copies of the paper-era CSVs; every surviving copy has
identical split assignments; counts as in the data (see ERRATA E9, E10); all files
present; masks binary. SHA-256 in `docs/records/datasets_2026-10-14.txt`. FAZ split
file not yet found.

Original note:
**Resolve by:** `scripts/check_datasets.py` (counts must equal the paper's;
records SHA-256 of each CSV). Without the original splits this is not the same
experiment.

### K7. Paper-era artifacts — INVENTORIED 2026-10-14
4,890 runs: cup full grid (no baseline), GA γ = 3–5 only (no baseline, no
α = 0.05), FAZ full grid (2 of 10 baseline runs). These are the runs behind
Figures 1–6. Details: docs/records/paper_era_inventory.md.

Original note:
Found: `/persist/QTIM/Active/22-2198/uat_ophth_bce_medsam/training_essentials_paper/`
(layout `<dataset>/medsam/weighted_bce_v2/gamma_X/recalc_Y/<arm>/run_N/`,
e.g. `.../cup/medsam/weighted_bce_v2/gamma_1/recalc_1/conformal_lac_005/run_1/test_metrics.txt`).
**Incomplete:** many results were deleted from Alpine scratch before they could
be preserved. Other places checked/to check: `/scratch90/chris_/uat_bce/`
(path hard-coded in the refactor's curves analysis), Alpine
`/scratch/alpine/cclark@xsede.org/uat/`.
**Do not delete or modify anything there; snapshot first.** Whatever survives is
the reproduction target and may settle E1, K1, K2, K3, K5 directly.
**Next:** inventory exactly which cells/runs survive and record it in
`docs/records/paper_artifacts_inventory.md`.

### K8. Exact numbers will not reproduce
GPU non-determinism and the unseeded paper-era runs mean individual runs will
differ. Comparisons are of distributions (D12).

### K9. Definition of "validation uncertainty" in Figure 7 — RESOLVED 2026-10-14
The grid code logged `val_uncertainty` as the mean validation uncertainty map from
epoch 1 (entropy/variance non-zero during warmup; conformal 0 until the first
calibration). Ours logs the same quantity but only after warmup (NaN before).
(Corrected 2026-10-15: an earlier note said 0.0 during warmup, which holds only for
conformal arms.)

Original note:
The code that logged `val_uncertainty` is lost. We log the mean of the
uncertainty map over validation images, for epochs after warmup only (Figure 7's
curves start after warmup). Check against the paper-era CSVs if found (K7).

### K10. MedSAM preprocessing differs from SAM's? — RESOLVED 2026-10-14: no
`wanglab/medsam-vit-base`'s processor is identical to `facebook/sam-vit-huge`'s
(longest edge 1024, pad to 1024, rescale 1/255, ImageNet mean/std, RGB).
So the paper-era use of SAM's processor did not change the preprocessing. Note:
the original MedSAM code base normalizes intensities min-max to [0, 1]; we use the
Hugging Face port and its processor, as stated in DECISIONS D10.

Original note:
The paper-era code used SAM's processor. We use each backbone's own processor.
`inspect_backbone.py` records both; document the difference in the erratum if
there is one.

### K11. Does UAT improve Dice under a common stopping criterion? — ANSWERED for SAM ViT-H cup and GA (2026-10-19); OPEN for MedSAM and FAZ
SAM ViT-H, n = 10 per arm, validation-Dice stopping for every arm (D17 Q3):
cup entropy +0.004 (P = 0.31) — **no gain**; GA entropy +0.065 (P = 0.005, Holm 0.014)
— **gain**. Details: docs/records/core_sam_2026-10-19/RESULTS.md; ERRATA E14.
Next: `configs/errata/medsam_core.yaml` (same 80 cells on MedSAM); FAZ when its split
file is found.

Original note:
ERRATA E14. **Resolve by:** `results/errata_v2` runs with per-epoch tracking, analysed
with `scripts/select_offline.py` under DECISIONS D17.

### K12. Why did cup γ = 5, W = 1–4 (6 of 8 arms) run with unweighted validation? — OPEN, low priority
A separate batch with a different code version (E14). Only matters for explaining the
paper-era data; cup Conf-0.10 (γ5 W1) is the one reported cell affected.

### K13. "Traceback" in every run log — RESOLVED 2026-10-16: harmless
Python multiprocessing fails to delete a DataLoader worker temp folder at cleanup
(`multiprocessing/util.py _remove_temp_dir → shutil.rmtree`), a known nuisance on
network file systems such as `/scratch90`. Runs train and finish normally. To detect a
run that really died, check for a log without `[done]` whose run folder has no
`test_metrics.txt` and whose `epoch_log.csv` has stopped growing.

### K14. Is the cup gain the stopping rule or the weighted training loss? — OPEN; possible ablation (NOT implemented)
*Noted 2026-10-16, after the interim results (docs/records/interim_2026-10-16.md);
any test of it is post hoc and must be reported as such.*
Interim cup data suggest entropy-UAT's advantage under the paper-era (weighted)
stopping rule comes from training longer: a baseline stopped on validation Dice
reaches the same test Dice. **Possible test:** for every arm, including the baseline,
also log an entropy-weighted validation loss computed from the model's own predictions
(γ = 5, the reported entropy cell's γ, W = 1 + γH(p)), and stop on it. If the baseline
stopped this way matches entropy-UAT, the effect is the stopping rule alone; if UAT
still wins, the weighted training loss contributes. Logging only, no extra training;
would go into the MedSAM runs (and any further SAM runs) if adopted.
**Decision deferred** (C.W.C., 2026-10-16): recorded as an option; runs continue over
the weekend unchanged.
