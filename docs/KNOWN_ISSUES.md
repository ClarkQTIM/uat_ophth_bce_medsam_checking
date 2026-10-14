# Known issues and open questions

Each item: the question, why it matters, and how it will be resolved.
Update the status line when something changes; never delete an item, mark it
RESOLVED with the date and evidence.

### K1. Which weight formula did the grid use? — OPEN
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

### K4. Are MedSAM's three multimask tokens meaningful? — OPEN, check first
As far as we know, MedSAM was fine-tuned with single-mask output, so its three
multimask tokens may be essentially SAM's weights. The variance arm on MedSAM
may then behave very differently (near-zero or degenerate variance).
**Resolve by:** `scripts/inspect_backbone.py --backbone medsam` before the MedSAM
runs (REPRODUCTION step 2). If degenerate, the variance arm on MedSAM is still
run and reported, with this explanation.

### K5. GA Conf-0.05 in the Table — LIKELY ERROR
ERRATA E6. **Resolve by:** paper-era GA conformal_lac_005 run folders (do they
end in a crash at the first calibration?).

### K6. Are the split CSVs the paper's? — MUST CHECK FIRST
**Resolve by:** `scripts/check_datasets.py` (counts must equal the paper's;
records SHA-256 of each CSV). Without the original splits this is not the same
experiment.

### K7. Paper-era artifacts — PARTIALLY FOUND (2026-10-14)
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

### K9. Definition of "validation uncertainty" in Figure 7 — ASSUMED
The code that logged `val_uncertainty` is lost. We log the mean of the
uncertainty map over validation images, for epochs after warmup only (Figure 7's
curves start after warmup). Check against the paper-era CSVs if found (K7).

### K10. MedSAM preprocessing differs from SAM's? — CHECK
The paper-era code used SAM's processor. We use each backbone's own processor.
`inspect_backbone.py` records both; document the difference in the erratum if
there is one.
