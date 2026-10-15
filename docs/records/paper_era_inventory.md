# Paper-era results: inventory and analysis (2026-10-14)

**Source:** `/persist/QTIM/Active/22-2198/uat_ophth_bce_medsam/training_essentials_paper/`
(text files only: `loss_parameters.csv`, `test_metrics.txt`, `test_df.csv`; copied
as `data/paper_era_runs_text.tgz`, not in git). Per-run table:
`docs/records/paper_era_runs.csv`, made by
`scripts/inventory_paper_era.py <tree> --out docs/records/paper_era_runs.csv`.
Layout: `<dataset>/medsam/weighted_bce_v2/gamma_<γ>/recalc_<W>/<arm>/run_<n>/`;
baselines at `<dataset>/medsam/weighted_bce_v2/none/run_<n>/`.

## What survives

| Dataset | UQ cells (γ×W) per arm | Runs per cell | Baseline runs |
|---|---|---|---|
| cup | 25/25 for all 8 arms (incl. conformal α=0.05) | 10 | **none** |
| GA | 12–13/25 per arm (γ = 3, 4, 5 only; no γ = 1, 2; γ4 partly); no α=0.05 arm at all | 10 (one cell 8) | **none** |
| FAZ (octa) | 25/25 for all 8 arms | 10 | 2 (of 10) |

4,890 runs in total (cup 2,000; GA 888; FAZ 2,002).

## These are the runs behind Figures 1–6

For every reported arm whose cell survives (22 of 23), choosing the (γ, W) cell
with the highest mean test Dice over the surviving grid reproduces exactly the cell
reported in the paper, and the cell means agree with the values read off the
figures to ±0.001 (e.g. cup entropy γ5 W5: 0.7813; FAZ entropy γ4 W2: 0.9067).
The only missing reported cell is GA Conf-0.15 (γ2, W1), because no GA γ = 2 runs
survive. This also confirms that cells were selected on test Dice (ERRATA E7).

Exact values for the reported cells (mean ± SD of test soft Dice, n = 10;
epoch rows in `loss_parameters.csv`):

| Dataset | Arm | γ | W | Mean | SD | Epoch rows |
|---|---|---|---|---|---|---|
| cup | entropy | 5 | 5 | 0.7813 | 0.0098 | 15.5 |
| cup | variance | 3 | 5 | 0.7421 | 0.0158 | 10.5 |
| cup | conf 0.05 | 1 | 5 | 0.7155 | 0.0145 | 8.5 |
| cup | conf 0.10 | 5 | 1 | 0.7233 | 0.0189 | 9.8 |
| cup | conf 0.15 | 1 | 1 | 0.7160 | 0.0374 | 9.3 |
| cup | conf 0.20 | 2 | 1 | 0.7390 | 0.0155 | 10.8 |
| cup | conf 0.25 | 2 | 2 | 0.7305 | 0.0301 | 9.8 |
| cup | conf 0.30 | 3 | 1 | 0.7246 | 0.0210 | 9.3 |
| GA | entropy | 5 | 5 | 0.8244 | 0.0198 | 15.2 |
| GA | variance | 5 | 3 | 0.7865 | 0.0306 | 13.0 |
| GA | conf 0.10 | 3 | 2 | 0.7789 | 0.0402 | 14.3 |
| GA | conf 0.20 | 5 | 2 | 0.7881 | 0.0219 | 17.9 |
| GA | conf 0.25 | 4 | 2 | 0.8007 | 0.0128 | 21.4 |
| GA | conf 0.30 | 5 | 1 | 0.7696 | 0.0376 | 18.2 |
| FAZ | baseline (2 runs only) | – | – | 0.8950 | 0.0016 | 25.0 |
| FAZ | entropy | 4 | 2 | 0.9067 | 0.0055 | 17.6 |
| FAZ | variance | 4 | 3 | 0.8980 | 0.0069 | 13.9 |
| FAZ | conf 0.05 | 1 | 1 | 0.8918 | 0.0092 | 16.5 |
| FAZ | conf 0.10 | 3 | 3 | 0.8877 | 0.0105 | 15.8 |
| FAZ | conf 0.15 | 3 | 4 | 0.8894 | 0.0087 | 16.7 |
| FAZ | conf 0.20 | 1 | 3 | 0.8907 | 0.0076 | 18.0 |
| FAZ | conf 0.25 | 2 | 4 | 0.8879 | 0.0089 | 15.7 |
| FAZ | conf 0.30 | 3 | 4 | 0.8829 | 0.0146 | 14.9 |

## Weight formula (KNOWN_ISSUES K1): γ = 1 was weighted

If γ = 1 meant no weighting (refactor W = 1 + (γ−1)U, or `uat_old` W = γ^U), every
γ = 1 cell of the entropy and conformal arms would be a replicate of the same
unweighted training: identical loss, data and settings, differing only by run-to-
run noise. (The variance arm is excluded: it predicts from a different mask token.)

Across the 35 γ = 1 entropy/conformal cells (7 arms × 5 W, 10 runs each):

| Dataset | Kruskal–Wallis, test Dice | Kruskal–Wallis, epochs | Cell means |
|---|---|---|---|
| cup | P = 1.6 × 10⁻¹⁵ | P = 1.2 × 10⁻¹⁶ | 0.605 – 0.742 |
| FAZ | P = 4.3 × 10⁻³ | P = 9.8 × 10⁻⁴ | 0.858 – 0.896 |

And the differences are systematic, not noise. In cup, γ = 1 conformal arms order
by α, i.e. by how much of the image is flagged uncertain: α = 0.05 (largest
uncertain area) has the lowest Dice and stops earliest (0.665, 6.7 epochs), rising
through α = 0.10 (0.696), 0.15 (0.697), 0.20 (0.714), 0.25 (0.713), 0.30 (0.716) to
entropy (0.733, 9.7 epochs). A dose–response in the amount of uncertainty is
exactly what weighting predicts and impossible without it.

**Conclusion:** at γ = 1 the loss was weighted. The grid did **not** use the
refactor's 1 + (γ−1)U or `uat_old`'s γ^U. This is consistent with the paper's
W = 1 + γU (the only documented formula that weights at γ = 1); the data cannot
distinguish it from other formulas with that property.

## The Table's epoch counts cannot be reproduced from these runs (ERRATA E13)

The convergence script counts epoch rows in `loss_parameters.csv` for the best cell
(`2_run_training_curves_analysis.py`, `avg_total_epochs`). For the reported cells
above, only one Table value matches (GA Conf-0.25: 21.4 = 21.4). No other measure
matches either (epochs to 90/95/99 % of the loss drop, best epoch, stop epoch),
e.g. cup entropy: Table 9.8, rows 15.5, to-90 % 5.7, best epoch 11.5;
FAZ Conf-0.05: Table 9.4, rows 16.5. The Table was most likely computed from a
different run set (earlier runs, later overwritten or deleted), which we cannot
check, so we cannot say it is wrong about its own runs, only that it is not
reproducible from the runs behind the figures. Its baseline columns cannot be
recomputed at all (no cup/GA baseline runs survive). The parity re-runs will
produce a convergence table from runs whose provenance is recorded.

## Other confirmations

- **E8 confirmed:** all 4,888 early-stopped runs have exactly (stop epoch − 1)
  epoch rows; the triggering epoch is never written.
- **K9 resolved:** the grid code logged `val_uncertainty` (0.0 during warmup,
  then the mean of the uncertainty map); same quantity as ours (we log NaN during
  warmup instead of 0.0).
- **"Final Gamma: 5.000000" in baseline `test_metrics.txt`** is an artifact: the
  writer prints the configured γ of the loss object, which the baseline never
  applies (ERRATA E12).
- **E6 (GA α = 0.05) cannot be confirmed** from these files: no GA α = 0.05 run
  folder survives at all, consistent with every such run crashing before writing
  results, but equally consistent with deletion.
