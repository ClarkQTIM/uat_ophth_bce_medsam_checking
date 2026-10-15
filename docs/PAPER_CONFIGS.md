# What the paper reports: configurations and reference values

Source: the published PDF (TVST 2026;15(2):19). Figure pages were rendered at
300 dpi and the x-axis labels and box plots read from the renders on 2026-10-14.
These values are the reproduction targets. If the paper-era result folders are
recovered (KNOWN_ISSUES K7), replace the figure-read values below with the exact
numbers and note it here.

## Reported configurations (→ `configs/errata/paper_arms.yaml`)

γ in the paper's convention (W = 1 + γU); W = n_warmup.

| Dataset | Arm | γ | W | Source |
|---|---|---|---|---|
| cup (OC) | entropy | 5 | 5 | Fig 1, Fig 7a legend |
| cup | variance | 3 | 5 | Fig 1, Fig 7a |
| cup | conformal α=0.05 | 1 | 5 | Fig 4 |
| cup | conformal α=0.10 | 5 | 1 | Fig 4 |
| cup | conformal α=0.15 | 1 | 1 | Fig 4 |
| cup | conformal α=0.20 | 2 | 1 | Fig 1, Fig 4, Fig 7a |
| cup | conformal α=0.25 | 2 | 2 | Fig 4 |
| cup | conformal α=0.30 | 3 | 1 | Fig 4 |
| GA | entropy | 5 | 5 | Fig 2, Fig 7b |
| GA | variance | 5 | 3 | Fig 2, Fig 7b |
| GA | conformal α=0.10 | 3 | 2 | Fig 5 |
| GA | conformal α=0.15 | 2 | 1 | Fig 5 |
| GA | conformal α=0.20 | 5 | 2 | Fig 5 |
| GA | conformal α=0.25 | 4 | 2 | Fig 2, Fig 5, Fig 7b |
| GA | conformal α=0.30 | 5 | 1 | Fig 5 |
| octa (FAZ) | entropy | 4 | 2 | Fig 3, Fig 7c |
| octa | variance | 4 | 3 | Fig 3, Fig 7c |
| octa | conformal α=0.05 | 1 | 1 | Fig 3, Fig 6, Fig 7c |
| octa | conformal α=0.10 | 3 | 3 | Fig 6 |
| octa | conformal α=0.15 | 3 | 4 | Fig 6 |
| octa | conformal α=0.20 | 1 | 3 | Fig 6 |
| octa | conformal α=0.25 | 2 | 4 | Fig 6 |
| octa | conformal α=0.30 | 3 | 4 | Fig 6 |

GA has no α = 0.05 arm in Figure 5 (cannot be calibrated with n_cal = 10;
ERRATA E6). Every Figure 7 legend matches the corresponding Figure 1–3 label.

## Exact values from the paper's own runs (2026-10-14)

The paper-era result tree survives in part, and it is the data behind Figures 1–6
(docs/records/paper_era_inventory.md). Exact means and SDs for every reported cell
are tabulated there and used by `scripts/compare_arms.py`; they agree with the
figure readings below to ±0.001. No cup or GA baseline runs survive, so the
baselines remain figure readings.

## Reference values: mean test soft Dice (read from Figures 1–3, ±0.003)

Green triangles = means; orange lines = medians. n = 10 runs per arm.

| Dataset | Baseline | Entropy | Variance | Best conformal |
|---|---|---|---|---|
| cup (Fig 1) | 0.705 (median 0.713) | 0.781 | 0.742 | 0.739 (α=0.20) |
| GA (Fig 2) | 0.728 | 0.825 | 0.786 | 0.801 (α=0.25) |
| octa (Fig 3) | 0.888 | 0.907 | 0.898 | 0.892 (α=0.05) |

Significance markers in the paper (Mann-Whitney vs baseline):
cup: entropy, variance and Conf-0.20 all P < 0.01. GA: entropy, variance,
Conf-0.25 all P < 0.01. FAZ: entropy P < 0.01, variance P < 0.05, Conf-0.05 n.s.

## Reference values: Table (convergence, exact as printed)

**Warning (ERRATA E13):** these numbers do not come from the runs behind the
figures; they are recorded here as printed, not as reproduction targets.

| Dataset | Method | Avg. epochs | Diff. | P |
|---|---|---|---|---|
| Cup | Baseline | 8.5 | — | — |
| Cup | Conf-0.05 | 7.1 | −16.5% | 0.021 |
| Cup | Conf-0.10 | 8.1 | −4.7% | 0.608 |
| Cup | Conf-0.15 | 8.2 | −3.5% | 0.607 |
| Cup | Conf-0.20 | 9.0 | +5.9% | 0.331 |
| Cup | Conf-0.25 | 8.8 | +3.5% | 0.815 |
| Cup | Conf-0.30 | 8.2 | −3.5% | 0.559 |
| Cup | Variance | 9.0 | +5.9% | 0.744 |
| Cup | Entropy | 9.8 | +15.3% | 0.084 |
| GA | Baseline | 12.0 | — | — |
| GA | Conf-0.05 | 5.0 | −58.3% | 0.150 (suspect: ERRATA E6) |
| GA | Conf-0.10 | 12.8 | +6.7% | 0.909 |
| GA | Conf-0.15 | 10.8 | −10.0% | 0.158 |
| GA | Conf-0.20 | 14.1 | +17.5% | 0.108 |
| GA | Conf-0.25 | 21.4 | +78.3% | 0.004 |
| GA | Conf-0.30 | 16.0 | +33.3% | 0.254 |
| GA | Variance | 11.0 | −8.3% | 1.000 |
| GA | Entropy | 12.4 | +3.3% | 0.909 |
| FAZ | Baseline | 17.9 | — | — |
| FAZ | Conf-0.05 | 9.4 | −47.5% | <0.001 |
| FAZ | Conf-0.10 | 15.2 | −15.1% | 0.129 |
| FAZ | Conf-0.15 | 15.2 | −15.1% | 0.093 |
| FAZ | Conf-0.20 | 13.9 | −22.3% | 0.005 |
| FAZ | Conf-0.25 | 15.9 | −11.2% | 0.321 |
| FAZ | Conf-0.30 | 13.5 | −24.6% | 0.020 |
| FAZ | Variance | 13.0 | −27.4% | 0.150 |
| FAZ | Entropy | 19.9 | +11.2% | 0.595 |

Epoch counts follow the paper-era counting (ERRATA E8); compare with the
`loss_parameters.csv` numeric-row count, not `epoch_log.csv`.
