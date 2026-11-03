# Does UAT make MedSAM better at flagging its own errors? — results 2026-11-03

Pre-registered in DECISIONS D22. MedSAM; baseline, entropy and conformal with the paper's cells;
GA and cup on patient-level splits (D20), FAZ on the seed-2026 split (D19); 10 runs per arm,
90/90 runs, all on the H100 NVL machine. Every epoch logged test error-detection AUROC
(entropy of the model's own predicted probabilities, the same measure for every arm),
calibration (ECE, Brier, NLL) and Dice; each stopping rule read off offline.
Primary: entropy AUROC at the validation-Dice-selected epoch. Raw: `select_offline_output.txt`.

Δ = arm − baseline; P = Mann-Whitney (Holm across the 2 UQ arms within a dataset).

## Primary: error-detection AUROC (higher = better), validation-Dice stopping

| Dataset | Baseline | Entropy-UAT Δ (P, Holm) | Conformal-UAT Δ (P, Holm) |
|---|---|---|---|
| Cup | 0.950 | **−0.019 (0.0003, 0.0007)**; worse in 10/10 seeds | −0.001 (0.16, 0.16) |
| GA | 0.943 | −0.004 (0.62, 1.0) | −0.005 (0.73, 1.0) |
| FAZ | 0.981 | +0.004 (1.0, 1.0) | +0.004 (0.43, 0.86) |

**Pre-registered verdict: not upheld for any dataset.** No UAT arm detects its own errors better
than the baseline; entropy-UAT is significantly *worse* for cup.

## Secondary: calibration (lower = better), validation-Dice stopping

| Dataset | ECE: baseline → entropy Δ (Holm) | NLL: entropy Δ (Holm) | Brier: entropy Δ (Holm) | Conformal |
|---|---|---|---|---|
| Cup | 0.019 → **+0.009 (0.002)** | **+0.035 (0.004)** | **+0.004 (0.003)** | no significant change |
| GA | 0.013 → **+0.004 (0.002)** | **+0.016 (0.012)** | −0.001 (0.35) | no significant change |
| FAZ | 0.0012 → **+0.0001 (0.035)** | +0.000 (1.0) | +0.000 (0.86) | no significant change |

Entropy-UAT is **worse calibrated** in all three datasets (significant ECE increase everywhere).

## Other observations

- **Conformal's own uncertainty map (prediction-set size) barely detects errors:** AUROC
  0.52–0.66 (0.5 = chance), against ≥ 0.93 for plain entropy of the same models.
- **Faster convergence replicated:** under the common rule entropy reaches its best validation
  Dice earlier than the baseline in all 3 datasets again (cup 10.9 vs 18.4, P = 0.011; GA 15.1 vs
  18.0, P = 0.40; FAZ 20.8 vs 32.9, P < 0.001). Across all experiments: 12 of 12 settings.
- **Dice in this replicate** (same cells, splits, seeds and hardware as the D20/FAZ runs; only
  the extra logging differs): GA entropy +0.037 (P = 0.009, Holm 0.018) vs +0.026 (P = 0.064)
  in the first run; FAZ +0.004 (P = 0.12) vs +0.009 (P = 0.006); cup +0.008 (P = 0.43) vs
  +0.017 (P = 0.12). The direction repeats; whether it crosses P = 0.05 does not
  (KNOWN_ISSUES K18).

## Reading

Entropy-UAT does not make the model's uncertainty better at flagging its own mistakes. It makes
calibration worse and, for cup, error detection worse. That fits how it trains: up-weighting
pixels where the model is unsure pushes those predictions towards confident 0 or 1, so the model
becomes more confident exactly where it is most often wrong. Conformal-UAT changes nothing
measurable. The hypothesis that motivated the work is not supported by these runs.
