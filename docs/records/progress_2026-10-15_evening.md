# Progress, 2026-10-15 18:44 (results/errata_v2, ~6 h after restart)

Finished: GA baseline, entropy, variance, Conf-0.25 — runs 3 and 8 each (8 runs).
Running: cup baseline runs 1, 4, 6, 9 (5–23 epochs).

**E14 reproduced directly:** GA entropy γ5 W5 epoch-1 validation loss — ours weighted
1.368 / 1.651, unweighted 0.452 / 0.623; paper-era logged value for the cell ≈ 1.52.

**Timing:** cup epochs 900–1,270 s (train 238 + val 51 + test 146 images, two SAM
ViT-H runs per A100, ≈ 2.3 s per image, GPU-bound); GA 180–270 s. With training
continued until every criterion has stopped, cup runs reach ~19–27 epochs (val Dice
keeps improving), ≈ 6–7 h per cup run. Estimated completion of the 80 core runs:
~Sunday night / Monday (2026-10-18/19).

**Illustration only (n = 2 per arm, not for inference), GA:** test soft Dice
baseline / entropy — unweighted criterion 0.641 / 0.641 (entropy stopped inside
warmup); weighted 0.641 / 0.818 (paper-era runs 0.824); validation-Dice criterion
0.724 / 0.818. Same-seed curves: baseline validation Dice peaks at epoch 5 then
declines; entropy (identical through epoch 5) keeps rising to 0.92 (test 0.86 at
epoch 19, run 3). Under entropy weighting validation BCE rises while Dice improves.
Cup baseline under the validation-Dice criterion reaches test Dice ≈ 0.77–0.79 at
epochs 14–22 (two runs still running).

**Open:** every log contains a non-fatal "Traceback" (runs complete); to be identified.
