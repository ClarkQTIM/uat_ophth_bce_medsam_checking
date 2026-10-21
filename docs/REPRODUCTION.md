# How to reproduce and re-run

Everything here runs from the repo root. Each step says what it checks and what
to do if it fails. Record outcomes in `docs/records/` (one dated file per step)
so the erratum can cite them.

## 0. Environment and tests (10 min)

```bash
cd /scratch90/chris_/paper_crisis
conda activate <env with torch, transformers, monai, pandas, scipy, scikit-learn, pyyaml, pytest>
pytest -q
```

The tests check, without real data or weights:
- each weight formula, and that γ = 1 is weighted only under the paper's formula;
- entropy, variance and conformal maps match the paper-era definitions;
- the GA α = 0.05 calibration failure (ERRATA E6) is reproduced;
- a full training run of every arm on a tiny random SAM writes paper-compatible outputs.

`tests/test_forward_and_loop.py` skips itself if the tiny SAM cannot be built with
your `transformers` version; the smoke run (step 2b) then covers the same ground.

Record: `pip freeze > docs/records/environment_<date>.txt`.

## 1. Datasets: are these the paper's splits? (K6)

Fill in `configs/datasets.yaml`, then:

```bash
python scripts/check_datasets.py --datasets configs/datasets.yaml | tee docs/records/datasets_<date>.txt
```

Must print `ALL CHECKS PASSED`: split counts equal the paper's, every file exists,
masks are binary. The SHA-256 of each CSV goes into the erratum record.
**If counts differ, stop:** without the original splits this is not the same experiment.

## 2. Backbones: right model, sane outputs (E1, K4, K10)

```bash
python scripts/inspect_backbone.py --backbone sam-vit-huge --csv <cup csv> --n 8
python scripts/inspect_backbone.py --backbone medsam       --csv <cup csv> --n 8
mv inspect_*.json docs/records/
```

Check: SAM ≈ 641M parameters, MedSAM ≈ 94M (the script refuses otherwise);
processor settings for each; for MedSAM, whether the three multimask outputs are
distinct (`multi_absdiff_*`, `multi_var_mean` not ≈ 0). If they are degenerate,
note it in KNOWN_ISSUES K4 before step 5.

## 2b. Smoke run: every code path on real data (minutes)

```bash
python scripts/make_jobs.py configs/errata/smoke.yaml --out jobs/smoke.jsonl
scripts/run_local.sh jobs/smoke.jsonl 0
ls results/smoke/cup/*/weighted_bce_v2/*/*/*/run_1/test_metrics.txt   # expect 8
```

Use the `seconds` and `peak_gpu_gb` columns of the `epoch_log.csv` files to
estimate the cost of steps 3-5 and how many runs fit side by side on one GPU
(batch size stays 1, as in the paper; parallelism comes from concurrent runs).

## 3. Parity: SAM ViT-H, paper's configurations

Runs track every validation criterion and test Dice per epoch (DECISIONS D16).
Priority subset first (80 runs: cup + GA, the arms of Figures 1–2), then the rest:

```bash
python scripts/make_jobs.py configs/errata/parity_sam_core.yaml --out jobs/parity_core.jsonl
python scripts/make_jobs.py configs/errata/parity_sam.yaml --out jobs/parity_sam_cupga.jsonl --only-dataset cup ga
# 2 SAM ViT-H runs per A100 80GB; e.g. 4 screens, each a slice of the job file:
#   scripts/run_local.sh jobs/parity_core.jsonl <gpu> <start> <end>
python scripts/select_offline.py --root results/errata_v2 --backbone sam_vit_huge
```

Analysis and criteria: DECISIONS D17 (Q1 parity under `as_run`; Q2 `unweighted`;
Q3 `dice`). `scripts/compare_arms.py` still summarizes the primary criterion only.
Runs already complete are skipped, so the second job file only adds what is missing.

## 4. Weight-formula check (K1) — no longer needed (resolved from the paper-era runs, ERRATA E3/E14)

```bash
python scripts/make_jobs.py configs/errata/gamma1_formula_check.yaml --out jobs/gamma1.jsonl
sbatch --array=0-19 scripts/slurm_array.sbatch jobs/gamma1.jsonl
python scripts/compare_arms.py --root results/errata --backbone sam_vit_huge --loss_dir weighted_bce_v2_gm1
```

Compare the γ = 1 arms under `weighted_bce_v2` (paper formula, from step 3) and
`weighted_bce_v2_gm1` (no weighting) with the paper's Table and Figures 4 and 6.

## 5a. MedSAM core first (80 runs; started 2026-10-19 after the SAM core finished)

Same 8 cells as `parity_sam_core`, on MedSAM (≈ 9 GB per run, so several per GPU).
Screens, 4 per GPU, each a slice of 10 lines (lines alternate cup/GA per run index):

```bash
python scripts/make_jobs.py configs/errata/medsam_core.yaml --out jobs/medsam_core.jsonl
screen -dmS m0 bash -c "scripts/run_local.sh jobs/medsam_core.jsonl 0 0 9"
screen -dmS m1 bash -c "scripts/run_local.sh jobs/medsam_core.jsonl 0 10 19"
# ... m2..m7: GPU 0 for 20-39, GPU 1 for 40-79
python scripts/select_offline.py --root results/errata_v2 --backbone medsam
```

## 5b. FAZ on a new split (DECISIONS D19; ERRATA E15)

```bash
python scripts/create_octa_faz_csv.py --images_dir <OCTA_6mm projection folder> \
    --masks_dir <OCTA_6mm GT_FAZ folder> --out_csv data/octa_faz_split.csv --seed 2026
# set octa.csv in configs/datasets.yaml to data/octa_faz_split.csv, then
python scripts/check_datasets.py
python scripts/make_jobs.py configs/errata/faz_core_medsam.yaml --out jobs/faz_core_medsam.jsonl
python scripts/make_jobs.py configs/errata/faz_core_sam.yaml --out jobs/faz_core_sam.jsonl
```

## 5c. Patient-level split sensitivity (DECISIONS D20)

```bash
python scripts/make_patient_split.py --dataset ga  --in data/ga_split.csv  --out data/ga_patient_split.csv  --seed 1020
python scripts/make_patient_split.py --dataset cup --in data/cup_split.csv --out data/cup_patient_split.csv --seed 1020
python scripts/make_jobs.py configs/patient_split/medsam_patient.yaml \
    --datasets configs/patient_split/datasets_patient.yaml --out jobs/patient_medsam.jsonl
python scripts/select_offline.py --root results/patient_split_v1 --backbone medsam
```

## 5. MedSAM primary: the erratum's evidence (260 runs)

```bash
python scripts/make_jobs.py configs/errata/medsam_primary.yaml --out jobs/medsam_primary.jsonl
sbatch --array=0-259%15 scripts/slurm_array.sbatch jobs/medsam_primary.jsonl
python scripts/compare_arms.py --root results/errata --backbone medsam --mode primary
```

Criterion fixed in advance: DECISIONS D12. Do not re-tune γ or W.

## Output layout

```
results/errata/<dataset>/<backbone_tag>/<loss_dir>/gamma_<γ>/n_warmup_<W>/<arm>/run_<n>/
    run_manifest.json    backbone id + parameter count, git commit, full config, environment
    loss_parameters.csv  paper-era format (+ test row)
    epoch_log.csv        complete per-epoch log
    test_metrics.txt     "Foreground Dice Score: ..." (paper metric) + extra metrics
    test_per_image.csv   per-image soft/hard Dice, IoU
    conformal_log.json   (conformal arms) every recalibration: level, n_cal, q̂ stats
    epoch_images/        first validation batch per epoch
```

`<backbone_tag>` is `sam_vit_huge` or `medsam`; `<loss_dir>` is `weighted_bce_v2`
(paper formula) or `weighted_bce_v2_gm1` / `weighted_bce_v2_exp`. The baseline is
stored under `gamma_0/n_warmup_0/none/`. The layout matches the paper-era analysis
scripts (which also accept the old `recalc_<W>` name).

A run is skipped if its `test_metrics.txt` already exists, so arrays can be
resubmitted after failures.
