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

## 3. Parity: SAM ViT-H, paper's configurations (260 runs)

```bash
python scripts/make_jobs.py configs/errata/parity_sam.yaml --out jobs/parity_sam.jsonl
sbatch --array=0-259%15 scripts/slurm_array.sbatch jobs/parity_sam.jsonl   # Alpine
# or: scripts/run_local.sh jobs/parity_sam.jsonl <gpu> [start] [end]
python scripts/compare_arms.py --root results/errata --backbone sam_vit_huge --mode parity
```

Pass criterion: DECISIONS D12. **If parity fails, stop** and diagnose before
MedSAM: compare against the surviving paper-era runs (K7), then the fidelity
switches (`weight_form`, `entropy_normalize`).

## 4. Weight-formula check (K1, 20 runs)

```bash
python scripts/make_jobs.py configs/errata/gamma1_formula_check.yaml --out jobs/gamma1.jsonl
sbatch --array=0-19 scripts/slurm_array.sbatch jobs/gamma1.jsonl
python scripts/compare_arms.py --root results/errata --backbone sam_vit_huge --loss_dir weighted_bce_v2_gm1
```

Compare the γ = 1 arms under `weighted_bce_v2` (paper formula, from step 3) and
`weighted_bce_v2_gm1` (no weighting) with the paper's Table and Figures 4 and 6.

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
