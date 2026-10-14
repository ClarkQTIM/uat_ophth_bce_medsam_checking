# paper_crisis

Verification, correction and re-run of

> Clark CW, Kinder S, Nebbia G, et al. **Uncertainty-Aware Training for Ophthalmic
> Segmentation Using MedSAM.** *Transl Vis Sci Technol.* 2026;15(2):19.
> doi:10.1167/tvst.15.2.19

On 2026-10-14 we found that the code that produced the paper loaded the original
**SAM ViT-H** (`facebook/sam-vit-huge`), not MedSAM, and that a later refactor of
the code had broken the variance method. This repository exists to establish
exactly what the paper did, re-run it faithfully, re-run it on real MedSAM, and
document every step so the correction can be made openly and checked by anyone.

It also hosts a second, separate track (evidential/Dirichlet MedSAM), set up but
not yet run.

## Start here

| Document | What it is |
|---|---|
| [docs/ERRATA_LOG.md](docs/ERRATA_LOG.md) | Every discrepancy found, with evidence and status (E1–E8) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Every implementation choice and why (D1–D15), incl. pre-registered criteria |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | Open questions and how each will be resolved (K1–K10) |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Step-by-step: tests → data check → backbone check → parity → MedSAM |
| [docs/PAPER_CONFIGS.md](docs/PAPER_CONFIGS.md) | The configurations and numbers the paper reports (reproduction targets) |
| `docs/records/` | Dated outputs of each step (environment, dataset hashes, backbone checks, verdicts) |

## Summary of the plan

1. **Parity:** re-run the paper's 26 reported configurations × 10 runs with SAM
   ViT-H in this new pipeline. If they match the paper, the pipeline is faithful.
2. **Weight-formula check:** 20 runs settle which loss-weight formula the paper used.
3. **MedSAM:** the same 26 configurations × 10 runs with real MedSAM weights,
   judged by criteria fixed in advance (DECISIONS D12). This is the erratum's evidence.

## Layout

```
src/uatcheck/            the package
  backbones.py           backbone registry + parameter-count guardrail
  data.py                paper-era dataset (CSV splits, whole-image box prompt)
  forward.py             paper-era forward pass (variance from 3 multimask outputs)
  uncertainty.py         entropy / variance / conformal maps
  conformal.py           per-pixel LAC calibration
  losses.py              uncertainty-weighted BCE, explicit weight formula
  metrics.py             paper metric (soft Dice) + additional metrics
  manifest.py            run_manifest.json: what ran, on what, with which code
  train_uat.py           one UAT training run
  evidential/            evidential track (not yet run)
configs/
  datasets.yaml          split CSV paths (fill in)
  errata/                paper_arms, parity_sam, medsam_primary, gamma1_formula_check, smoke
  evidential/            papila_edl.yaml
scripts/                 run_uat, make_jobs, slurm_array.sbatch, run_local.sh,
                         check_datasets, inspect_backbone, compare_arms,
                         run_evidential, edl_loss_surface
tests/                   pytest suite (CPU, tiny random SAM; no data or weights needed)
```

## Provenance of the code

Each function reproduces a specific paper-era function from `uat_old/src/` and
says so in its docstring; every deliberate deviation is listed in DECISIONS.md.
Nothing here imports the old code.

## Status (2026-10-14)

- [x] Paper-era behavior established from `uat_old`; findings logged (ERRATA E1–E8)
- [x] Pipeline, configs, launchers, tests written (tests not yet run: run `pytest -q` first)
- [ ] Step 0–2: environment, dataset and backbone checks on the server
- [ ] Inventory surviving paper-era results (KNOWN_ISSUES K7)
- [ ] Step 3: SAM parity · Step 4: formula check · Step 5: MedSAM primary
- [ ] Draft erratum
