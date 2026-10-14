# Records

Dated outputs of each step in docs/REPRODUCTION.md, committed as they are produced,
so the erratum can cite exactly what was run and observed. Suggested names:

- `environment_YYYY-MM-DD.txt` — `pip freeze` of the environment used
- `datasets_YYYY-MM-DD.txt` — output of `scripts/check_datasets.py` (split counts, CSV SHA-256)
- `inspect_sam-vit-huge.json`, `inspect_medsam.json` — `scripts/inspect_backbone.py`
- `paper_artifacts_inventory.md` — what survives of the paper-era results (K7)
- `verdict_parity_YYYY-MM-DD.txt`, `verdict_medsam_YYYY-MM-DD.txt` — `scripts/compare_arms.py`
