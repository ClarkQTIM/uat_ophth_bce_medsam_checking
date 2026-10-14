"""Job generation and output-path conventions."""

import json
import os
import subprocess
import sys

import pytest

from uatcheck.train_uat import arm_dir_name, run_dir_for, validate_config

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@pytest.mark.parametrize("manifest,n", [("parity_sam", 260), ("medsam_primary", 260),
                                        ("gamma1_formula_check", 20), ("smoke", 8)])
def test_job_counts(tmp_path, manifest, n):
    out = tmp_path / f"{manifest}.jsonl"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "make_jobs.py"),
                           os.path.join(ROOT, "configs", "errata", f"{manifest}.yaml"),
                           "--datasets", os.path.join(ROOT, "configs", "datasets.yaml"),
                           "--out", str(out)], cwd=ROOT)
    jobs = [json.loads(l) for l in open(out)]
    assert len(jobs) == n
    for j in jobs:
        validate_config(j)


def test_arm_names_match_analysis_conventions():
    assert arm_dir_name("none", None) == "none"
    assert arm_dir_name("conformal", 0.05) == "conformal_lac_005"
    assert arm_dir_name("conformal", 0.2) == "conformal_lac_020"
    assert arm_dir_name("sam_var", None) == "sam_var"


def test_run_dir_layout():
    d = run_dir_for({"output_root": "r", "dataset": "cup", "backbone": "sam-vit-huge",
                     "weight_form": "paper", "gamma": 5.0, "n_warmup": 5,
                     "uq_type": "entropy", "alpha": None, "run": 3})
    assert d.replace(os.sep, "/") == "r/cup/sam_vit_huge/weighted_bce_v2/gamma_5/n_warmup_5/entropy/run_3"
