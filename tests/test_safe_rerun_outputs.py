from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from evaluation.rubrics.custom_metrics import aggregate_v2


ROOT = Path(__file__).resolve().parents[1]

PROTECTED_CANONICAL_OUTPUTS = [
    "human_study/preferences/data/dpo_rubric_up_pref_down.csv",
    "evaluation/results/rubric_scores/bootstrap/bootstrap_v2_results.csv",
    "evaluation/results/rubric_scores/aggregations_v2/total_scores.csv",
    "evaluation/results/rubric_scores/aggregations_v2/metric_scores.csv",
    "evaluation/results/rubric_scores/aggregations_v2/stacked_total_scores.png",
]


def git_blob_hash(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def protected_blob_hashes() -> dict[str, str]:
    return {
        relative: git_blob_hash(ROOT / relative)
        for relative in PROTECTED_CANONICAL_OUTPUTS
        if (ROOT / relative).is_file()
    }


def test_aggregate_v2_help_exposes_bootstrap_dir():
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.rubrics.custom_metrics.aggregate_v2", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--bootstrap-dir" in result.stdout


def test_missing_bootstrap_generates_under_bootstrap_dir(tmp_path, monkeypatch):
    rubric_dir = tmp_path / "rubric_scores"
    rubric_dir.mkdir()
    scratch_bootstrap = tmp_path / "scratch_bootstrap"
    calls: list[dict[str, str | None]] = []

    def fake_bootstrap(directory, n_bootstrap=10000, bootstrap_dir=None, **kwargs):
        del n_bootstrap, kwargs
        calls.append({"directory": directory, "bootstrap_dir": bootstrap_dir})
        bootstrap_path = Path(bootstrap_dir)
        bootstrap_path.mkdir(parents=True, exist_ok=True)
        out = bootstrap_path / "bootstrap_v2_results.csv"
        pd.DataFrame(
            {
                "Model": ["model_a"],
                "Total_Score": [0.5],
                "Bootstrap_SE": [0.01],
                "CI_Lower": [0.4],
                "CI_Upper": [0.6],
            }
        ).to_csv(out, index=False)
        return pd.read_csv(out)

    monkeypatch.setitem(
        sys.modules,
        "evaluation.rubrics.custom_metrics.bootstrap",
        types.SimpleNamespace(bootstrap_analysis_v2=fake_bootstrap),
    )

    result = aggregate_v2.load_bootstrap_confidence_intervals(
        str(rubric_dir),
        expected_models={"model_a"},
        bootstrap_dir=str(scratch_bootstrap),
    )

    assert result is not None
    assert calls == [
        {"directory": str(rubric_dir), "bootstrap_dir": str(scratch_bootstrap)}
    ]
    assert (scratch_bootstrap / "bootstrap_v2_results.csv").is_file()
    assert not (rubric_dir / "bootstrap").exists()


def test_metaphor_overoptimization_requires_output():
    result = subprocess.run(
        [sys.executable, "human_study/preferences/metaphor_overoptimization.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--output" in result.stderr


def test_safe_documented_commands_preserve_canonical_blobs(tmp_path):
    before = protected_blob_hashes()
    assert before, "expected at least one protected canonical artifact"

    aggregate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.rubrics.custom_metrics.aggregate_v2",
            "--output-dir",
            str(tmp_path / "rubric_aggregations_v2"),
            "--bootstrap-dir",
            str(tmp_path / "rubric_bootstrap"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert aggregate_result.returncode == 0, aggregate_result.stderr or aggregate_result.stdout

    metaphor_result = subprocess.run(
        [
            sys.executable,
            "human_study/preferences/metaphor_overoptimization.py",
            "--output",
            str(tmp_path / "dpo_rubric_up_pref_down.csv"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert metaphor_result.returncode == 0, metaphor_result.stderr or metaphor_result.stdout

    after = protected_blob_hashes()
    assert before == after
