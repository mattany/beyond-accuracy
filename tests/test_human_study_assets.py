import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "human_study_canonical_blobs.json"


def git_blob_hash(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["parent_commit"], manifest["artifacts"]


def test_final_human_study_artifacts_are_retained():
    expected = [
        "human_study/preferences/data/first_exp.csv",
        "human_study/preferences/data/sec_exp.csv",
        "human_study/preferences/teacher_significance.py",
        "human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json",
        "human_study/judge_validation/tie_breaker_v2/labelstudio_output.json",
        "human_study/judge_validation/labeling_interface/labelstudio_v2.xml",
        "human_study/judge_validation/labeling_interface/labelstudio_tiebreaker_v2.xml",
    ]
    assert all((ROOT / path).is_file() for path in expected)


def test_obsolete_label_studio_interfaces_are_absent():
    obsolete = [
        "labelstudio_v1.xml",
        "labelstudio_v3_metaphor.xml",
        "labelstudio_v6_metaphor.xml",
        "labelstudio_v8_metaphor.xml",
        "labelstudio_tiebreaker.xml",
        "labelstudio_scaffolding_v2.xml",
        "labelstudio_metaphor_v8_scaffolding_v2.xml",
        "labelstudio_humor_v4_connection_v3.xml",
        "labelstudio_humor_v5_connection_v4.xml",
    ]
    interface_dir = ROOT / "human_study/judge_validation/labeling_interface"
    assert all(not (interface_dir / name).exists() for name in obsolete)


def test_superseded_judge_validation_areas_are_absent():
    obsolete = [
        "balanced_dataset",
        "balanced_dataset_v2",
        "tie_breaker_dataset",
        "unbalanced_dataset",
    ]
    validation = ROOT / "human_study/judge_validation"
    assert all(not (validation / name).exists() for name in obsolete)


def test_preference_scripts_use_canonical_result_directories():
    sys.path.insert(0, str(ROOT / "human_study/preferences"))
    try:
        import logistic_regression
        import teacher_significance
    finally:
        sys.path.pop(0)

    assert logistic_regression.METRICS_DIR == (
        ROOT / "evaluation/results/preference_metrics"
    )
    assert teacher_significance.RUN_RESULTS == (
        ROOT / "evaluation/results/rubric_scores"
    )


def test_logistic_regression_cli_supports_formality():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "human_study/preferences/logistic_regression.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--with-formality" in result.stdout


def test_readability_rerun_uses_moved_preference_dataset():
    source = (
        ROOT / "evaluation/rubrics/custom_metrics/rerun_readability.py"
    ).read_text()
    assert "human_study/preferences/data/experiment_b_eval_dataset.csv" in source
    assert "scripts/experiment_b" not in source


@pytest.mark.parametrize("artifact", load_manifest()[1], ids=lambda item: item["path"])
def test_moved_canonical_artifact_matches_parent_blob(artifact):
    path = ROOT / artifact["path"]
    assert path.is_file(), f"missing canonical artifact: {artifact['path']}"
    assert git_blob_hash(path) == artifact["git_blob"]


def test_verification_outputs_not_added_beside_final_export():
    export_dir = ROOT / "human_study/judge_validation/balanced_dataset_v2_human"
    assert not (export_dir / "human_llm_correlations.png").exists()
    assert not (export_dir / "intercoder_reliability.png").exists()


def test_intercoder_reliability_writes_to_output_dir(tmp_path):
    source_json = (
        ROOT
        / "human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json"
    )
    export_copy = tmp_path / "labelstudio_output.json"
    shutil.copy2(source_json, export_copy)
    before = git_blob_hash(source_json)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "human_study/judge_validation/intercoder_reliability.py"),
            str(export_copy),
            "--output-dir",
            str(tmp_path / "results"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    results_dir = tmp_path / "results"
    assert (results_dir / "intercoder_reliability.csv").is_file()
    assert git_blob_hash(source_json) == before
    assert not (export_copy.parent / "intercoder_reliability.csv").exists()


def test_logistic_regression_with_formality_writes_to_explicit_output(tmp_path):
    canonical = (
        ROOT / "human_study/preferences/data/logistic_regression_continuous_with_formality.csv"
    )
    before = git_blob_hash(canonical)
    output_path = tmp_path / "formality.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "human_study/preferences/logistic_regression.py"),
            "--mode",
            "continuous",
            "--with-formality",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.is_file()
    assert git_blob_hash(canonical) == before
