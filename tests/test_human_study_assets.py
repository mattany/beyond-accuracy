import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "human_study_canonical_blobs.json"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

OBSOLETE_JUDGE_VALIDATION_DIRS = [
    "balanced_30_dataset_humor_v5_conn_v4",
    "balanced_30_metaphor_v8_scaffolding_v2",
    "balanced_dataset",
    "balanced_dataset_humor_v4_conn_v3",
    "balanced_dataset_humor_v5_conn_v4",
    "balanced_dataset_scaf_v2",
    "balanced_dataset_v2",
    "balanced_dataset_v8",
    "metaphor_v6_human",
    "tie_breaker_dataset",
    "unbalanced_dataset",
]

LABELSTUDIO_EXPORTS = [
    "human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json",
    "human_study/judge_validation/tie_breaker_v2/labelstudio_output.json",
]


def git_blob_hash(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest


def load_labelstudio_export(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


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


@pytest.mark.parametrize("directory", OBSOLETE_JUDGE_VALIDATION_DIRS)
def test_superseded_judge_validation_areas_are_absent(directory):
    assert not (ROOT / "human_study/judge_validation" / directory).exists()


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


def test_prep_scripts_document_api_requirement():
    for relative_path in (
        "human_study/judge_validation/analyze_balance_options.py",
        "human_study/judge_validation/run_connection_reasons.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "API-required prep utility" in text
        assert "publication rerun command set" in text


def test_prep_scripts_import_canonical_rubrics_modules():
    for relative_path in (
        "human_study/judge_validation/analyze_balance_options.py",
        "human_study/judge_validation/run_connection_reasons.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "from evaluation.rubrics.settings import OPENAI_API_KEY" in text
        assert "from evaluation.rubrics.custom_metrics.metrics import" in text
        assert "from custom_metrics.metrics import" not in text
        assert "from config import OPENAI_API_KEY" not in text


def test_fetch_reddit_formatting_uses_environment_credentials():
    text = (
        ROOT / "human_study/judge_validation/fetch_reddit_formatting.py"
    ).read_text(encoding="utf-8")
    assert 'REDDIT_CLIENT_ID = "' not in text
    assert 'REDDIT_CLIENT_SECRET = "' not in text
    assert 'required_env("REDDIT_CLIENT_ID")' in text
    assert 'required_env("REDDIT_CLIENT_SECRET")' in text


def test_fetch_reddit_formatting_requires_env_before_api_use(monkeypatch):
    script = ROOT / "human_study/judge_validation/fetch_reddit_formatting.py"
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(sys, "argv", [str(script), "missing.csv"])

    namespace = runpy.run_path(str(script), run_name="credential_check")

    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        namespace["main"]()


def test_prep_import_paths_resolve_without_api_calls():
    script = """
import importlib.util
from pathlib import Path

root = Path(r"{root}")
for module_name in (
    "evaluation.rubrics.settings",
    "evaluation.rubrics.custom_metrics.metrics",
):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, module_name
print("ok")
""".format(root=ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("artifact", load_manifest()["artifacts"], ids=lambda item: item["path"])
def test_moved_canonical_artifact_matches_parent_blob(artifact):
    path = ROOT / artifact["path"]
    assert path.is_file(), f"missing canonical artifact: {artifact['path']}"
    assert git_blob_hash(path) == artifact["git_blob"]


@pytest.mark.parametrize("relative_path", LABELSTUDIO_EXPORTS)
def test_labelstudio_exports_are_anonymized(relative_path):
    tasks = load_labelstudio_export(relative_path)
    serialized = json.dumps(tasks)
    assert not EMAIL_RE.search(serialized)
    for task in tasks:
        assert "updated_by" not in task
        for ann in task.get("annotations", []):
            assert "updated_by" not in ann
            email = ann["completed_by"]["email"]
            assert email.startswith("annotator_")


@pytest.mark.parametrize("relative_path", LABELSTUDIO_EXPORTS)
def test_labelstudio_export_stats_preserved(relative_path):
    manifest = load_manifest()
    expected = manifest["privacy_sanitized_exports"][relative_path]["annotation_stats"]
    tasks = load_labelstudio_export(relative_path)
    from human_study.judge_validation.labelstudio_privacy import annotation_stats

    assert annotation_stats(tasks) == expected


@pytest.mark.parametrize("relative_path", LABELSTUDIO_EXPORTS)
def test_labelstudio_export_matches_privacy_manifest(relative_path):
    manifest = load_manifest()
    expected_blob = manifest["privacy_sanitized_exports"][relative_path]["git_blob"]
    assert git_blob_hash(ROOT / relative_path) == expected_blob


def test_verification_outputs_not_added_beside_final_export():
    export_dir = ROOT / "human_study/judge_validation/balanced_dataset_v2_human"
    assert not (export_dir / "human_llm_correlations.png").exists()
    assert not (export_dir / "intercoder_reliability.png").exists()


def test_intercoder_reliability_requires_output_dir():
    source_json = (
        ROOT
        / "human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "human_study/judge_validation/intercoder_reliability.py"),
            str(source_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--output-dir" in result.stderr or "--output-dir" in result.stdout


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
    assert "@" not in (results_dir / "intercoder_reliability.csv").read_text()


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


def test_teacher_significance_does_not_mutate_canonical_outputs(tmp_path):
    watched = sorted(
        (ROOT / "human_study/preferences/data").glob("*.csv"),
        key=lambda path: path.name,
    )
    before = {path: git_blob_hash(path) for path in watched}

    result = subprocess.run(
        [sys.executable, str(ROOT / "human_study/preferences/teacher_significance.py")],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert "@" not in result.stdout
    assert "@" not in result.stderr
    for path in watched:
        assert git_blob_hash(path) == before[path]
