from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


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
