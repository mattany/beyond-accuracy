from __future__ import annotations

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_artifacts_use_public_layout():
    required = [
        "evaluation/README.md",
        "evaluation/rubrics/settings.py",
        "evaluation/results/rubric_scores/analogy_v2.csv",
        "evaluation/results/preference_metrics/regression_metrics_merged.csv",
        "evaluation/results/preference_metrics/.checkpoints/metaphor_v8/explanation_a.json",
        "evaluation/model_outputs/main/all_models_joined.csv",
        "evaluation/model_outputs/scillama3/output.csv",
        "evaluation/factuality/truthfulqa_results/truthfulqa_summary_latest.csv",
        "evaluation/factuality/trust_llm/heatmap.py",
    ]

    assert [path for path in required if not (ROOT / path).is_file()] == []


def test_environment_settings_strip_values_and_resolve_repository(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  example-openai-key  ")
    monkeypatch.setenv("HF_TOKEN", "   ")

    from evaluation.rubrics import settings

    settings = importlib.reload(settings)

    assert settings.PROJECT_ROOT == ROOT
    assert settings.OPENAI_API_KEY == "example-openai-key"
    assert settings.HF_TOKEN is None
    assert settings.result_directory(9) == ROOT / "evaluation/results/rubric_scores"
    assert settings.result_directory(10) == ROOT / "evaluation/results/preference_metrics"
    with pytest.raises(ValueError, match="canonical result run"):
        settings.result_directory(8)


def test_canonical_results_remain_complete():
    rubric_scores = ROOT / "evaluation/results/rubric_scores"
    preference_metrics = ROOT / "evaluation/results/preference_metrics"

    assert len(list(rubric_scores.glob("*.csv"))) == 18
    assert len(list(preference_metrics.glob("*.csv"))) == 10
    assert (rubric_scores / "wilcoxon_tests.py").is_file()
    assert (rubric_scores / "aggregations_v2/total_scores.csv").is_file()


def test_evaluation_readme_distinguishes_factuality_provenance():
    text = (ROOT / "evaluation/README.md").read_text(encoding="utf-8")

    assert "truthfulqa_visualization.py" in text
    assert "custom_metrics/aggregate_v2.py" in text
    assert "GPT-5.2" in text
    assert "TrustLLM" in text
    assert "not" in text.split("TrustLLM", maxsplit=1)[1]
