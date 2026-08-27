from __future__ import annotations

import asyncio
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dpo import preference_dataset_generation as preference_generation


@pytest.mark.parametrize("name", ["naive_dpo.py", "organic_sft.py"])
def test_model_variant_scripts_are_valid_python(name):
    path = ROOT / "training" / "model_variants" / name
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_model_variant_representation_is_documented():
    readme = ROOT / "training" / "model_variants" / "README.md"
    assert "standalone Python" in readme.read_text(encoding="utf-8")


def test_generation_cli_passes_paths_to_main(monkeypatch, tmp_path):
    script = ROOT / "evaluation" / "rubrics" / "generate_kto_llama3_1_outputs.py"
    input_path = tmp_path / "questions.csv"
    output_path = tmp_path / "answers.csv"
    captured = {}

    deepeval = ModuleType("deepeval")
    deepeval.assert_test = lambda *args, **kwargs: None
    test_case = ModuleType("deepeval.test_case")
    test_case.LLMTestCase = object
    metrics = ModuleType("deepeval.metrics")
    metrics.AnswerRelevancyMetric = object
    mlx_model = ModuleType("mlx_model")
    mlx_model.MLXModel = object
    ollama_model = ModuleType("evaluation.rubrics.ollama_model")
    ollama_model.OllamaModel = lambda *args, **kwargs: object()
    prompt_templates = ModuleType("evaluation.rubrics.prompt_templates")
    prompt_templates.generate_prompt = lambda prompt: prompt
    prompt_templates.system_prompt = "system"

    for name, module in {
        "deepeval": deepeval,
        "deepeval.test_case": test_case,
        "deepeval.metrics": metrics,
        "mlx_model": mlx_model,
        "evaluation.rubrics.ollama_model": ollama_model,
        "evaluation.rubrics.prompt_templates": prompt_templates,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def capture_run(coroutine):
        captured["workflow"] = coroutine.cr_code.co_name
        captured["args"] = coroutine.cr_frame.f_locals.get("args")
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--input", str(input_path), "--output", str(output_path)],
    )

    runpy.run_path(str(script), run_name="__main__")

    assert captured["workflow"] == "main"
    assert captured["args"].input == input_path
    assert captured["args"].output == output_path


def test_openai_playground_fails_clearly_without_api_key(monkeypatch):
    script = ROOT / "training" / "data_generation" / "open_ai_playground.py"
    openai = ModuleType("openai")
    openai.OpenAI = lambda *args, **kwargs: pytest.fail("client created without an API key")
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", [str(script)])

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        runpy.run_path(str(script), run_name="__main__")


def test_data_generation_constants_load_keys_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " example-openai-key ")
    monkeypatch.setenv("MOONSHOT_API_KEY", " example-moonshot-key ")
    monkeypatch.delenv("TEACHER_PROVIDER", raising=False)

    import importlib

    from training.data_generation import constants

    constants = importlib.reload(constants)

    assert constants.PROVIDERS["openai"]["api_key"] == "example-openai-key"
    assert constants.PROVIDERS["kimi"]["api_key"] == "example-moonshot-key"


def test_data_generation_client_requires_provider_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TEACHER_PROVIDER", raising=False)

    import importlib

    from training.data_generation import constants

    constants = importlib.reload(constants)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        constants.get_client()


def test_push_to_hub_requires_repository(tmp_path):
    common = [
        "--scores",
        str(tmp_path / "scores.csv"),
        "--answers",
        str(tmp_path / "answers.csv"),
        "--output",
        str(tmp_path / "pairs.csv"),
        "--push-to-hub",
    ]

    with pytest.raises(SystemExit):
        preference_generation.parse_args(common)


def test_push_to_hub_uses_requested_repository(monkeypatch, tmp_path):
    scores = tmp_path / "scores.csv"
    answers = tmp_path / "answers.csv"
    output = tmp_path / "pairs.csv"
    pd.DataFrame(
        {"model_a": [2.0], "model_b": [0.0]},
        index=pd.Index([0], name="Index"),
    ).to_csv(scores)
    pd.DataFrame(
        {"question": ["q0"], "model_a": ["a0"], "model_b": ["b0"]},
        index=pd.Index([0], name="index"),
    ).to_csv(answers)
    pushed = []

    class FakeDataset:
        @staticmethod
        def from_pandas(frame):
            return frame

    class FakeDatasetDict(dict):
        def push_to_hub(self, repo_id):
            pushed.append(repo_id)

    monkeypatch.setattr(preference_generation, "Dataset", FakeDataset)
    monkeypatch.setattr(preference_generation, "DatasetDict", FakeDatasetDict)

    preference_generation.generate_pairwise_comparisons(
        scores,
        answers,
        output,
        min_diff=0,
        shuffle=False,
        push_to_hub=True,
        hub_repo="example/preference-data",
    )

    assert pushed == ["example/preference-data"]
