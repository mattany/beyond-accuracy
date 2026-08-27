from __future__ import annotations

import importlib
import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _entrypoint_stubs() -> str:
    return """
import contextlib
import sys
from types import ModuleType, SimpleNamespace

class Dummy:
    def __init__(self, *args, **kwargs):
        pass

deepeval = ModuleType("deepeval")
deepeval.__path__ = []
deepeval_metrics = ModuleType("deepeval.metrics")
deepeval_metrics.GEval = Dummy
deepeval_models = ModuleType("deepeval.models")
deepeval_models.DeepEvalBaseLLM = object
deepeval_test_case = ModuleType("deepeval.test_case")
deepeval_test_case.LLMTestCase = Dummy
sys.modules.update({
    "deepeval": deepeval,
    "deepeval.metrics": deepeval_metrics,
    "deepeval.models": deepeval_models,
    "deepeval.test_case": deepeval_test_case,
})

metric_module = ModuleType("evaluation.rubrics.custom_metrics.metrics")
for name in (
    "scaffolding_metric_v2", "metaphor_metric_explicit_v8",
    "humor_metric_explicit_v5", "analogy_metric_explicit_v2",
    "jargon_metric", "flesch_kincaid", "flesch_reading_ease",
    "dale_chall", "ari",
):
    setattr(metric_module, name, Dummy())
sys.modules["evaluation.rubrics.custom_metrics.metrics"] = metric_module

ollama_model = ModuleType("evaluation.rubrics.ollama_model")
ollama_model.OllamaModel = Dummy
sys.modules["evaluation.rubrics.ollama_model"] = ollama_model
prompt_templates = ModuleType("evaluation.rubrics.prompt_templates")
prompt_templates.generate_prompt = lambda value: value
prompt_templates.system_prompt = "system"
sys.modules["evaluation.rubrics.prompt_templates"] = prompt_templates

transformers = ModuleType("transformers")
transformers.BitsAndBytesConfig = Dummy
transformers.AutoModelForCausalLM = Dummy
transformers.AutoTokenizer = Dummy
sys.modules["transformers"] = transformers
torch = ModuleType("torch")
torch.cuda = SimpleNamespace(is_available=lambda: False)
torch.float16 = object()
torch.inference_mode = contextlib.nullcontext
sys.modules["torch"] = torch
"""


def _run_entrypoint(tmp_path: Path, target: str, module: str, mode: str) -> subprocess.CompletedProcess[str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "sitecustomize.py").write_text(_entrypoint_stubs(), encoding="utf-8")
    env = os.environ.copy()
    paths = [str(stubs)]
    if mode == "module":
        paths.append(str(ROOT))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop("OPENAI_API_KEY", None)

    if mode == "module":
        command = [sys.executable, "-m", module]
    else:
        command = [sys.executable, str(ROOT / target)]
    if target.endswith(("run.py", "generate_kto_llama3_1_outputs.py")):
        command.append("--help")
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


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


@pytest.mark.parametrize("mode", ["module", "direct"])
@pytest.mark.parametrize(
    ("target", "module"),
    [
        (
            "evaluation/rubrics/custom_metrics/run.py",
            "evaluation.rubrics.custom_metrics.run",
        ),
        (
            "evaluation/rubrics/generate_kto_llama3_1_outputs.py",
            "evaluation.rubrics.generate_kto_llama3_1_outputs",
        ),
        (
            "evaluation/rubrics/models/llama_2_model.py",
            "evaluation.rubrics.models.llama_2_model",
        ),
    ],
)
def test_evaluator_entrypoints_support_module_and_direct_modes(
    tmp_path, target, module, mode
):
    result = _run_entrypoint(tmp_path, target, module, mode)
    assert result.returncode == 0, result.stderr


def test_declared_component_readmes_exist():
    pyprojects = [
        ROOT / "evaluation/rubrics/pyproject.toml",
        ROOT / "evaluation/model_generation/pyproject.toml",
        ROOT / "evaluation/factuality/trust_llm/pyproject.toml",
    ]

    for pyproject in pyprojects:
        readme_line = next(
            line for line in pyproject.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("readme =")
        )
        readme = readme_line.split("=", maxsplit=1)[1].strip().strip('"')
        assert (pyproject.parent / readme).is_file()


def test_trust_llm_heatmap_runs_outside_component_directory(tmp_path):
    script = ROOT / "evaluation/factuality/trust_llm/heatmap.py"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_model_dataset_generator_import_is_credential_free(monkeypatch):
    anthropic = ModuleType("langchain_anthropic")
    anthropic.ChatAnthropic = lambda *args, **kwargs: object()
    prompts = ModuleType("langchain_core.prompts")
    prompts.ChatPromptTemplate = object()
    monkeypatch.setitem(sys.modules, "langchain_anthropic", anthropic)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    from evaluation.rubrics import settings

    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(settings, "LANGCHAIN_API_KEY", None)

    namespace = runpy.run_path(
        str(
            ROOT
            / "evaluation/rubrics/evaluation_dataset_generation/"
            "add_model_to_eval_dataset.py"
        ),
        run_name="credential_free_import",
    )

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        namespace["main"]()


def test_component_docs_use_current_paths():
    truthfulqa = (
        ROOT / "evaluation/factuality/truthfulqa_visualization.py"
    ).read_text(encoding="utf-8")
    generation = (
        ROOT / "evaluation/rubrics/evaluation_dataset_generation/README.md"
    ).read_text(encoding="utf-8")

    assert "default: evaluation/factuality/truthfulqa_results/" in truthfulqa
    assert "data/test_data" not in generation
    assert "evaluation/model_outputs/main" in generation


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    [
        (
            "evaluation/model_outputs/main/all_models_joined.csv",
            "548146d41e6547e122a363798b480333d194fb3ea92817172f2df94e8d30c8b2",
        ),
        (
            "evaluation/model_outputs/scillama3/output.csv",
            "0825bf67e0cedf5a145f7b4c680c495620e0d3fd83fea7d61687ba56c79b71f4",
        ),
    ],
)
def test_canonical_output_blob_identity(relative_path, expected_sha256):
    digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    assert digest == expected_sha256


def test_bootstrap_module_imports_package_safe():
    from evaluation.rubrics.custom_metrics.bootstrap import (
        bootstrap_analysis,
        bootstrap_analysis_v2,
    )

    assert callable(bootstrap_analysis)
    assert callable(bootstrap_analysis_v2)


def test_scillama3_processing_import_is_side_effect_free():
    namespace = runpy.run_path(
        str(ROOT / "evaluation/model_outputs/scillama3/processing.py"),
        run_name="scillama3_processing_import",
    )
    assert callable(namespace["process_file"])
    assert callable(namespace["main"])


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
