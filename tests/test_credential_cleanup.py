from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARENT_REV = "a2c1bdb"
NOTEBOOK_REGRESSION_PATHS = (
    "evaluation/model_generation/llama3_1/llama3_evaluation.ipynb",
    "training/sft/GPT_SFT_only.ipynb",
    "evaluation/factuality/truthfulqa_benchmark.ipynb",
)
HF_TOUCHING_SCRIPTS = (
    "evaluation/factuality/truthfulqa_benchmark.py",
    "training/model_variants/upload_organic_sft.py",
    "training/model_variants/upload_organic_dpo.py",
)
AUTH_WARNING = re.compile(
    r"The secret `HF_TOKEN`|"
    r"authentication is recommended but still optional to access public models",
    re.IGNORECASE,
)
AUTH_NOTEBOOK_PATTERNS = (
    re.compile(r"\bnotebook_login\s*\("),
    re.compile(r"hugging_face_key\s*="),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"PasswordModel"),
    re.compile(r"The secret `HF_TOKEN`"),
)
AUTH_SCRIPT_PATTERNS = (re.compile(r"\bnotebook_login\s*\("),)
SCOPED_AUTH_CELLS = {
    "evaluation/model_generation/llama3_1/llama3_evaluation.ipynb": {3, 4},
    "training/sft/GPT_SFT_only.ipynb": {4, 6},
    "evaluation/factuality/truthfulqa_benchmark.ipynb": {2},
}
TRUTHFULQA_SETUP_CELL = 2
TRUTHFULQA_SETUP_MARKERS = (
    "drive.mount('/content/drive/', force_remount=True)",
    "os.environ['HF_HOME']",
    "OUTPUT_DIR = '/content/drive/MyDrive/thesis/truthfulqa_results'",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)",
    "import pandas as pd",
    "import numpy as np",
    "from datetime import datetime",
    "from tqdm import tqdm",
    'login(token=os.environ["HF_TOKEN"])',
)


def tracked_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def load_parent_notebook(relative_path: str) -> dict:
    raw = subprocess.check_output(
        ["git", "show", f"{PARENT_REV}:{relative_path}"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(raw)


def notebook_blob(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def scientific_output_count(nb: dict, auth_cells: set[int]) -> int:
    return sum(
        len(cell.get("outputs", []))
        for index, cell in enumerate(nb["cells"])
        if index not in auth_cells
    )


@pytest.mark.parametrize("relative_path", NOTEBOOK_REGRESSION_PATHS)
def test_scoped_notebooks_are_valid_json(relative_path):
    json.loads(notebook_blob(relative_path))


@pytest.mark.parametrize("relative_path", NOTEBOOK_REGRESSION_PATHS)
def test_scoped_notebooks_have_no_auth_patterns(relative_path):
    blob = notebook_blob(relative_path)
    for pattern in AUTH_NOTEBOOK_PATTERNS:
        assert not pattern.search(blob), pattern.pattern


@pytest.mark.parametrize("relative_path", NOTEBOOK_REGRESSION_PATHS)
def test_scoped_auth_cells_have_no_outputs_or_widget_metadata(relative_path):
    nb = json.loads(notebook_blob(relative_path))
    auth_cells = SCOPED_AUTH_CELLS[relative_path]
    for index in auth_cells:
        cell = nb["cells"][index]
        assert cell.get("outputs") == []
        assert "outputId" not in cell.get("metadata", {})
    widget_state = (
        nb.get("metadata", {})
        .get("widgets", {})
        .get("application/vnd.jupyter.widget-state+json", {})
    )
    if isinstance(widget_state, dict):
        assert not any(
            "PasswordModel" in json.dumps(value)
            for value in widget_state.values()
            if isinstance(value, dict)
        )


def test_tracked_tree_has_no_notebook_login():
    matches = []
    for relative_path in tracked_files():
        if not relative_path.endswith((".py", ".ipynb")):
            continue
        text = (ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")
        if AUTH_SCRIPT_PATTERNS[0].search(text):
            matches.append(relative_path)
    assert matches == []


def auth_warning_output_count(nb: dict, auth_cells: set[int]) -> int:
    count = 0
    for index, cell in enumerate(nb["cells"]):
        if index in auth_cells:
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = output.get("text", "")
            blob = "".join(text) if isinstance(text, list) else str(text)
            if AUTH_WARNING.search(blob):
                count += 1
    return count


@pytest.mark.parametrize("relative_path", NOTEBOOK_REGRESSION_PATHS)
def test_scoped_notebook_diff_is_auth_limited(relative_path):
    parent = load_parent_notebook(relative_path)
    current = json.loads(notebook_blob(relative_path))
    auth_cells = SCOPED_AUTH_CELLS[relative_path]

    assert len(parent["cells"]) == len(current["cells"])
    removed_warnings = auth_warning_output_count(parent, auth_cells)
    assert scientific_output_count(current, auth_cells) == (
        scientific_output_count(parent, auth_cells) - removed_warnings
    )

    diff_stat = subprocess.check_output(
        ["git", "diff", "--numstat", PARENT_REV, "--", relative_path],
        cwd=ROOT,
        text=True,
    ).strip()
    added, deleted, _ = diff_stat.split()
    line_delta = int(added) + int(deleted)
    limit = 1200 if "truthfulqa_benchmark.ipynb" not in relative_path else 80
    assert line_delta < limit, diff_stat


def test_truthfulqa_setup_cell_preserves_required_definitions():
    nb = json.loads(notebook_blob("evaluation/factuality/truthfulqa_benchmark.ipynb"))
    setup_source = "".join(nb["cells"][TRUTHFULQA_SETUP_CELL]["source"])
    for marker in TRUTHFULQA_SETUP_MARKERS:
        assert marker in setup_source
    parent = load_parent_notebook("evaluation/factuality/truthfulqa_benchmark.ipynb")
    for index in range(len(parent["cells"])):
        if index == TRUTHFULQA_SETUP_CELL:
            continue
        assert parent["cells"][index] == nb["cells"][index]


@pytest.mark.parametrize("relative_path", HF_TOUCHING_SCRIPTS)
def test_hf_scripts_use_env_login(relative_path):
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    assert 'login(token=os.environ["HF_TOKEN"])' in text
    assert "notebook_login" not in text


def extract_login_source(relative_path: str) -> str:
    lines = (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    chunk: list[str] = []
    capture = False
    for line in lines:
        if line.strip().startswith("# Login to HuggingFace"):
            capture = True
        if capture:
            chunk.append(line)
            if line.strip().startswith("login(token="):
                break
    assert chunk, f"missing login block in {relative_path}"
    return "\n".join(chunk) + "\n"


@pytest.mark.parametrize("relative_path", HF_TOUCHING_SCRIPTS)
def test_hf_scripts_do_not_contact_hub_at_import(relative_path, monkeypatch):
    calls: list[str] = []

    hub = ModuleType("huggingface_hub")

    def fake_login(*, token: str) -> None:
        calls.append(token)

    hub.login = fake_login
    hub.HfApi = lambda *args, **kwargs: pytest.fail("HfApi instantiated during login")
    hub.create_repo = lambda *args, **kwargs: pytest.fail("create_repo called during login")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_TOKEN", "example-hf-token")

    namespace = {"os": __import__("os")}
    exec(extract_login_source(relative_path), namespace)
    assert calls == ["example-hf-token"]


def test_realtime_gen_error_points_to_env_example():
    text = (ROOT / "training/data_generation/realtime_gen.py").read_text(encoding="utf-8")
    assert ".env.example" in text
    assert "training/data_generation/config.py" not in text
