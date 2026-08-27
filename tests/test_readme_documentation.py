from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools.verify_repository import verify_repository


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

REPO_PATH_ROOTS = (
    "data/",
    "docs/",
    "evaluation/",
    "human_study/",
    "training/",
    "tools/",
    "tests/",
)
REPO_TOP_LEVEL = frozenset(REPO_PATH_ROOTS)
REPO_FILES = frozenset({"LICENSE", ".env.example", ".env"})
SKIP_PATH_PREFIXES = (
    "http",
    "https",
    "sec:",
    "tab:",
    "fig:",
    "annotator_",
    "mattany/",
    "unsloth/",
    "meta-llama/",
    "dhmeltzer/",
    "/tmp/",
    "/path/",
    "/content/",
)
DOCUMENTED_OFFLINE_COMMANDS = (
    (
        "data/qa_pairs/create_split.py",
        ["python", "data/qa_pairs/create_split.py", "--help"],
    ),
    (
        "evaluation/rubrics/custom_metrics/aggregate_v2",
        ["python", "-m", "evaluation.rubrics.custom_metrics.aggregate_v2", "--help"],
    ),
    (
        "evaluation/rubrics/custom_metrics/winrate",
        ["python", "-m", "evaluation.rubrics.custom_metrics.winrate", "--help"],
    ),
    (
        "evaluation/factuality/truthfulqa_visualization.py",
        ["python", "evaluation/factuality/truthfulqa_visualization.py", "--help"],
    ),
    (
        "human_study/judge_validation/intercoder_reliability.py",
        [
            "python",
            "human_study/judge_validation/intercoder_reliability.py",
            "--help",
        ],
    ),
    (
        "human_study/preferences/teacher_significance.py",
        ["python", "human_study/preferences/teacher_significance.py"],
    ),
    (
        "human_study/preferences/logistic_regression.py",
        ["python", "human_study/preferences/logistic_regression.py", "--help"],
    ),
    (
        "human_study/preferences/metaphor_overoptimization.py",
        [
            "python",
            "human_study/preferences/metaphor_overoptimization.py",
            "--output",
            "/tmp/dpo_rubric_up_pref_down_test.csv",
        ],
    ),
)


def _clean_backtick(token: str) -> str:
    return token.strip().rstrip(".,;:!?)").lstrip("(")


def extract_readme_repo_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", text):
        candidate = _clean_backtick(match.group(1))
        if not candidate or "\n" in candidate:
            continue
        if candidate.startswith(SKIP_PATH_PREFIXES):
            continue
        if candidate in REPO_FILES or candidate in REPO_TOP_LEVEL:
            paths.add(candidate)
            continue
        if candidate.startswith(REPO_PATH_ROOTS):
            paths.add(candidate)
    return paths


def extract_readme_command_scripts(text: str) -> set[str]:
    scripts: set[str] = set()
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("python "):
                token = stripped.split()[1]
                if token == "-m":
                    continue
                scripts.add(token)
            elif stripped.startswith(("mkdir", "cp", "cd", "source", "poetry", "MPLBACKEND")):
                continue
            elif "/" in stripped and stripped.endswith(".py"):
                scripts.add(stripped.split()[0])
    return scripts


def export_git_archive(destination: Path) -> Path:
    archive_path = destination / "publication-export.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive_path), "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    extract_dir = destination / "export"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(path=extract_dir, filter="data")
    return extract_dir


def test_readme_repo_paths_exist():
    missing = sorted(
        path
        for path in extract_readme_repo_paths(README.read_text(encoding="utf-8"))
        if not (ROOT / path).exists()
    )
    assert missing == []


def test_readme_documents_safe_output_flags():
    text = README.read_text(encoding="utf-8")
    assert "--output-dir" in text
    assert "--bootstrap-dir" in text
    assert "--output" in text
    assert "Safe reruns" in text
    assert "evaluation/results/rubric_scores/.checkpoints/" in text
    assert "evaluation/results/preference_metrics/.checkpoints/" in text
    assert "human_study/preferences/data/first_exp.csv" in text
    assert "evaluation/rubrics/custom_metrics/winrate.py" in text
    assert "human_study/preferences/teacher_significance.py" in text


def test_readme_bash_blocks_reference_existing_scripts():
    text = README.read_text(encoding="utf-8")
    missing = sorted(
        script
        for script in extract_readme_command_scripts(text)
        if not (ROOT / script).exists()
    )
    assert missing == []


@pytest.mark.parametrize("label,command", DOCUMENTED_OFFLINE_COMMANDS)
def test_documented_offline_commands_run(label, command):
    del label
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_git_archive_passes_publication_verifier(tmp_path):
    export_root = export_git_archive(tmp_path)
    violations = verify_repository(export_root)
    assert violations == []
