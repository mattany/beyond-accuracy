from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.verify_repository import (
    ALLOWED_TOP_LEVEL,
    PLANNED_ENV_EXAMPLE,
    REQUIRED_PUBLIC_PATHS,
    REQUIRED_STRUCTURE,
    verify_repository,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = [path for path, kind in REQUIRED_STRUCTURE if kind == "dir"]
REQUIRED_FILES = [path for path, kind in REQUIRED_STRUCTURE if kind == "file"]


def build_valid_public_tree(root: Path) -> None:
    for relative_path, kind in REQUIRED_STRUCTURE:
        path = root / relative_path
        if kind == "dir":
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == ".env.example":
            path.write_text(PLANNED_ENV_EXAMPLE, encoding="utf-8")
        elif relative_path == "tools/verify_repository.py":
            shutil.copy2(ROOT / "tools" / "verify_repository.py", path)
        elif relative_path == "tests/test_public_repository.py":
            path.write_text("# contract tests\n", encoding="utf-8")
        elif relative_path == "README.md":
            path.write_text("# Beyond Accuracy\n", encoding="utf-8")
        elif relative_path == "LICENSE":
            path.write_text("MIT License\n", encoding="utf-8")
        elif relative_path == ".gitignore":
            path.write_text(".env\n__pycache__/\n", encoding="utf-8")
        elif relative_path == "docs/artifacts.md":
            path.write_text("# Paper artifact map\n", encoding="utf-8")
        else:
            path.write_text(f"# placeholder for {relative_path}\n", encoding="utf-8")


def run_verifier_cli(tree_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/verify_repository.py"],
        cwd=tree_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_public_tree_passes_verification(tmp_path):
    build_valid_public_tree(tmp_path)
    assert verify_repository(tmp_path) == []


def test_planned_env_example_including_reddit_user_agent_passes(tmp_path):
    build_valid_public_tree(tmp_path)
    env_example = (tmp_path / ".env.example").read_text(encoding="utf-8")
    assert "REDDIT_USER_AGENT=beyond-accuracy research script" in env_example
    assert verify_repository(tmp_path) == []


def test_sparse_tree_is_rejected(tmp_path):
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    (tmp_path / "legacy").mkdir()
    errors = verify_repository(tmp_path)
    assert any(error.startswith("required public path missing:") for error in errors)
    assert "disallowed top-level entry: legacy" in errors


@pytest.mark.parametrize("relative_path", REQUIRED_PUBLIC_PATHS)
def test_missing_required_public_path_is_reported(tmp_path, relative_path):
    build_valid_public_tree(tmp_path)
    target = tmp_path / relative_path
    if target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)
    errors = verify_repository(tmp_path)
    assert f"required public path missing: {relative_path}" in errors


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_wrong_type_is_reported(tmp_path, relative_path):
    build_valid_public_tree(tmp_path)
    target = tmp_path / relative_path
    target.unlink()
    target.mkdir()
    errors = verify_repository(tmp_path)
    assert f"required public path has wrong type: {relative_path} (expected file)" in errors


@pytest.mark.parametrize("relative_path", REQUIRED_DIRECTORIES)
def test_required_directory_wrong_type_is_reported(tmp_path, relative_path):
    build_valid_public_tree(tmp_path)
    target = tmp_path / relative_path
    shutil.rmtree(target)
    target.write_text("not a directory\n", encoding="utf-8")
    errors = verify_repository(tmp_path)
    assert (
        f"required public path has wrong type: {relative_path} (expected directory)"
        in errors
    )


def test_disallowed_top_level_entry_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    (tmp_path / "scratch").mkdir()
    errors = verify_repository(tmp_path)
    assert "disallowed top-level entry: scratch" in errors


def test_forbidden_legacy_root_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    (tmp_path / "Benchmarking").mkdir()
    errors = verify_repository(tmp_path)
    assert "forbidden root remains: Benchmarking" in errors
    assert "disallowed top-level entry: Benchmarking" in errors


@pytest.mark.parametrize(
    "artifact_name",
    [
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        ".idea",
    ],
)
def test_generated_directory_artifact_is_reported(tmp_path, artifact_name):
    build_valid_public_tree(tmp_path)
    (tmp_path / artifact_name).mkdir()
    errors = verify_repository(tmp_path)
    assert f"generated/IDE artifact remains: {artifact_name}" in errors


@pytest.mark.parametrize("suffix", [".iml", ".pyc", ".pyo", ".pyd"])
def test_generated_file_artifact_is_reported(tmp_path, suffix):
    build_valid_public_tree(tmp_path)
    artifact = tmp_path / f"module{suffix}"
    artifact.write_bytes(b"")
    errors = verify_repository(tmp_path)
    assert f"generated/IDE artifact remains: module{suffix}" in errors


def test_env_file_is_rejected(tmp_path):
    build_valid_public_tree(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    errors = verify_repository(tmp_path)
    assert "credential file remains: .env" in errors


def test_env_example_with_non_example_value_is_rejected(tmp_path):
    build_valid_public_tree(tmp_path)
    secret_value = "x" * 24
    (tmp_path / ".env.example").write_text(
        f"OPENAI_API_KEY={secret_value}\n",
        encoding="utf-8",
    )
    errors = verify_repository(tmp_path)
    assert "non-example value in .env.example" in errors


def test_secret_in_csv_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    secret = "sk-" + "x" * 24
    (tmp_path / "data" / "qa_pairs" / "tokens.csv").write_text(
        f"model,token\nllama,{secret}\n",
        encoding="utf-8",
    )
    errors = verify_repository(tmp_path)
    assert any(
        error.startswith("OpenAI-style key in data/qa_pairs/tokens.csv") for error in errors
    )


def test_secret_near_end_of_large_file_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    secret = "sk-" + "y" * 24
    large_path = tmp_path / "evaluation" / "large.csv"
    with large_path.open("wb") as handle:
        handle.write(b"x" * (1024 * 1024 + 1))
        handle.write(b"\n")
        handle.write(secret.encode("ascii"))
    errors = verify_repository(tmp_path)
    assert any(error.startswith("OpenAI-style key in evaluation/large.csv") for error in errors)


def test_unreadable_file_requires_manual_review(tmp_path):
    build_valid_public_tree(tmp_path)
    target = tmp_path / "evaluation" / "locked.txt"
    target.write_text("safe\n", encoding="utf-8")
    target.chmod(0o000)
    try:
        errors = verify_repository(tmp_path)
        assert "unreadable file requires manual review: evaluation/locked.txt" in errors
    finally:
        target.chmod(0o644)


def test_secret_in_extensionless_text_file_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    secret = "hf_" + "a" * 24
    (tmp_path / "credentials").write_text(f"{secret}\n", encoding="utf-8")
    errors = verify_repository(tmp_path)
    assert any(error.startswith("Hugging Face token in credentials") for error in errors)


def test_xai_secret_pattern_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    secret = "xai-" + "b" * 24
    (tmp_path / "notes.txt").write_text(f"key={secret}\n", encoding="utf-8")
    errors = verify_repository(tmp_path)
    assert any(error.startswith("xAI API key in notes.txt") for error in errors)


def test_langchain_secret_pattern_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    secret = "lsv2_" + "c" * 24
    (tmp_path / "evaluation" / "config.json").write_text(
        f'{{"key": "{secret}"}}',
        encoding="utf-8",
    )
    errors = verify_repository(tmp_path)
    assert any(error.startswith("LangChain API key in evaluation/config.json") for error in errors)


def test_reddit_client_assignment_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    assignment = "d" * 20
    (tmp_path / "reddit.cfg").write_text(
        f'client_secret = "{assignment}"\n',
        encoding="utf-8",
    )
    errors = verify_repository(tmp_path)
    assert any(error.startswith("Reddit client assignment in reddit.cfg") for error in errors)


def test_symlink_is_rejected_without_reading_target(tmp_path):
    build_valid_public_tree(tmp_path)
    outside = tmp_path.parent / "outside-secret-tree"
    outside.mkdir(exist_ok=True)
    secret_file = outside / "secret.txt"
    secret_file.write_text("hf_" + "z" * 24 + "\n", encoding="utf-8")
    link = tmp_path / "external-link"
    link.symlink_to(secret_file)
    errors = verify_repository(tmp_path)
    assert any(error.startswith("symlink escapes repository: external-link") for error in errors)
    assert not any("Hugging Face token" in error for error in errors)


def test_symlink_escape_is_reported(tmp_path):
    build_valid_public_tree(tmp_path)
    outside = tmp_path.parent / "outside-escape-tree"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "escape-link"
    link.symlink_to(outside)
    errors = verify_repository(tmp_path)
    assert any(error.startswith("symlink escapes repository: escape-link") for error in errors)


def test_symlink_loop_requires_manual_review(tmp_path):
    build_valid_public_tree(tmp_path)
    loop_a = tmp_path / "training" / "loop-a"
    loop_b = tmp_path / "training" / "loop-b"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    errors = verify_repository(tmp_path)
    assert any(
        error.startswith("symlink requires manual review: training/loop-a")
        or error.startswith("symlink requires manual review: training/loop-b")
        for error in errors
    )


def test_allowed_top_level_matches_publication_contract():
    assert ALLOWED_TOP_LEVEL == {
        "README.md",
        "LICENSE",
        ".env.example",
        ".gitignore",
        "data",
        "docs",
        "evaluation",
        "human_study",
        "tests",
        "tools",
        "training",
    }


def test_isolated_cli_passes_on_valid_tree(tmp_path):
    build_valid_public_tree(tmp_path)
    result = run_verifier_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "publication-clean" in result.stdout


def test_isolated_cli_fails_on_invalid_tree(tmp_path):
    (tmp_path / "README.md").write_text("# incomplete\n", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "verify_repository.py", tmp_path / "tools" / "verify_repository.py")
    result = run_verifier_cli(tmp_path)
    assert result.returncode == 1
    combined = result.stderr + result.stdout
    assert "required public path missing:" in combined
    assert not re.search(r"\bhf_[A-Za-z0-9]{20,}", combined)


def test_live_repository_has_violations_without_leaking_secrets():
    violations = verify_repository(ROOT)
    assert violations
    output = "\n".join(violations)
    assert not re.search(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}", output)
    assert not re.search(r"\bhf_[A-Za-z0-9]{20,}", output)
    assert not re.search(r"\bAIza[A-Za-z0-9_-]{30,}", output)
    assert not re.search(r"\bxai-[A-Za-z0-9_-]{20,}", output)
    assert not re.search(r"\blsv2_[A-Za-z0-9_:-]{20,}", output)
