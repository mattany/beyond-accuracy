from pathlib import Path

import pytest

from tools.verify_repository import verify_repository


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PUBLIC_PATHS = [
    "README.md",
    "LICENSE",
    ".env.example",
    "docs/artifacts.md",
    "training",
    "evaluation",
    "human_study",
    "data/qa_pairs",
]


def test_clean_repository_has_no_layout_violations(tmp_path):
    (tmp_path / "README.md").write_text("# publication repo\n", encoding="utf-8")
    assert verify_repository(tmp_path) == []


def test_forbidden_root_is_reported(tmp_path):
    (tmp_path / "Benchmarking").mkdir()
    errors = verify_repository(tmp_path)
    assert "forbidden root remains: Benchmarking" in errors


def test_generated_artifact_is_reported(tmp_path):
    (tmp_path / "thesis.iml").write_text("", encoding="utf-8")
    errors = verify_repository(tmp_path)
    assert "generated/IDE artifact remains: thesis.iml" in errors


def test_secret_pattern_is_reported_in_text_files(tmp_path):
    secret = "sk-" + "x" * 24
    (tmp_path / "config.py").write_text(
        f'API_KEY = "{secret}"\n',
        encoding="utf-8",
    )
    errors = verify_repository(tmp_path)
    assert any(error.startswith("OpenAI-style key in config.py") for error in errors)


def test_live_repository_still_has_layout_violations_before_cleanup():
    assert verify_repository(ROOT) != []


def test_required_public_files_still_missing_before_cleanup():
    missing = [path for path in REQUIRED_PUBLIC_PATHS if not (ROOT / path).exists()]
    assert missing == [
        "README.md",
        "LICENSE",
        ".env.example",
        "training",
        "evaluation",
        "human_study",
        "data/qa_pairs",
    ]


@pytest.mark.parametrize("relative_path", REQUIRED_PUBLIC_PATHS)
def test_required_public_paths_eventually_exist(relative_path):
    pytest.skip("Passes after Tasks 3-7 create the publication layout.")
