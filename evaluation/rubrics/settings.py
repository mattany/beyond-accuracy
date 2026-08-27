from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def require_env(name: str, value: str | None) -> str:
    if value is None:
        raise RuntimeError(
            f"{name} is required for this API-backed operation; "
            f"export {name} before running it."
        )
    return value


def result_directory(run_number: int) -> Path:
    directories = {
        9: PROJECT_ROOT / "evaluation/results/rubric_scores",
        10: PROJECT_ROOT / "evaluation/results/preference_metrics",
    }
    try:
        return directories[run_number]
    except KeyError as error:
        raise ValueError(f"Unknown canonical result run: {run_number}") from error


OPENAI_API_KEY = optional_env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = optional_env("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = optional_env("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = optional_env("GOOGLE_API_KEY")
XAI_API_KEY = optional_env("XAI_API_KEY")
MOONSHOT_API_KEY = optional_env("MOONSHOT_API_KEY")
HF_TOKEN = optional_env("HF_TOKEN")
LANGCHAIN_API_KEY = optional_env("LANGCHAIN_API_KEY")
