from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_ROOTS = {
    "Benchmarking",
    "DPO",
    "RAG",
    "SFT",
    "rebuttal",
    "science-QA_jsonl",
    "scripts",
    "trust_llm",
}
FORBIDDEN_NAMES = {".DS_Store", ".idea", "__pycache__"}
FORBIDDEN_SUFFIXES = {".iml", ".pyc"}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
TEXT_SUFFIXES = {
    ".cfg", ".ini", ".ipynb", ".json", ".md", ".py", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}


def verify_repository(root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"forbidden root remains: {name}"
        for name in sorted(FORBIDDEN_ROOTS)
        if (root / name).exists()
    )
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/IDE artifact remains: {relative}")
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} in {relative}")
    return sorted(errors)


if __name__ == "__main__":
    violations = verify_repository(Path(__file__).resolve().parents[1])
    if violations:
        print("\n".join(violations), file=sys.stderr)
        raise SystemExit(1)
    print("Repository layout and tracked text are publication-clean.")
