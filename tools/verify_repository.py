from __future__ import annotations

import re
import sys
from pathlib import Path


ALLOWED_TOP_LEVEL = {
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
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".idea",
    "__pycache__",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
FORBIDDEN_SUFFIXES = {".iml", ".pyc", ".pyo", ".pyd"}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "xAI API key": re.compile(r"\bxai-[A-Za-z0-9_-]{20,}"),
    "LangChain API key": re.compile(r"\blsv2_[A-Za-z0-9_:-]{20,}"),
    "Reddit client assignment": re.compile(
        r"(?i)(?:client_id|client_secret)\s*=\s*[\"']([A-Za-z0-9_-]{16,})[\"']"
    ),
}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXAMPLE_VALUE_MARKERS = (
    "example",
    "your_",
    "changeme",
    "placeholder",
    "replace",
    "insert",
    "optional",
    "xxx",
    "***",
)
ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*)$")
MAX_SCAN_BYTES = 1_000_000


def verify_repository(root: Path) -> list[str]:
    errors: list[str] = []
    root = root.resolve()

    for relative_path in REQUIRED_PUBLIC_PATHS:
        if not (root / relative_path).exists():
            errors.append(f"required public path missing: {relative_path}")

    for entry in root.iterdir():
        if entry.name == ".git":
            continue
        if entry.name not in ALLOWED_TOP_LEVEL:
            errors.append(f"disallowed top-level entry: {entry.name}")

    errors.extend(
        f"forbidden root remains: {name}"
        for name in sorted(FORBIDDEN_ROOTS)
        if (root / name).exists()
    )

    for path in _iter_repository_paths(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.extend(_check_symlink(path, relative, root))
            continue
        if path.name == ".env":
            errors.append(f"credential file remains: {relative}")
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/IDE artifact remains: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name == ".env.example":
            text = _read_text_safely(path)
            if text is not None:
                errors.extend(_scan_secrets(text, relative))
                errors.extend(_validate_env_example(text, relative))
            continue
        if _should_scan_file(path):
            text = _read_text_safely(path)
            if text is not None:
                errors.extend(_scan_secrets(text, relative))

    return sorted(errors)


def _iter_repository_paths(root: Path):
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        yield path


def _check_symlink(path: Path, relative: Path, root: Path) -> list[str]:
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except ValueError:
        return [f"symlink escapes repository: {relative}"]
    return [f"symlink remains: {relative}"]


def _should_scan_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    if suffix == "":
        return True
    return False


def _read_text_safely(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_SCAN_BYTES:
        return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="ignore")


def _scan_secrets(text: str, relative: Path) -> list[str]:
    errors: list[str] = []
    for label, pattern in SECRET_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if label == "Reddit client assignment":
            value = match.group(1)
            if _is_placeholder_value(value):
                continue
        errors.append(f"{label} in {relative}")
    return errors


def _validate_env_example(text: str, relative: Path) -> list[str]:
    errors: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGNMENT.match(stripped)
        if not match:
            continue
        value = match.group(2).strip().strip('"').strip("'")
        if not value:
            continue
        if _is_placeholder_value(value):
            continue
        errors.append(f"non-example value in {relative}")
    return errors


def _is_placeholder_value(value: str) -> bool:
    if value.startswith("<") and value.endswith(">"):
        return True
    lowered = value.lower()
    if lowered in {"changeme", "example", "placeholder", "optional", "***"}:
        return True
    return any(
        marker in lowered
        for marker in EXAMPLE_VALUE_MARKERS
        if marker not in {"xxx", "***", "example", "changeme", "placeholder", "optional"}
    )


if __name__ == "__main__":
    violations = verify_repository(Path(__file__).resolve().parents[1])
    if violations:
        print("\n".join(violations), file=sys.stderr)
        raise SystemExit(1)
    print("Repository layout and tracked text are publication-clean.")
