from __future__ import annotations

import os
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
REQUIRED_STRUCTURE: tuple[tuple[str, str], ...] = (
    ("README.md", "file"),
    ("LICENSE", "file"),
    (".env.example", "file"),
    (".gitignore", "file"),
    ("data", "dir"),
    ("data/qa_pairs", "dir"),
    ("docs", "dir"),
    ("docs/artifacts.md", "file"),
    ("evaluation", "dir"),
    ("evaluation/factuality", "dir"),
    ("evaluation/model_generation", "dir"),
    ("evaluation/model_outputs", "dir"),
    ("evaluation/results/rubric_scores", "dir"),
    ("evaluation/results/preference_metrics", "dir"),
    ("evaluation/rubrics", "dir"),
    ("evaluation/visualization", "dir"),
    ("human_study", "dir"),
    ("human_study/judge_validation", "dir"),
    ("human_study/preferences", "dir"),
    ("training", "dir"),
    ("training/data_generation", "dir"),
    ("training/dpo", "dir"),
    ("training/model_variants", "dir"),
    ("training/sft", "dir"),
    ("tools", "dir"),
    ("tools/verify_repository.py", "file"),
    ("tests", "dir"),
    ("tests/test_public_repository.py", "file"),
)
REQUIRED_PUBLIC_PATHS = [relative_path for relative_path, _ in REQUIRED_STRUCTURE]
PLANNED_ENV_EXAMPLE = """# Copy to .env and set only the providers you use.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
MOONSHOT_API_KEY=
HF_TOKEN=
LANGCHAIN_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=beyond-accuracy research script
"""
ALLOWED_ENV_EXAMPLE_VALUES = {
    "REDDIT_USER_AGENT": frozenset({"beyond-accuracy research script"}),
}
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
SCAN_CHUNK_SIZE = 65_536
SCAN_OVERLAP = 256
BINARY_SAMPLE_SIZE = 8192


def verify_repository(root: Path) -> list[str]:
    errors: list[str] = []
    root = root.resolve()

    errors.extend(_check_required_structure(root))

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
            errors.extend(_scan_env_example(path, relative))
            continue
        if _should_scan_file(path):
            errors.extend(_scan_file(path, relative))

    return sorted(errors)


def _check_required_structure(root: Path) -> list[str]:
    errors: list[str] = []
    for relative_path, kind in REQUIRED_STRUCTURE:
        path = root / relative_path
        if not path.exists():
            errors.append(f"required public path missing: {relative_path}")
            continue
        if kind == "file":
            if not path.is_file():
                errors.append(
                    f"required public path has wrong type: {relative_path} (expected file)"
                )
        elif not path.is_dir():
            errors.append(
                f"required public path has wrong type: {relative_path} (expected directory)"
            )
    return errors


def _iter_repository_paths(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        if ".git" in current.parts:
            dirnames.clear()
            continue
        if current == root:
            dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in dirnames:
            yield current / name
        for name in filenames:
            yield current / name


def _check_symlink(path: Path, relative: Path, root: Path) -> list[str]:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, RecursionError):
        return [f"symlink requires manual review: {relative}"]
    try:
        resolved.relative_to(root)
    except ValueError:
        return [f"symlink escapes repository: {relative}"]
    return [f"symlink remains: {relative}"]


def _should_scan_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return suffix == ""


def _scan_env_example(path: Path, relative: Path) -> list[str]:
    text, read_errors = _read_text_for_scan(path, relative)
    if read_errors:
        return read_errors
    if text is None:
        return [f"unreadable file requires manual review: {relative}"]
    errors = _scan_secrets(text, relative)
    errors.extend(_validate_env_example(text, relative))
    return errors


def _scan_file(path: Path, relative: Path) -> list[str]:
    try:
        if _looks_binary(path):
            return []
    except OSError:
        return [f"unreadable file requires manual review: {relative}"]
    return _stream_scan_secrets(path, relative)


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        sample = handle.read(BINARY_SAMPLE_SIZE)
    if not sample:
        return False
    return b"\x00" in sample


def _read_text_for_scan(path: Path, relative: Path) -> tuple[str | None, list[str]]:
    try:
        with path.open("rb") as handle:
            data = handle.read()
    except OSError:
        return None, [f"unreadable file requires manual review: {relative}"]
    if b"\x00" in data[:BINARY_SAMPLE_SIZE]:
        return None, []
    return data.decode("utf-8", errors="ignore"), []


def _stream_scan_secrets(path: Path, relative: Path) -> list[str]:
    errors: list[str] = []
    found_labels: set[str] = set()
    carry = b""
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(SCAN_CHUNK_SIZE)
                if not chunk:
                    break
                combined = carry + chunk
                text = combined.decode("utf-8", errors="ignore")
                if len(chunk) == SCAN_CHUNK_SIZE:
                    carry = combined[-SCAN_OVERLAP:]
                else:
                    carry = b""
                for label, pattern in SECRET_PATTERNS.items():
                    if label in found_labels:
                        continue
                    match = pattern.search(text)
                    if not match:
                        continue
                    if label == "Reddit client assignment" and _is_placeholder_value(
                        match.group(1)
                    ):
                        continue
                    errors.append(f"{label} in {relative}")
                    found_labels.add(label)
    except OSError:
        return [f"unreadable file requires manual review: {relative}"]
    return errors


def _scan_secrets(text: str, relative: Path) -> list[str]:
    errors: list[str] = []
    for label, pattern in SECRET_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if label == "Reddit client assignment" and _is_placeholder_value(match.group(1)):
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
        variable = match.group(1)
        value = match.group(2).strip().strip('"').strip("'")
        if not value:
            continue
        if variable in ALLOWED_ENV_EXAMPLE_VALUES and value in ALLOWED_ENV_EXAMPLE_VALUES[variable]:
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
