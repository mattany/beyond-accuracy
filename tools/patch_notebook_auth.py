from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

AUTH_WARNING = re.compile(
    r"The secret `HF_TOKEN`|"
    r"authentication is recommended but still optional to access public models",
    re.IGNORECASE,
)
AUTH_WIDGET_MODELS = frozenset(
    {
        "PasswordModel",
        "ButtonModel",
        "CheckboxModel",
    }
)
IPY_MODEL_REF = re.compile(r"IPY_MODEL_([0-9a-f]+)")
HF_ENV_LOGIN = (
    'import os\n'
    'from huggingface_hub import login\n\n'
    'login(token=os.environ["HF_TOKEN"])\n'
)
HF_ENV_COMMENT = "# Authenticated via HF_TOKEN in the cell above.\n"
HF_LEGACY_COMMENT = (
    "# Legacy hard-coded token removed; use HF_TOKEN from the environment.\n"
)

NOTEBOOK_PATCHES: dict[str, dict[str, Any]] = {
    "evaluation/model_generation/llama3_1/llama3_evaluation.ipynb": {
        "source_updates": {3: HF_ENV_LOGIN, 4: HF_ENV_COMMENT},
        "clear_outputs": {3, 4},
        "auth_cells": {3, 4},
    },
    "training/sft/GPT_SFT_only.ipynb": {
        "source_updates": {4: HF_ENV_LOGIN, 6: HF_LEGACY_COMMENT},
        "clear_outputs": {4},
        "auth_cells": {4, 6},
    },
    "evaluation/factuality/truthfulqa_benchmark.ipynb": {
        "source_updates": {2: HF_ENV_LOGIN},
        "clear_outputs": {2},
        "auth_cells": {2},
    },
}


def load_parent_notebook(repo_root: Path, relative_path: str, parent_rev: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{parent_rev}:{relative_path}"],
        cwd=repo_root,
        text=True,
    )
    return json.loads(raw)


def resolve_widget_key(widget_state: dict[str, Any], ref: str) -> str | None:
    if ref in widget_state:
        return ref
    for key in widget_state:
        if key.startswith(ref):
            return key
    return None


def collect_widget_subtree(widget_state: dict[str, Any], root_id: str) -> set[str]:
    to_remove: set[str] = set()
    queue = [root_id]
    while queue:
        widget_id = queue.pop()
        if widget_id in to_remove or widget_id not in widget_state:
            continue
        to_remove.add(widget_id)
        blob = json.dumps(widget_state[widget_id])
        for ref in IPY_MODEL_REF.findall(blob):
            resolved = resolve_widget_key(widget_state, ref)
            if resolved:
                queue.append(resolved)
    return to_remove


def widget_root_from_cell_output(output: dict[str, Any]) -> str | None:
    view = output.get("data", {}).get("application/vnd.jupyter.widget-view+json")
    if isinstance(view, dict):
        model_id = view.get("model_id")
        if isinstance(model_id, str):
            return model_id
    return None


def is_auth_widget_model(widget_obj: dict[str, Any]) -> bool:
    state = widget_obj.get("state", {})
    model_name = widget_obj.get("model_name") or state.get("_model_name")
    return model_name in AUTH_WIDGET_MODELS


def strip_auth_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for output in outputs:
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text_blob = "".join(text)
            else:
                text_blob = str(text)
            if AUTH_WARNING.search(text_blob):
                continue
        cleaned.append(output)
    return cleaned


def strip_auth_cell_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(metadata)
    for key in ("colab", "outputId", "execution"):
        cleaned.pop(key, None)
    return cleaned


def patch_notebook(nb: dict[str, Any], spec: dict[str, Any]) -> set[str]:
    removed_widget_roots: set[str] = set()
    widget_state = (
        nb.get("metadata", {})
        .get("widgets", {})
        .get("application/vnd.jupyter.widget-state+json", {})
    )

    for cell_index in spec.get("auth_cells", set()):
        cell = nb["cells"][cell_index]
        for output in list(cell.get("outputs", [])):
            root_id = widget_root_from_cell_output(output)
            if root_id:
                removed_widget_roots.add(root_id)

    for cell_index, source in spec.get("source_updates", {}).items():
        cell = nb["cells"][cell_index]
        cell["source"] = source.splitlines(keepends=True)
        cell["execution_count"] = None
        cell["metadata"] = strip_auth_cell_metadata(cell.get("metadata", {}))

    for cell_index in spec.get("clear_outputs", set()):
        nb["cells"][cell_index]["outputs"] = []

    for cell_index, cell in enumerate(nb["cells"]):
        if cell_index in spec.get("auth_cells", set()):
            continue
        cell["outputs"] = strip_auth_outputs(cell.get("outputs", []))

    if isinstance(widget_state, dict):
        to_remove: set[str] = set()
        for root_id in removed_widget_roots:
            to_remove |= collect_widget_subtree(widget_state, root_id)
        for widget_id, widget_obj in list(widget_state.items()):
            if isinstance(widget_obj, dict) and is_auth_widget_model(widget_obj):
                to_remove |= collect_widget_subtree(widget_state, widget_id)
        for widget_id in to_remove:
            widget_state.pop(widget_id, None)
        if not widget_state:
            nb.get("metadata", {}).pop("widgets", None)

    return removed_widget_roots


def write_notebook(path: Path, nb: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    repo_root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    parent_rev = argv[2] if len(argv) > 2 else "a2c1bdb"

    for relative_path, spec in NOTEBOOK_PATCHES.items():
        nb = load_parent_notebook(repo_root, relative_path, parent_rev)
        patch_notebook(nb, spec)
        write_notebook(repo_root / relative_path, nb)
        print(f"patched {relative_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
