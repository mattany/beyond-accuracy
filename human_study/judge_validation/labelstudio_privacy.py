"""Anonymize Label Studio export metadata for publication."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ANNOTATOR_PREFIX = "annotator_"


def _identity_key(completed_by: Dict[str, Any]) -> str:
    email = completed_by.get("email")
    if email:
        return str(email)
    if completed_by.get("id") is not None:
        return str(completed_by["id"])
    raise ValueError("completed_by missing email/id")


def build_annotator_mapping(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    identities = sorted(
        {
            _identity_key(ann["completed_by"])
            for task in tasks
            for ann in task.get("annotations", [])
            if ann.get("completed_by")
        }
    )
    return {
        identity: f"{ANNOTATOR_PREFIX}{index}"
        for index, identity in enumerate(identities, start=1)
    }


def anonymize_completed_by(
    completed_by: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, Any]:
    label = mapping[_identity_key(completed_by)]
    suffix = label.split("_")[-1]
    sanitized = dict(completed_by)
    sanitized["email"] = label
    sanitized["first_name"] = "Annotator"
    sanitized["last_name"] = suffix
    return sanitized


def sanitize_labelstudio_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapping = build_annotator_mapping(tasks)
    sanitized = deepcopy(tasks)
    for task in sanitized:
        task.pop("updated_by", None)
        task.pop("comment_authors", None)
        task.pop("drafts", None)
        for ann in task.get("annotations", []):
            ann.pop("updated_by", None)
            if ann.get("completed_by"):
                ann["completed_by"] = anonymize_completed_by(ann["completed_by"], mapping)
    return sanitized


def sanitize_labelstudio_file(path: Path) -> Dict[str, str]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    sanitized = sanitize_labelstudio_tasks(tasks)
    path.write_text(json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return build_annotator_mapping(tasks)


def annotation_stats(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    annotators = set()
    annotation_count = 0
    for task in tasks:
        for ann in task.get("annotations", []):
            annotation_count += 1
            if ann.get("completed_by"):
                annotators.add(_identity_key(ann["completed_by"]))
    return {
        "task_count": len(tasks),
        "annotation_count": annotation_count,
        "annotator_count": len(annotators),
    }


def assert_no_identifying_metadata(tasks: List[Dict[str, Any]]) -> None:
    serialized = json.dumps(tasks)
    if EMAIL_RE.search(serialized):
        raise AssertionError("email-shaped value remains in Label Studio export")
    for task in tasks:
        for ann in task.get("annotations", []):
            email = (ann.get("completed_by") or {}).get("email", "")
            if not str(email).startswith(ANNOTATOR_PREFIX):
                raise AssertionError("annotator label is not anonymized")
