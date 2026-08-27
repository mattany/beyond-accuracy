 #!/usr/bin/env python3
"""
Apply Label Studio *tie-breaker* annotations to existing formatted CSVs.

Why this exists:
- Your formatted CSVs (e.g. balanced_30_humor_v5_conn_v4_formatted.csv) already contain
  two annotators' binary labels (e.g. mattany_connection_v4, nirgrn_connection_v4).
- Tie-breaker labeling in Label Studio uses a single field (tiebreaker_decision)
  + optional free-text (tiebreaker_reasoning), and tasks include Index + disagreement_metric
  like "connection_v4", "metaphor_v8", etc.
- For `intercoder_reliability_v2.py --csv-mode`, missing annotator values are currently
  treated as 0. Therefore, if we only fill tie-breaker values for disagreement items,
  we'd bias the majority vote.

So this script:
1) Reads tie-breaker JSON export, extracting (Index, disagreement_metric, annotator) -> decision/reason.
2) Writes a new annotator column per tie-breaker annotator into the *correct* formatted CSV:
     {tb_annotator}_{metric}_{vN}
     {tb_annotator}_{metric}_{vN}_reason
3) Fills the tie-breaker annotator values for *non-disagreement rows* to match the
   existing consensus between mattany/nirgrn (so there are no missing values).

This yields correct 3-coder majority behavior:
- If mattany==nirgrn, majority is unchanged.
- If mattany!=nirgrn, majority follows the tie-breaker decision.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TBKey:
    index: str
    disagreement_metric: str  # e.g. "connection_v4"
    tb_annotator: str  # sanitized


def _sanitize_annotator_name(name: str) -> str:
    """
    CSV-mode parser (`intercoder_reliability_v2.find_annotator_columns`) expects annotator prefix
    to match `[a-z_]+`. We therefore:
    - lowercase
    - replace any non-letter with underscore
    - collapse repeated underscores
    - strip leading/trailing underscores
    """
    s = (name or "").lower()
    s = re.sub(r"[^a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tiebreaker"


def _extract_ls_choice(result: List[dict], from_name: str) -> Optional[str]:
    for r in result:
        if r.get("type") == "choices" and r.get("from_name") == from_name:
            choices = r.get("value", {}).get("choices", [])
            if choices:
                return str(choices[0])
    return None


def _extract_ls_text(result: List[dict], from_name: str) -> str:
    for r in result:
        if r.get("type") == "textarea" and r.get("from_name") == from_name:
            text_list = r.get("value", {}).get("text", [])
            if not text_list:
                return ""
            return ". ".join([str(t) for t in text_list if t is not None]).strip()
    return ""


def _parse_disagreement_metric(dm: str) -> Tuple[str, str]:
    """
    Parse "connection_v4" -> ("connection", "v4").
    """
    dm = (dm or "").strip().lower()
    m = re.match(r"^([a-z_]+)_(v\d+)$", dm)
    if not m:
        raise ValueError(f"Unrecognized disagreement_metric format: {dm!r}")
    return m.group(1), m.group(2)


def load_tiebreaker_decisions(
    json_path: Path,
    decision_field: str = "tiebreaker_decision",
    reasoning_field: str = "tiebreaker_reasoning",
) -> Dict[TBKey, Tuple[int, str]]:
    """
    Returns:
      {(Index, disagreement_metric, tb_annotator): (decision01, reasoning_text)}
    """
    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    out: Dict[TBKey, Tuple[int, str]] = {}

    for t in tasks:
        data = t.get("data", {})
        idx = data.get("Index", data.get("index", data.get("qid", "")))
        dm = data.get("disagreement_metric", data.get("disagreement", data.get("metric", "")))
        if idx == "" or dm == "":
            continue

        # normalize Index key to string (CSV uses int sometimes)
        idx_str = str(idx)
        dm_str = str(dm).strip().lower()

        latest_by_annotator: Dict[str, dict] = {}
        for ann in t.get("annotations", []):
            email = (ann.get("completed_by", {}) or {}).get("email", "") or ""
            tb_annotator = _sanitize_annotator_name(email.split("@")[0] if "@" in email else email)
            ts = ann.get("updated_at", "")
            if tb_annotator not in latest_by_annotator or ts > latest_by_annotator[tb_annotator].get("updated_at", ""):
                latest_by_annotator[tb_annotator] = ann

        for tb_annotator, ann in latest_by_annotator.items():
            result = ann.get("result", []) or []
            choice = _extract_ls_choice(result, decision_field)
            if choice is None:
                continue
            if choice not in ("Yes", "No"):
                continue
            decision01 = 1 if choice == "Yes" else 0
            reasoning = _extract_ls_text(result, reasoning_field)
            out[TBKey(idx_str, dm_str, tb_annotator)] = (decision01, reasoning)

    return out


def _index_mask(df: pd.DataFrame, idx_value: str) -> pd.Series:
    if "Index" not in df.columns:
        raise ValueError("Target formatted CSV missing required 'Index' column.")
    return df["Index"].astype(str) == str(idx_value)


def apply_to_formatted_csv(
    formatted_csv_path: Path,
    tb_decisions: Dict[TBKey, Tuple[int, str]],
    *,
    fill_consensus: bool = False,
    allow_unresolved: bool = False,
    output_csv_path: Optional[Path] = None,
) -> None:
    output_path = output_csv_path or formatted_csv_path
    df = pd.read_csv(formatted_csv_path)

    # Determine which metrics exist in this formatted CSV based on the standard annotator columns.
    # We use these as "ground truth" consensus for filling non-disagreement rows.
    base_annotators = ["mattany", "nirgrn"]
    existing_metric_cols = [
        c for c in df.columns
        if any(c.lower().startswith(f"{a}_") for a in base_annotators)
        and re.search(r"_v\d+$", c.lower())
    ]
    metric_versions = sorted({c.lower().split("_", 1)[1] for c in existing_metric_cols})
    # metric_versions contains e.g. "connection_v4", "humor_v5"

    # Create and fill tie-breaker annotator columns
    tb_annotators = sorted({k.tb_annotator for k in tb_decisions.keys()})
    if not tb_annotators:
        raise ValueError(f"No tie-breaker annotations found in provided JSON (nothing to apply).")

    # Pre-create columns for every tb_annotator × metric_version present in this CSV
    for tb_annotator in tb_annotators:
        for mv in metric_versions:
            col = f"{tb_annotator}_{mv}"
            reason_col = f"{tb_annotator}_{mv}_reason"
            if col not in df.columns:
                df[col] = np.nan
            if reason_col not in df.columns:
                df[reason_col] = ""

    # Apply decisions only for matching (Index, disagreement_metric)
    for k, (decision01, reasoning) in tb_decisions.items():
        if k.disagreement_metric not in metric_versions:
            continue
        mask = _index_mask(df, k.index)
        if not mask.any():
            continue
        col = f"{k.tb_annotator}_{k.disagreement_metric}"
        reason_col = f"{k.tb_annotator}_{k.disagreement_metric}_reason"
        df.loc[mask, col] = decision01
        if reasoning:
            df.loc[mask, reason_col] = reasoning

    # Optionally fill missing tie-breaker values on consensus rows.
    # This is useful if you want to treat the tie-breaker as a full third coder
    # across all items. For "tie-breaker only breaks ties", keep this OFF.
    if not fill_consensus:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return

    # Fill missing tie-breaker values so CSV-mode doesn't interpret missing as 0.
    for mv in metric_versions:
        m_col = f"mattany_{mv}"
        n_col = f"nirgrn_{mv}"
        if m_col not in df.columns or n_col not in df.columns:
            # If your formatted CSV uses different base annotator column names, update here.
            continue
        m_vals = df[m_col]
        n_vals = df[n_col]

        # consensus rows
        consensus_mask = m_vals.notna() & n_vals.notna() & (m_vals.astype(float) == n_vals.astype(float))
        consensus_val = m_vals.where(consensus_mask)

        for tb_annotator in tb_annotators:
            tb_col = f"{tb_annotator}_{mv}"
            missing_mask = df[tb_col].isna()

            # Fill where we have consensus
            fill_mask = missing_mask & consensus_mask
            df.loc[fill_mask, tb_col] = consensus_val.loc[fill_mask].astype(float)

            # Remaining missing = unresolved disagreements (no tie-breaker recorded)
            remaining_missing = df[tb_col].isna().sum()
            if remaining_missing > 0 and not allow_unresolved:
                # If any remain, error out with actionable info
                unresolved_idx = df.loc[df[tb_col].isna(), "Index"].astype(str).head(10).tolist()
                raise ValueError(
                    f"{formatted_csv_path.name}: still missing {remaining_missing} values for {tb_col}. "
                    f"Example Index values: {unresolved_idx}. "
                    f"Either label them in Label Studio or pass --allow-unresolved."
                )
            elif remaining_missing > 0 and allow_unresolved:
                # Worst-case fallback: fill missing with consensus-breaking default (0)
                df.loc[df[tb_col].isna(), tb_col] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply tie-breaker Label Studio JSON to formatted CSV(s).")
    parser.add_argument("--tiebreaker-json", required=True, type=str, help="Path to Label Studio JSON export for tie-breaker tasks.")
    parser.add_argument(
        "--formatted-csv",
        required=True,
        nargs="+",
        type=str,
        help="One or more formatted CSVs to update in-place (e.g. balanced_30_*_formatted.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="If set, write updated CSV copies into this directory (same basenames), leaving originals untouched.",
    )
    parser.add_argument("--decision-field", default="tiebreaker_decision", help="Label Studio from_name for Yes/No field.")
    parser.add_argument("--reasoning-field", default="tiebreaker_reasoning", help="Label Studio from_name for reasoning textarea.")
    parser.add_argument(
        "--fill-consensus",
        action="store_true",
        help="If set, fills tie-breaker annotator values on consensus rows to match mattany==nirgrn.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="If set, unresolved tie-breaker values (neither consensus nor annotated) are filled with 0 instead of erroring.",
    )
    args = parser.parse_args()

    tb_json_path = Path(args.tiebreaker_json)
    decisions = load_tiebreaker_decisions(
        tb_json_path,
        decision_field=args.decision_field,
        reasoning_field=args.reasoning_field,
    )

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path_str in args.formatted_csv:
        src = Path(csv_path_str)
        out_path = (output_dir / src.name) if output_dir else None
        apply_to_formatted_csv(
            src,
            decisions,
            fill_consensus=args.fill_consensus,
            allow_unresolved=args.allow_unresolved,
            output_csv_path=out_path,
        )

    print("Done. Updated formatted CSV(s) in-place.")


if __name__ == "__main__":
    main()


