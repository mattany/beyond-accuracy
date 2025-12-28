#!/usr/bin/env python3
"""
Merge metaphor v8 intermediate results with v6 human / baseline columns.

Inputs:
  1) v8 intermediate_results.csv (has many rows per question_idx due to repetitions)
  2) metaphor_v6_balanced_20_formatted.csv (1 row per question; includes v6 columns)

Output:
  A CSV with exactly these 5 columns:
    - metaphor_v8_score
    - metaphor_v8_reason
    - mattany_metaphor_v6
    - metaphor_v6_score
    - metaphor_v6_reason

Notes:
  - We aggregate v8 to one row per question_idx by:
      * metaphor_v8_score = mean(score) across repetitions
      * metaphor_v8_reason = explanation from the repetition whose score is closest
        to the mean score (ties -> first)
  - We align v6 rows to question_idx by row order in the v6 CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_reason_closest_to_mean(group: pd.DataFrame, mean_score: float) -> str:
    if group.empty:
        return ""
    g = group.copy()
    g["_dist"] = (g["score"] - mean_score).abs()
    g = g.sort_values(["_dist", "repetition"], ascending=[True, True])
    val = g.iloc[0].get("explanation", "")
    return "" if pd.isna(val) else str(val)


def build_merged_df(v8_intermediate_path: Path, v6_formatted_path: Path) -> pd.DataFrame:
    v8 = pd.read_csv(v8_intermediate_path)
    required_v8 = {"question_idx", "repetition", "score", "explanation"}
    missing_v8 = required_v8 - set(v8.columns)
    if missing_v8:
        raise ValueError(
            f"v8 intermediate file missing columns {sorted(missing_v8)}. "
            f"Found columns: {list(v8.columns)}"
        )

    # Aggregate to one row per question_idx
    agg_rows = []
    for q_idx, group in v8.groupby("question_idx", sort=True):
        mean_score = float(group["score"].mean())
        reason = _pick_reason_closest_to_mean(group, mean_score)
        agg_rows.append(
            {
                "question_idx": int(q_idx),
                "metaphor_v8_score": mean_score,
                "metaphor_v8_reason": reason,
            }
        )
    v8_agg = pd.DataFrame(agg_rows).sort_values("question_idx").reset_index(drop=True)

    v6 = pd.read_csv(v6_formatted_path)
    required_v6 = {"mattany_metaphor_v6", "metaphor_v6_score", "metaphor_v6_reason"}
    missing_v6 = required_v6 - set(v6.columns)
    if missing_v6:
        raise ValueError(
            f"v6 formatted file missing columns {sorted(missing_v6)}. "
            f"Found columns: {list(v6.columns)}"
        )

    v6_sel = v6[list(required_v6)].copy()
    v6_sel = v6_sel.reset_index(drop=True)
    v6_sel["question_idx"] = v6_sel.index.astype(int)

    merged = v8_agg.merge(v6_sel, on="question_idx", how="left")
    out = merged[
        [
            "metaphor_v8_score",
            "metaphor_v8_reason",
            "mattany_metaphor_v6",
            "metaphor_v6_score",
            "metaphor_v6_reason",
        ]
    ].copy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v8",
        required=True,
        type=Path,
        help="Path to v8 intermediate_results.csv",
    )
    p.add_argument(
        "--v6",
        required=True,
        type=Path,
        help="Path to metaphor_v6_balanced_20_formatted.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: next to v8 file as merged_v8_with_v6.csv",
    )
    args = p.parse_args()

    out_path: Path = args.out or (args.v8.parent / "merged_v8_with_v6.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_merged_df(args.v8, args.v6)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()


