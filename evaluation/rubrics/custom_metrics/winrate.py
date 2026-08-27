"""
Per-question win-rate comparison between two models using the same
metric weights and normalization as aggregate_v2.py.

Usage:
    python custom_metrics/winrate.py
    python custom_metrics/winrate.py --model-a llama-2-7b --model-b gpt-3.5-turbo-0125
    python custom_metrics/winrate.py --run 10
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from evaluation.rubrics.settings import result_directory
from evaluation.rubrics.custom_metrics.aggregate_v2 import (
    LOWER_IS_BETTER,
    METRIC_WEIGHTS,
    NORMALIZATION_RANGES,
    RUN_NUMBER,
    load_metric_data,
)


def normalize(values: pd.Series, metric_name: str) -> pd.Series:
    """Normalize a score series using the same fixed ranges as aggregate_v2."""
    if metric_name in NORMALIZATION_RANGES:
        lo, hi = NORMALIZATION_RANGES[metric_name]
    else:
        lo, hi = 0.0, 1.0

    if lo == hi:
        return pd.Series(0.5, index=values.index)

    norm = (values - lo) / (hi - lo)
    norm = norm.clip(0.0, 1.0)
    if metric_name in LOWER_IS_BETTER:
        norm = 1.0 - norm
    return norm


def per_question_scores(directory: str, model: str) -> pd.Series:
    """
    Return a Series of per-question weighted aggregate scores for *model*.
    Missing metric values for a question contribute 0 weight (graceful degradation).
    """
    weighted: pd.Series | None = None
    weight_used: pd.Series | None = None

    for metric_name, info in METRIC_WEIGHTS.items():
        w = info["weight"]
        df, _ = load_metric_data(directory, metric_name)
        if df is None:
            continue

        col = f"{model}__score"
        if col not in df.columns:
            print(f"  Warning: {col} not in {metric_name}.csv — skipping")
            continue

        normed = normalize(df[col], metric_name)  # NaN where original was NaN

        if weighted is None:
            weighted = normed.fillna(0.0) * w
            weight_used = normed.notna().astype(float) * w
        else:
            weighted = weighted.add(normed.fillna(0.0) * w, fill_value=0.0)
            weight_used = weight_used.add(normed.notna().astype(float) * w, fill_value=0.0)

    if weighted is None:
        return pd.Series(dtype=float)

    # Divide by the weight actually used per row (handles missing metrics gracefully)
    return weighted / weight_used.replace(0.0, np.nan)


def main(model_a: str, model_b: str, run_number: int):
    directory = result_directory(run_number)

    print(f"\nRun directory : {directory}")
    print(f"Model A       : {model_a}")
    print(f"Model B       : {model_b}\n")

    scores_a = per_question_scores(directory, model_a)
    scores_b = per_question_scores(directory, model_b)

    if scores_a.empty or scores_b.empty:
        sys.exit("Error: could not compute scores for one or both models.")

    # Align on shared row index
    idx = scores_a.index.intersection(scores_b.index)
    a = scores_a[idx]
    b = scores_b[idx]
    n = len(idx)

    a_wins = int((a > b).sum())
    b_wins = int((b > a).sum())
    ties   = int((a == b).sum())

    print(f"{'─'*50}")
    print(f"Questions evaluated : {n}")
    print(f"{'─'*50}")
    print(f"{model_a:<35} wins: {a_wins:3d}  ({a_wins/n*100:5.1f}%)")
    print(f"{model_b:<35} wins: {b_wins:3d}  ({b_wins/n*100:5.1f}%)")
    print(f"{'Ties':<35}      : {ties:3d}  ({ties/n*100:5.1f}%)")
    print(f"{'─'*50}")
    print(f"Mean score {model_a:<25}: {a.mean():.4f}")
    print(f"Mean score {model_b:<25}: {b.mean():.4f}")
    print(f"{'─'*50}\n")

    # Per-question detail
    detail = pd.DataFrame({
        "question_idx": idx,
        f"{model_a}__agg": a.values,
        f"{model_b}__agg": b.values,
        "winner": np.where(
            a.values > b.values, model_a,
            np.where(b.values > a.values, model_b, "tie"),
        ),
    })
    slug_b = model_b.replace("/", "_")
    out_path = os.path.join(directory, f"winrate__{model_a}_vs_{slug_b}.csv")
    detail.to_csv(out_path, index=False)
    print(f"Per-question breakdown saved to:\n  {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="llama-2-7b")
    parser.add_argument("--model-b", default="gpt-3.5-turbo-0125")
    parser.add_argument("--run", type=int, default=RUN_NUMBER)
    args = parser.parse_args()

    main(args.model_a, args.model_b, args.run)
