"""Test whether GPT-3.5- vs GPT-5-distilled students differ significantly.

Reconstructs per-question aggregate rubric scores from the canonical rubric
metric CSVs, replicating the exact normalization/weighting in aggregate_v2.py,
then runs a paired Wilcoxon signed-rank test (and paired t-test) per inference mode.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RESULTS = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "results"
    / "rubric_scores"
)

# metric -> (file, weight), mirroring aggregate_v2.CLUSTERS
METRICS = {
    "jargon": ("jargon.csv", 1 / 6),
    "scaffolding_v2": ("scaffolding_v2.csv", 1 / 6),
    "analogy_v2": ("analogy_v2.csv", 1 / 6),
    "metaphor_v8": ("metaphor_v8.csv", 1 / 6),
    "humor_v5": ("humor_v5.csv", 1 / 6),
    "flesch_reading_ease": ("flesch_reading_ease.csv", 1 / 24),
    "flesch_kincaid": ("flesch_kincaid.csv", 1 / 24),
    "dale_chall": ("dale_chall.csv", 1 / 24),
    "ari": ("ari.csv", 1 / 24),
}
NORMALIZATION_RANGES = {
    "jargon": (0.65, 1.0),
    "flesch_kincaid": (6, 16),
    "ari": (6, 16),
    "dale_chall": (7, 12),
    "flesch_reading_ease": (0.3, 0.7),
}
LOWER_IS_BETTER = ["ari", "dale_chall", "flesch_kincaid"]

MODELS = {
    "gpt35_unprompted": "SciComma-3.1-8B_y",
    "gpt5_unprompted": "SciComma-3.1-8B_gpt5",
    "gpt35_prompted": "SciComma-3.1-8B_prompt",
    "gpt5_prompted": "SciComma-3.1-8B_gpt5_prompt",
}


def normalize(series, metric):
    lo, hi = NORMALIZATION_RANGES.get(metric, (0, 1))
    if lo == hi:
        return pd.Series(0.5, index=series.index)
    out = ((series - lo) / (hi - lo)).clip(0, 1)
    if metric in LOWER_IS_BETTER:
        out = 1 - out
    return out


def per_question_total(model):
    total = None
    col = f"{model}__score"
    for metric, (fname, weight) in METRICS.items():
        # read only question + this model's score column (skip huge __reason cols)
        df = pd.read_csv(RUN_RESULTS / fname, usecols=["question", col])
        s = df.set_index("question")[col]
        s = normalize(s, metric) * weight
        total = s if total is None else total.add(s, fill_value=np.nan)
    return total.dropna()


DIMENSIONS = {  # dimension -> list of (file, metric) whose normalized scores are averaged
    "jargon": [("jargon.csv", "jargon")],
    "readability": [
        ("flesch_reading_ease.csv", "flesch_reading_ease"),
        ("flesch_kincaid.csv", "flesch_kincaid"),
        ("dale_chall.csv", "dale_chall"),
        ("ari.csv", "ari"),
    ],
    "scaffolding": [("scaffolding_v2.csv", "scaffolding_v2")],
    "analogy": [("analogy_v2.csv", "analogy_v2")],
    "metaphor": [("metaphor_v8.csv", "metaphor_v8")],
    "humor": [("humor_v5.csv", "humor_v5")],
}


def per_question_dim(model, parts):
    col = f"{model}__score"
    acc, n = None, 0
    for fname, metric in parts:
        df = pd.read_csv(RUN_RESULTS / fname, usecols=["question", col])
        s = normalize(df.set_index("question")[col], metric)
        acc = s if acc is None else acc.add(s, fill_value=np.nan)
        n += 1
    return (acc / n).dropna()


def per_dimension_tests():
    for mode, a, b in [
        ("Unprompted", "SciComma-3.1-8B_y", "SciComma-3.1-8B_gpt5"),
        ("Prompted", "SciComma-3.1-8B_prompt", "SciComma-3.1-8B_gpt5_prompt"),
    ]:
        print(f"=== Per-dimension: {mode} (GPT-3.5 vs GPT-5 student) ===")
        for dim, parts in DIMENSIONS.items():
            xa = per_question_dim(a, parts)
            ya = per_question_dim(b, parts)
            pair = pd.concat([xa, ya], axis=1, join="inner").dropna()
            x, y = pair.iloc[:, 0].values, pair.iloc[:, 1].values
            if np.allclose(x, y):
                p = float("nan")
                ptxt = "identical"
            else:
                try:
                    p = stats.wilcoxon(x, y).pvalue
                    ptxt = f"p={p:.4f}"
                except ValueError:
                    ptxt = "n/a (zero diffs)"
            print(f"  {dim:12s} {x.mean():.3f} -> {y.mean():.3f}  (d={y.mean()-x.mean():+.3f})  {ptxt}")
        print()


def main():
    totals = {k: per_question_total(v) for k, v in MODELS.items()}

    print("Mean per-question totals (sanity check vs total_scores.csv):")
    for k, v in totals.items():
        print(f"  {k:18s} n={len(v):3d}  mean={v.mean():.4f}")
    print()

    for mode, a, b in [
        ("Unprompted", "gpt35_unprompted", "gpt5_unprompted"),
        ("Prompted", "gpt35_prompted", "gpt5_prompted"),
    ]:
        pair = pd.concat([totals[a], totals[b]], axis=1, join="inner").dropna()
        x, y = pair.iloc[:, 0].values, pair.iloc[:, 1].values
        w = stats.wilcoxon(x, y)
        t = stats.ttest_rel(x, y)
        diff = y.mean() - x.mean()
        print(f"=== {mode} (N={len(x)} paired questions) ===")
        print(f"  GPT-3.5 student mean = {x.mean():.4f}")
        print(f"  GPT-5   student mean = {y.mean():.4f}")
        print(f"  mean diff (GPT-5 - GPT-3.5) = {diff:+.4f}")
        print(f"  Wilcoxon signed-rank: W={w.statistic:.1f}, p={w.pvalue:.4f}")
        print(f"  paired t-test:        t={t.statistic:.3f}, p={t.pvalue:.4f}")
        print()

    per_dimension_tests()


if __name__ == "__main__":
    main()
