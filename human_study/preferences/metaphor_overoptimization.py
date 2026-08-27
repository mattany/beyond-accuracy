#!/usr/bin/env python3
"""
Metaphor over-optimization analysis (rebuttal: cP8p / X6Ai).

Two reviewer asks:
  * cP8p: the paper identifies DPO/over-prompting overusing metaphors but
    proposes no mitigation strategy.
  * X6Ai: metrics can be over-optimized; add qualitative analysis of cases
    where DPO improves rubric scores but hurts human preference.

This script provides the *quantitative support* for the response WITHOUT
introducing a new (unvalidated) metric: it reuses the existing, already-
validated G-Eval metaphor judge (metaphor_v8) and the per-answer rubric
dimensions already computed for the paper.

Outputs:
  1. Metaphor rate (mean metaphor_v8 score = fraction of answers judged to
     contain an active metaphor) per model, DPO vs. the rest.
  2. Rubric aggregate (equal-weight mean of the six normalized dimensions,
     matching the paper's weighting) per answer.
  3. "Rubric up, preference down" case study: DPO comparisons where the DPO
     answer scored higher on the rubric aggregate yet lost the human vote,
     with the metaphor gap highlighted.

Usage:
  python metaphor_overoptimization.py --output /tmp/dpo_rubric_up_pref_down.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from logistic_regression import normalize_score  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results"
RUN_DIR = RESULTS_DIR / "preference_metrics"
# rubric_scores is the per-model rubric evaluation (one column per model), which unlike
# the pairwise Experiment-B set includes the Human-DPO (organic_dpo) variant.
PER_MODEL_RUN = RESULTS_DIR / "rubric_scores"
EVAL = DATA_DIR / "experiment_b_eval_dataset.csv"

DPO_MODELS = {"scicomma-3.1-dpo", "scicomma-3.1-dpo_prompt"}

# SFT -> DPO pairs for isolating the cause of metaphor over-optimization.
# (label, SFT model column stem, DPO model column stem) in rubric_scores.
SFT_DPO_PAIRS = [
    ("Synthetic, unprompted", "SciComma-3.1-8B_y", "scicomma-3.1-dpo"),
    ("Synthetic, prompted", "SciComma-3.1-8B_prompt", "scicomma-3.1-dpo_prompt"),
    ("Human/organic, unprompted", "organic_sft", "organic_dpo"),
    ("Human/organic, prompted", "organic_sft_prompt", "organic_dpo_prompt"),
]

# Six equally-weighted rubric dimensions (paper weighting). Each is mapped to
# [0,1]: readability is the mean of its four normalized sub-metrics, jargon is
# min-max normalized, and the four G-Eval judges are already on [0,1].
GEVAL_METRICS = {
    "scaffolding": "scaffolding_v2",
    "analogy": "analogy_v2",
    "metaphor": "metaphor_v8",
    "humor": "humor_v5",
}
READABILITY_SUB = ["flesch_reading_ease", "flesch_kincaid", "ari", "dale_chall"]


def _load_side_scores() -> dict:
    """Load each metric's per-row a/b scores (row-aligned with the eval set).

    Returns {metric: {'a': np.array, 'b': np.array}} of raw scores.
    """
    scores = {}
    metrics = list(GEVAL_METRICS.values()) + ["jargon"] + READABILITY_SUB
    for metric in metrics:
        df = pd.read_csv(RUN_DIR / f"{metric}.csv")
        scores[metric] = {
            "a": df["explanation_a__score"].values,
            "b": df["explanation_b__score"].values,
        }
    return scores


def _rubric_side(scores: dict, side: str) -> np.ndarray:
    """Equal-weight mean of the six [0,1] rubric dimensions for one side."""
    dims = []
    for _, col in GEVAL_METRICS.items():
        dims.append(np.asarray(scores[col][side], dtype=float))
    jarg = np.array([normalize_score(v, "jargon") for v in scores["jargon"][side]])
    dims.append(jarg)
    read = np.mean(
        [[normalize_score(v, m) for v in scores[m][side]] for m in READABILITY_SUB],
        axis=0,
    )
    dims.append(read)
    return np.mean(dims, axis=0)


def _to_long(scores: dict, eval_df: pd.DataFrame) -> pd.DataFrame:
    """One row per answer with model, metaphor score, rubric aggregate, win."""
    recs = []
    for side in ("a", "b"):
        selected = (
            (eval_df["human_choice"] == "Explanation A").values
            if side == "a"
            else (eval_df["human_choice"] == "Explanation B").values
        )
        recs.append(pd.DataFrame({
            "model": eval_df[f"model_{side}"].values,
            "metaphor": scores["metaphor_v8"][side],
            "rubric": _rubric_side(scores, side),
            "selected": selected.astype(int),
        }))
    return pd.concat(recs, ignore_index=True)


def metaphor_by_model(long: pd.DataFrame):
    print("=" * 70)
    print("1) METAPHOR RATE BY MODEL (mean metaphor_v8 = fraction w/ metaphor)")
    print("=" * 70)
    g = (long.groupby("model")["metaphor"]
         .agg(["mean", "count"])
         .sort_values("mean", ascending=False))
    for model, r in g.iterrows():
        tag = "  <-- DPO" if model in DPO_MODELS else ""
        print(f"  {model:<28} {r['mean']:.3f}  (n={int(r['count'])}){tag}")

    dpo = long[long["model"].isin(DPO_MODELS)]["metaphor"]
    rest = long[~long["model"].isin(DPO_MODELS)]["metaphor"]
    print("-" * 70)
    print(f"  DPO models        : {dpo.mean():.3f}  (n={len(dpo)})")
    print(f"  All non-DPO models: {rest.mean():.3f}  (n={len(rest)})")
    print(f"  Absolute gap      : {dpo.mean() - rest.mean():+.3f} "
          f"({dpo.mean() / rest.mean():.2f}x)")
    try:
        from scipy import stats
        t, p = stats.ttest_ind(dpo, rest, equal_var=False)
        print(f"  Welch t-test      : t={t:.2f}, p={p:.2e}")
    except Exception as e:  # pragma: no cover
        print(f"  (scipy unavailable: {e})")


def case_study(scores: dict, eval_df: pd.DataFrame, output_path: Path, top_k: int = 5):
    print("\n" + "=" * 70)
    print("2) CASE STUDY: DPO 'RUBRIC UP, PREFERENCE DOWN'")
    print("=" * 70)
    ra = _rubric_side(scores, "a")
    rb = _rubric_side(scores, "b")
    ma = scores["metaphor_v8"]["a"]
    mb = scores["metaphor_v8"]["b"]
    choice = eval_df["human_choice"].values

    rows = []
    for i in range(len(eval_df)):
        for side in ("a", "b"):
            model = eval_df.iloc[i][f"model_{side}"]
            if model not in DPO_MODELS:
                continue
            dpo_rub = ra[i] if side == "a" else rb[i]
            opp_rub = rb[i] if side == "a" else ra[i]
            dpo_met = ma[i] if side == "a" else mb[i]
            opp_met = mb[i] if side == "a" else ma[i]
            sel_label = "Explanation A" if side == "a" else "Explanation B"
            selected = 1 if choice[i] == sel_label else 0
            # DPO scored higher on rubric but was NOT selected by humans.
            if dpo_rub > opp_rub and selected == 0:
                rows.append({
                    "idx": i,
                    "cluster": int(eval_df.iloc[i]["cluster"]),
                    "dpo_side": side,
                    "dpo_model": model,
                    "rubric_gap": dpo_rub - opp_rub,
                    "dpo_rubric": dpo_rub,
                    "opp_rubric": opp_rub,
                    "dpo_metaphor": dpo_met,
                    "opp_metaphor": opp_met,
                    "question": eval_df.iloc[i]["question"],
                })

    cs = pd.DataFrame(rows)
    n_dpo_answers = sum(
        eval_df[f"model_{s}"].isin(DPO_MODELS).sum() for s in ("a", "b")
    )
    print(f"  DPO answers total                       : {n_dpo_answers}")
    print(f"  DPO 'rubric-up but lost the vote' cases  : {len(cs)}")
    met_driven = cs[cs["dpo_metaphor"] > cs["opp_metaphor"]]
    print(f"    ...of which DPO also had MORE metaphor : {len(met_driven)} "
          f"({100*len(met_driven)/max(len(cs),1):.0f}%)")

    # Best illustrative examples: rubric-up AND metaphor-up, largest rubric gap.
    top = met_driven.sort_values("rubric_gap", ascending=False).head(top_k)
    print(f"\n  Top {len(top)} illustrative examples (largest rubric gap, "
          f"metaphor higher for DPO):")
    for _, r in top.iterrows():
        print("  " + "-" * 66)
        print(f"  idx={r['idx']} cluster={r['cluster']} dpo={r['dpo_model']}")
        print(f"    rubric: DPO={r['dpo_rubric']:.3f} vs opp={r['opp_rubric']:.3f}"
              f"  (gap +{r['rubric_gap']:.3f})")
        print(f"    metaphor: DPO={r['dpo_metaphor']:.1f} vs opp={r['opp_metaphor']:.1f}")
        print(f"    Q: {str(r['question'])[:90]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cs.to_csv(output_path, index=False)
    print(f"\n  Saved full case list: {output_path}")
    return cs


def sft_dpo_contrast():
    """Isolate the cause: metaphor rate change from each SFT base to its DPO.

    Uses the per-model rubric evaluation (rubric_scores), which includes Human-DPO
    (organic_dpo). Shows that the metaphor inflation is specific to the
    *synthetic* preference signal, not DPO itself.
    """
    print("\n" + "=" * 70)
    print("3) SFT -> DPO METAPHOR CONTRAST (rubric_scores per-model eval)")
    print("=" * 70)
    path = PER_MODEL_RUN / "metaphor_v8.csv"
    if not path.exists():
        print(f"  (skipped: {path} not found)")
        return
    m = pd.read_csv(path)
    for label, sft, dpo in SFT_DPO_PAIRS:
        sc, dc = f"{sft}__score", f"{dpo}__score"
        if sc not in m.columns or dc not in m.columns:
            print(f"  {label:<28} (missing columns)")
            continue
        s, d = m[sc].mean(), m[dpo + "__score"].mean()
        ratio = f"{d / s:.1f}x" if s > 0 else "n/a"
        print(f"  {label:<28} SFT {s:.3f} -> DPO {d:.3f}  "
              f"(Delta {d - s:+.3f}, {ratio})")


def main(output_path: Path):
    eval_df = pd.read_csv(EVAL)
    scores = _load_side_scores()
    # Confirm row-alignment: metaphor scores must correspond to the eval texts.
    assert len(scores["metaphor_v8"]["a"]) == len(eval_df), "row mismatch"

    long = _to_long(scores, eval_df)
    metaphor_by_model(long)
    case_study(scores, eval_df, output_path=output_path)
    sft_dpo_contrast()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="path for dpo_rubric_up_pref_down.csv (use /tmp/... to avoid overwriting canonical data)",
    )
    args = parser.parse_args()
    main(output_path=Path(args.output))
