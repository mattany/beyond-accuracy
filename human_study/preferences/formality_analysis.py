#!/usr/bin/env python3
"""
Formality covariate analysis for Experiment B (rebuttal, reviewer 4DDk).

Adds a sentence-level *formality* score to the human-preference logistic
regression to test whether the preference for AI/synthetic explanations is
driven by surface formality rather than substantive communication quality
(e.g., scaffolding, readability).

Formality operationalization: the classic Heylighen & Dewaele (2002) F-measure,
computed purely from part-of-speech frequencies (no neural model / torch):

    F = ( noun%% + adjective%% + preposition%% + article%%
        - pronoun%% - verb%% - adverb%% - interjection%%
        + 100 ) / 2                                     (range 0-100)

Higher F => more formal (explicit, context-independent language); lower F =>
more informal (deictic, context-dependent language).

The score is added as an extra difference predictor (formality_a - formality_b),
rescaled to [0,1] by dividing F by 100 so its coefficient is comparable to the
other [0,1] predictors (e.g., readability). We reuse the exact regression
pipeline from ``logistic_regression.py`` so results are directly comparable to
Table 3 in the paper.

Outputs (written to human_study/preferences/data/):
  - formality_scores.csv                                   (per-row F_a, F_b, diff)
  - logistic_regression_continuous_with_formality.csv                 (Full, N=800)
  - logistic_regression_continuous_with_formality_no_prompted_dpo.csv (No P-DPO, N=600)
  - logistic_regression_continuous_with_formality_no_human.csv        (No Human, N=700)
  - logistic_regression_continuous_with_formality_human_only.csv      (cluster 7)

The script also prints LaTeX-ready rows for Table 3 (formality added to every
spec) plus a no-formality sanity pass that should reproduce the published table.

Usage:
  python formality_analysis.py
"""

import os
import re
import sys
from functools import lru_cache

import nltk
import pandas as pd

# Reuse the existing regression pipeline so numbers match the paper exactly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logistic_regression import (  # noqa: E402
    DATA_DIR,
    HUMAN_ANSWER_CLUSTERS,
    MODEL_TYPE_DIFFERENCE,
    load_evaluation_data,
    load_metric_scores,
    prepare_data,
    print_results_table,
    run_logistic_regression,
)

# ---------------------------------------------------------------------------
# Heylighen & Dewaele (2002) F-measure
# ---------------------------------------------------------------------------
# Penn Treebank tag buckets.
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}
PREP_TAGS = {"IN"}  # prepositions + subordinating conjunctions (approximation)
PRON_TAGS = {"PRP", "PRP$", "WP", "WP$"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}
ADV_TAGS = {"RB", "RBR", "RBS", "WRB"}
INTJ_TAGS = {"UH"}
ARTICLES = {"a", "an", "the"}  # Penn has no article tag; match lexically.

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _ensure_nltk():
    """Download the POS tagger resources if missing."""
    for res in (
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ):
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


@lru_cache(maxsize=8192)
def formality_f_measure(text: str):
    """Return the Heylighen-Dewaele F-measure (0-100), or None if no words."""
    if not text:
        return None
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return None
    tagged = nltk.pos_tag(tokens)
    n = len(tagged)

    def pct(count):
        return 100.0 * count / n

    noun = sum(1 for _, t in tagged if t in NOUN_TAGS)
    adj = sum(1 for _, t in tagged if t in ADJ_TAGS)
    prep = sum(1 for _, t in tagged if t in PREP_TAGS)
    article = sum(1 for w, _ in tagged if w.lower() in ARTICLES)
    pron = sum(1 for _, t in tagged if t in PRON_TAGS)
    verb = sum(1 for _, t in tagged if t in VERB_TAGS)
    adv = sum(1 for _, t in tagged if t in ADV_TAGS)
    intj = sum(1 for _, t in tagged if t in INTJ_TAGS)

    return (
        pct(noun) + pct(adj) + pct(prep) + pct(article)
        - pct(pron) - pct(verb) - pct(adv) - pct(intj)
        + 100.0
    ) / 2.0


def add_formality_column(eval_df: pd.DataFrame, prepared_df: pd.DataFrame) -> pd.DataFrame:
    """Compute formality_diff aligned to prepared_df (same iterrows order).

    F is rescaled to [0,1] (divide by 100) so the coefficient is comparable to
    the other [0,1] predictors. Rows where either side has no words get NaN and
    are dropped by the regression, matching existing behavior.
    """
    diffs = []
    for _, row in eval_df.iterrows():
        fa = formality_f_measure(str(row.get("explanation_a", "")))
        fb = formality_f_measure(str(row.get("explanation_b", "")))
        if fa is None or fb is None:
            diffs.append(float("nan"))
        else:
            diffs.append((fa - fb) / 100.0)
    out = prepared_df.copy()
    out["formality_diff"] = diffs
    return out


def _prepare(eval_df: pd.DataFrame, with_formality: bool) -> pd.DataFrame:
    """Build the regression frame using the exact pipeline behind Table 3.

    When ``with_formality`` is True, the Heylighen-Dewaele formality difference
    is appended as one more control predictor (like log answer length, which is
    already present in every model).
    """
    scores = load_metric_scores()
    prepared = prepare_data(
        eval_df, scores, "continuous", model_type=MODEL_TYPE_DIFFERENCE
    )
    if with_formality:
        prepared = add_formality_column(eval_df, prepared)
    return prepared


def _run(eval_df: pd.DataFrame, label: str, with_formality: bool = True) -> pd.DataFrame:
    prepared = _prepare(eval_df, with_formality)
    print(f"\n{'#' * 90}\n# {label}\n{'#' * 90}")
    results_df, n = run_logistic_regression(
        prepared, "continuous", model_type=MODEL_TYPE_DIFFERENCE
    )
    print_results_table(results_df, "continuous")
    return results_df


# ---------------------------------------------------------------------------
# LaTeX-ready formatting to fill Table 3 in the paper
# ---------------------------------------------------------------------------
ROW_ORDER = [
    ("metaphor_v8", "Metaphor"),
    ("analogy_v2", "Analogy"),
    ("scaffolding_v2", "Scaffolding"),
    ("jargon", "Jargon"),
    ("readability", "Readability"),
    ("answer_length", "Ans.\\ length"),
    ("formality", "Formality"),
]


def _nz(x: float) -> str:
    """2 decimals, leading zero stripped, LaTeX minus (matches paper style)."""
    s = f"{x:.2f}"
    neg = s.startswith("-")
    s = s.lstrip("-")
    if s.startswith("0."):
        s = s[1:]
    return ("$-$" if neg else "") + s


def _nzp(x: float) -> str:
    """Positive-number (SE) formatter: 2 decimals, leading zero stripped."""
    s = f"{abs(x):.2f}"
    if s.startswith("0."):
        s = s[1:]
    return s


def _lookup(results_df: pd.DataFrame, metric: str):
    hit = results_df[results_df["metric"] == metric]
    return hit.iloc[0] if len(hit) else None


def _cell(row) -> str:
    if row is None:
        return "\\multicolumn{1}{c}{--}"
    p = row["p_value"]
    star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"{_nz(row['coefficient'])}{star}\\,({_nzp(row['std_error'])})"


def print_latex_table(results_by_spec):
    print(f"\n{'=' * 90}\nLATEX-READY ROWS FOR TABLE 3 (formality in every spec)\n{'=' * 90}")
    for metric, pretty in ROW_ORDER:
        cells = [_cell(_lookup(df, metric)) for _, df in results_by_spec]
        print(f"{pretty:<12} & " + " & ".join(cells) + r" \\")
    print("=" * 90)
    print("Column order: " + " | ".join(label for label, _ in results_by_spec))


def main():
    _ensure_nltk()

    # Recompute the three Table-3 specifications, now with the formality
    # covariate added to *every* model (parallel to log answer length, which is
    # already a control in every model). Uses the exact pipeline that produced
    # the published Table 3, so the numbers are directly comparable.
    specs = [
        ("Full (N=800)", {}),
        ("No P-DPO (N=600)", {"exclude_prompted_dpo": True}),
        ("No Human (N=700)", {"exclude_human": True}),
    ]

    results_by_spec = []
    for label, kwargs in specs:
        eval_df = load_evaluation_data(**kwargs)
        res = _run(eval_df, f"{label} (+ formality covariate)", with_formality=True)
        results_by_spec.append((label, res))
        fname = (
            "logistic_regression_continuous_with_formality"
            + ("_no_prompted_dpo" if kwargs.get("exclude_prompted_dpo") else "")
            + ("_no_human" if kwargs.get("exclude_human") else "")
            + ".csv"
        )
        res.to_csv(DATA_DIR / fname, index=False)
        print(f"\nSaved: {DATA_DIR / fname}")

    # Sanity check: reproduce the ORIGINAL (no-formality) specs to confirm the
    # pipeline matches the published Table 3 and to quantify the shift.
    print(f"\n{'=' * 90}\nSANITY: same specs WITHOUT formality (should match published Table 3)\n{'=' * 90}")
    for label, kwargs in specs:
        eval_df = load_evaluation_data(**kwargs)
        _run(eval_df, f"{label} (no formality)", with_formality=False)

    # Human-vs-AI only (cluster 7 = Human vs GPT_cot) with formality, for prose.
    eval_all = load_evaluation_data()
    eval_human = eval_all[eval_all["cluster"].isin(HUMAN_ANSWER_CLUSTERS)].copy()
    res_human = _run(
        eval_human, "HUMAN-vs-AI ONLY (cluster 7, + formality)", with_formality=True
    )
    res_human.to_csv(
        DATA_DIR / "logistic_regression_continuous_with_formality_human_only.csv",
        index=False,
    )

    # Per-row formality scores for transparency.
    rows = []
    for _, row in eval_all.iterrows():
        fa = formality_f_measure(str(row.get("explanation_a", "")))
        fb = formality_f_measure(str(row.get("explanation_b", "")))
        rows.append(
            {
                "comparison_id": row.get("comparison_id"),
                "cluster": row.get("cluster"),
                "formality_a": fa,
                "formality_b": fb,
                "formality_diff": (None if fa is None or fb is None else fa - fb),
            }
        )
    fdf = pd.DataFrame(rows)
    fdf.to_csv(DATA_DIR / "formality_scores.csv", index=False)
    # SD of the [0,1]-rescaled formality difference (for the paper's prose).
    sd01 = (fdf["formality_diff"].dropna() / 100.0).std()
    print(
        f"\nFormality (F, 0-100): A mean={fdf['formality_a'].mean():.1f}, "
        f"B mean={fdf['formality_b'].mean():.1f}; "
        f"SD of rescaled diff (/100) = {sd01:.3f}"
    )

    print_latex_table(results_by_spec)
    print("\n--- Human-vs-AI only (for prose sentence) ---")
    for metric, pretty in ROW_ORDER:
        r = _lookup(res_human, metric)
        if r is not None:
            print(f"  {pretty:<12} {_cell(r)}  (p={r['p_value']:.4f})")


if __name__ == "__main__":
    main()
