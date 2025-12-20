#!/usr/bin/env python3
import json
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


METRIC_NAMES = ["Analogy", "Metaphor", "Humor", "Connection"]


def load_tasks(json_path: str) -> List[Dict[str, Any]]:
    """Load Label Studio JSON export."""
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return tasks


def extract_annotators(tasks: List[Dict[str, Any]]) -> List[str]:
    """Get sorted list of annotator emails present in the data."""
    annotators = sorted(
        {
            ann["completed_by"]["email"]
            for t in tasks
            for ann in t.get("annotations", [])
        }
    )
    return annotators


def extract_metrics_from_result(result: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Extract binary metrics from a single annotation's `result` field.
    Returns a dict with keys in METRIC_NAMES and values in {0,1}.
    """
    metrics = {m: 0 for m in METRIC_NAMES}
    for r in result:
        if r.get("type") == "choices":
            for c in r.get("value", {}).get("choices", []):
                if c == "Analogy":
                    metrics["Analogy"] = 1
                elif c == "Metaphor":
                    metrics["Metaphor"] = 1
                elif c == "Humor":
                    metrics["Humor"] = 1
                elif c == "Connection to everyday life":
                    metrics["Connection"] = 1
    return metrics


def build_item_table(tasks: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a DataFrame with one row per (question_id, model) where all annotators are present.
    Columns include:
      - qid, model
      - per-annotator binary metrics
      - aggregated human mean metrics
      - LLM scores for each metric
    """
    annotators = extract_annotators(tasks)
    rows = []

    for t in tasks:
        qid = t["data"]["question_id"]
        model = t["data"]["model"]
        data = t["data"]

        # Latest annotation per annotator
        latest: Dict[str, Dict[str, Any]] = {}
        for ann in t.get("annotations", []):
            email = ann["completed_by"]["email"]
            ts = ann["updated_at"]
            if email not in latest or ts > latest[email]["ts"]:
                latest[email] = {
                    "ts": ts,
                    "metrics": extract_metrics_from_result(ann.get("result", [])),
                }

        # Only keep items where all annotators are present
        if len(latest) != len(annotators):
            continue

        row: Dict[str, Any] = {"qid": qid, "model": model}

        # Store per-annotator binary labels as list in a stable order
        for metric in METRIC_NAMES:
            row[f"{metric}_per_annotator"] = [
                latest[email]["metrics"][metric] for email in annotators
            ]

        # Human mean metrics
        for metric in METRIC_NAMES:
            arr = np.array(row[f"{metric}_per_annotator"], dtype=float)
            row[f"human_mean_{metric}"] = float(arr.mean())

        # LLM scores from data
        row["LLM_Analogy"] = data.get("analogy_explicit_score", np.nan)
        row["LLM_Metaphor"] = data.get("metaphor_explicit_score", np.nan)
        row["LLM_Humor"] = data.get("humor_explicit_score", np.nan)
        row["LLM_Connection"] = data.get("connection_to_everyday_life_score", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, annotators


def percent_agreement(matrix: List[List[int]]) -> float:
    """
    Compute percent agreement across coders for a metric.
    matrix: list of rows, each row is list of coder labels (e.g., [0,1,0])
    Returns proportion of items where all coders agree.
    """
    arr = np.array(matrix)
    agree = [len(set(row)) == 1 for row in arr]
    return float(np.mean(agree))


def fleiss_kappa(matrix: List[List[int]]) -> Tuple[float, float, float]:
    """
    Compute Fleiss' kappa for binary labels.
    matrix: list of rows, each row = ratings of k coders (0/1).
    Returns (kappa, P_bar, P_e).
    """
    X = np.array(matrix)
    N, n = X.shape
    n0 = (X == 0).sum(axis=1)
    n1 = (X == 1).sum(axis=1)

    # per-item agreement
    P_i = (n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1))
    P_bar = P_i.mean()

    # expected agreement by chance
    p0 = n0.sum() / (N * n)
    p1 = n1.sum() / (N * n)
    P_e = p0 ** 2 + p1 ** 2

    if (1 - P_e) == 0:
        kappa = np.nan
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    return float(kappa), float(P_bar), float(P_e)


def compute_intercoder_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percent agreement and Fleiss' kappa for each metric.
    Returns a summary DataFrame.
    """
    rows = []
    for metric in METRIC_NAMES:
        mat = df[f"{metric}_per_annotator"].tolist()
        pa = percent_agreement(mat)
        kappa, P_bar, P_e = fleiss_kappa(mat)
        rows.append(
            {
                "metric": metric,
                "percent_agreement": pa,
                "kappa": kappa,
                "P_bar": P_bar,
                "P_e": P_e,
            }
        )
    return pd.DataFrame(rows)


def compute_human_llm_correlations_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations between mean human labels and LLM scores.
    """
    rows = []
    for metric in METRIC_NAMES:
        human_col = f"human_mean_{metric}"
        llm_col = f"LLM_{metric}"
        x = df[human_col].values.astype(float)
        y = df[llm_col].values.astype(float)
        if np.all(np.isnan(y)):
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rows.append({"metric": metric, "corr_mean_human_vs_LLM": float(r)})
    return pd.DataFrame(rows)


def compute_majority_vote_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add majority-vote binary label columns (0/1) for each metric.
    Majority = at least 2 of 3 annotators label 1.
    """
    df = df.copy()
    for metric in METRIC_NAMES:
        per_annotator = df[f"{metric}_per_annotator"]
        majority_labels = []
        for row in per_annotator:
            arr = np.array(row, dtype=int)
            majority = 1 if arr.sum() >= 2 else 0
            majority_labels.append(majority)
        df[f"human_majority_{metric}"] = majority_labels
    return df


def compute_human_llm_correlations_majority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations between majority-vote human labels and LLM scores.
    """
    rows = []
    for metric in METRIC_NAMES:
        human_col = f"human_majority_{metric}"
        llm_col = f"LLM_{metric}"
        x = df[human_col].values.astype(float)
        y = df[llm_col].values.astype(float)
        if np.all(np.isnan(y)):
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rows.append({"metric": metric, "corr_majority_human_vs_LLM": float(r)})
    return pd.DataFrame(rows)


def main(json_path: str) -> None:
    print(f"Loading tasks from: {json_path}")
    tasks = load_tasks(json_path)
    df, annotators = build_item_table(tasks)

    print("\nAnnotators (sorted):")
    for a in annotators:
        print("  ", a)

    print(f"\nNumber of (question, model) items with complete annotations: {len(df)}")

    # Inter-coder reliability
    print("\n=== Inter-coder Reliability (Humans Only) ===")
    icr_df = compute_intercoder_reliability(df)
    print(icr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Human–LLM correlations (mean)
    print("\n=== Human–LLM Correlations (Mean Human Label) ===")
    corr_mean_df = compute_human_llm_correlations_mean(df)
    print(corr_mean_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Majority vote and Human–LLM correlations (majority)
    df_majority = compute_majority_vote_labels(df)
    print("\n=== Human–LLM Correlations (Majority-vote Human Label) ===")
    corr_maj_df = compute_human_llm_correlations_majority(df_majority)
    print(corr_maj_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Optional: save tables to CSV
    icr_df.to_csv("intercoder_reliability.csv", index=False)
    corr_mean_df.to_csv("human_llm_corr_mean.csv", index=False)
    corr_maj_df.to_csv("human_llm_corr_majority.csv", index=False)
    print("\nResults saved to:")
    print("  intercoder_reliability.csv")
    print("  human_llm_corr_mean.csv")
    print("  human_llm_corr_majority.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python annotation_analysis.py path/to/export.json")
        sys.exit(1)
    main(sys.argv[1])
