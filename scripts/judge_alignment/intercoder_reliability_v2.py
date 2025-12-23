#!/usr/bin/env python3
"""
Inter-coder reliability analysis for v2 metrics.

This script analyzes agreement between human annotators and compares 
human labels with LLM-generated v2 scores from the balanced_dataset_v2.csv.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


METRIC_NAMES = ["Analogy", "Metaphor", "Humor", "Connection"]

# Mapping from METRIC_NAMES to CSV column names for v2 scores
V2_SCORE_COLUMNS = {
    "Analogy": "analogy_v2_score",
    "Metaphor": "metaphor_v2_score",
    "Humor": "humor_v2_score",
    "Connection": "connection_to_everyday_life_v2_score",
}


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


def build_item_table(
    tasks: List[Dict[str, Any]], 
    v2_scores_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a DataFrame with one row per (question_id, model) where all annotators are present.
    Uses v2 scores from the provided DataFrame instead of from JSON.
    """
    annotators = extract_annotators(tasks)
    rows = []

    for t in tasks:
        qid = t["data"]["question_id"]
        model = t["data"]["model"]

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

        # Get v2 LLM scores from the CSV DataFrame
        v2_row = v2_scores_df[
            (v2_scores_df["question_id"] == qid) & 
            (v2_scores_df["model"] == model)
        ]
        
        if len(v2_row) == 0:
            print(f"  Warning: No v2 scores found for qid={qid}, model={model}")
            continue
            
        v2_row = v2_row.iloc[0]
        
        for metric in METRIC_NAMES:
            score_col = V2_SCORE_COLUMNS[metric]
            row[f"LLM_{metric}"] = v2_row.get(score_col, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, annotators


def percent_agreement(matrix: List[List[int]]) -> float:
    """Compute percent agreement across coders."""
    arr = np.array(matrix)
    agree = [len(set(row)) == 1 for row in arr]
    return float(np.mean(agree))


def fleiss_kappa(matrix: List[List[int]]) -> Tuple[float, float, float]:
    """Compute Fleiss' kappa for binary labels."""
    X = np.array(matrix)
    N, n = X.shape
    n0 = (X == 0).sum(axis=1)
    n1 = (X == 1).sum(axis=1)

    P_i = (n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1))
    P_bar = P_i.mean()

    p0 = n0.sum() / (N * n)
    p1 = n1.sum() / (N * n)
    P_e = p0 ** 2 + p1 ** 2

    if (1 - P_e) == 0:
        kappa = np.nan
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    return float(kappa), float(P_bar), float(P_e)


def gwet_ac1(matrix: List[List[int]]) -> Tuple[float, float, float]:
    """Compute Gwet's AC1 for binary labels."""
    X = np.array(matrix)
    N, n = X.shape
    n0 = (X == 0).sum(axis=1)
    n1 = (X == 1).sum(axis=1)

    P_i = (n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1))
    P_a = P_i.mean()

    p0 = n0.sum() / (N * n)
    p1 = n1.sum() / (N * n)
    P_e = 2 * p0 * p1

    if (1 - P_e) == 0:
        ac1 = np.nan
    else:
        ac1 = (P_a - P_e) / (1 - P_e)

    return float(ac1), float(P_a), float(P_e)


def compute_intercoder_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percent agreement, Fleiss' kappa, and Gwet's AC1 for each metric."""
    rows = []
    for metric in METRIC_NAMES:
        mat = df[f"{metric}_per_annotator"].tolist()
        pa = percent_agreement(mat)
        kappa, P_bar, P_e_kappa = fleiss_kappa(mat)
        ac1, _, P_e_ac1 = gwet_ac1(mat)
        rows.append({
            "metric": metric,
            "percent_agreement": pa,
            "kappa": kappa,
            "ac1": ac1,
            "P_e_kappa": P_e_kappa,
            "P_e_ac1": P_e_ac1,
        })
    return pd.DataFrame(rows)


def compute_human_llm_correlations(
    df: pd.DataFrame, label_type: str = "mean"
) -> pd.DataFrame:
    """Compute correlations between human labels and LLM v2 scores."""
    rows = []
    for metric in METRIC_NAMES:
        human_col = f"human_{label_type}_{metric}"
        llm_col = f"LLM_{metric}"
        x = df[human_col].values.astype(float)
        y = df[llm_col].values.astype(float)

        mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 2 or np.std(x_valid) == 0 or np.std(y_valid) == 0:
            pearson = np.nan
            spearman = np.nan
            kendall = np.nan
        else:
            pearson = np.corrcoef(x_valid, y_valid)[0, 1]
            spearman, _ = stats.spearmanr(x_valid, y_valid)
            kendall, _ = stats.kendalltau(x_valid, y_valid)

        rows.append({
            "metric": metric,
            f"pearson_{label_type}": float(pearson),
            f"spearman_{label_type}": float(spearman),
            f"kendall_{label_type}": float(kendall),
        })
    return pd.DataFrame(rows)


def compute_majority_vote_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add majority-vote binary label columns."""
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


def plot_intercoder_reliability(icr_df: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart showing percent agreement and Gwet's AC1."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = icr_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    percent_agreement = icr_df["percent_agreement"].values
    ac1 = icr_df["ac1"].values

    bars1 = ax.bar(x - width / 2, percent_agreement, width, label="Percent Agreement", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, ac1, width, label="Gwet's AC1", color="#55A868")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Inter-coder Reliability: Human Annotator Agreement (v2)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_human_llm_correlations(
    corr_mean_df: pd.DataFrame, corr_maj_df: pd.DataFrame, output_path: Path
) -> None:
    """Create a grouped bar chart showing Spearman correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = corr_mean_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    spearman_mean = corr_mean_df["spearman_mean"].values
    spearman_majority = corr_maj_df["spearman_majority"].values

    bars1 = ax.bar(x - width / 2, spearman_mean, width, label="Mean Human Label", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, spearman_majority, width, label="Majority-vote Label", color="#55A868")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Spearman Correlation", fontsize=12)
    ax.set_title("Human–LLM Agreement (Spearman ρ) - v2 Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            va = "bottom" if height >= 0 else "top"
            offset = 3 if height >= 0 else -3
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points", ha="center", va=va, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def main(v2_csv_path: str, json_path: str) -> None:
    """
    Main analysis function.
    
    Args:
        v2_csv_path: Path to balanced_dataset_v2.csv with v2 metric scores
        json_path: Path to labelstudio_output.json with human annotations
    """
    print(f"Loading v2 scores from: {v2_csv_path}")
    v2_df = pd.read_csv(v2_csv_path)
    
    print(f"Loading human annotations from: {json_path}")
    tasks = load_tasks(json_path)
    
    df, annotators = build_item_table(tasks, v2_df)

    output_dir = Path(v2_csv_path).parent

    print("\nAnnotators (sorted):")
    for a in annotators:
        print("  ", a)

    print(f"\nNumber of items with complete annotations: {len(df)}")

    # Inter-coder reliability (human-only, same as v1)
    print("\n=== Inter-coder Reliability (Humans Only) ===")
    icr_df = compute_intercoder_reliability(df)
    print(icr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Human–LLM correlations (mean)
    print("\n=== Human–LLM v2 Correlations (Mean Human Label) ===")
    corr_mean_df = compute_human_llm_correlations(df, label_type="mean")
    print(corr_mean_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Majority vote and Human–LLM correlations (majority)
    df_majority = compute_majority_vote_labels(df)
    print("\n=== Human–LLM v2 Correlations (Majority-vote Human Label) ===")
    corr_maj_df = compute_human_llm_correlations(df_majority, label_type="majority")
    print(corr_maj_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save tables to CSV
    icr_path = output_dir / "intercoder_reliability.csv"
    corr_mean_path = output_dir / "human_llm_corr_mean.csv"
    corr_maj_path = output_dir / "human_llm_corr_majority.csv"

    icr_df.to_csv(icr_path, index=False)
    corr_mean_df.to_csv(corr_mean_path, index=False)
    corr_maj_df.to_csv(corr_maj_path, index=False)

    print(f"\nResults saved to {output_dir}:")
    print(f"  {icr_path.name}")
    print(f"  {corr_mean_path.name}")
    print(f"  {corr_maj_path.name}")

    # Generate plots
    print("\nGenerating plots...")
    icr_plot_path = output_dir / "intercoder_reliability.png"
    human_llm_plot_path = output_dir / "human_llm_correlations.png"

    plot_intercoder_reliability(icr_df, icr_plot_path)
    plot_human_llm_correlations(corr_mean_df, corr_maj_df, human_llm_plot_path)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    # Default paths
    v2_csv = script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv"
    json_file = script_dir / "balanced_dataset" / "labelstudio_output.json"
    
    if len(sys.argv) == 3:
        v2_csv = Path(sys.argv[1])
        json_file = Path(sys.argv[2])
    
    main(str(v2_csv), str(json_file))

