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


METRIC_NAMES = ["Analogy", "Metaphor", "Humor", "Connection", "Scaffolding"]

# Mapping from METRIC_NAMES to CSV column names for v2 scores (with fallbacks)
V2_SCORE_COLUMNS = {
    "Analogy": ["analogy_v2_score", "analogy_v6_score", "analogy_v8_score", "analogy_score"],
    "Metaphor": ["metaphor_v2_score", "metaphor_v6_score", "metaphor_v8_score", "metaphor_v12_score", "metaphor_v11_score", "metaphor_v10_score", "metaphor_v9_score", "metaphor_v7_score", "metaphor_v5_score", "metaphor_v4_score", "metaphor_v3_score", "metaphor_score"],
    "Humor": ["humor_v5_score", "humor_v4_score", "humor_v2_score", "humor_v6_score", "humor_v8_score", "humor_score"],
    "Connection": ["connection_to_everyday_life_v4_score", "connection_to_everyday_life_v3_score", "connection_to_everyday_life_v2_score", "connection_v6_score", "connection_v8_score", "connection_score"],
    "Scaffolding": ["scaffolding_score", "scaffolding_v2_score", "scaffolding_v6_score", "scaffolding_v8_score"],
}


def find_score_column(df: pd.DataFrame, metric: str) -> str:
    """Find the first matching score column for a metric."""
    candidates = V2_SCORE_COLUMNS.get(metric, [])
    for col in candidates:
        if col in df.columns:
            return col
    return None


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


def extract_metrics_from_result(result: List[Dict[str, Any]], active_metrics: List[str] = None) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Extract binary metrics and per-metric reasoning from a single annotation's `result` field.
    Returns a tuple of (metrics dict, per-metric reasoning dict).
    
    Handles three formats:
    1. Multi-choice: choices like ["Metaphor", "Analogy", ...] with from_name="annotation"
    2. Yes/No global: choices like ["Yes"] or ["No"] for single-metric labeling (legacy)
    3. Yes/No per-metric: choices like ["Yes"] with from_name="humor", "connection", etc.
    
    Args:
        result: The annotation result field
        active_metrics: For Yes/No format, which metrics to set. Defaults to all METRIC_NAMES.
    """
    if active_metrics is None:
        active_metrics = METRIC_NAMES
    
    metrics = {m: 0 for m in METRIC_NAMES}
    reasoning = {m: "" for m in METRIC_NAMES}
    
    # Mapping from from_name patterns to metric names
    reasoning_field_map = {
        "reasoning": None,  # Generic reasoning field (legacy)
        "humor_reasoning": "Humor",
        "connection_reasoning": "Connection",
        "analogy_reasoning": "Analogy",
        "metaphor_reasoning": "Metaphor",
        "scaffolding_reasoning": "Scaffolding",
    }
    
    # Mapping from choice from_name to metric names (for per-metric Yes/No format)
    choice_field_map = {
        "humor": "Humor",
        "connection": "Connection",
        "analogy": "Analogy",
        "metaphor": "Metaphor",
        "scaffolding": "Scaffolding",
    }
    
    for r in result:
        if r.get("type") == "choices":
            from_name = r.get("from_name", "")
            choices = r.get("value", {}).get("choices", [])
            
            # Check if this is a per-metric Yes/No field
            if from_name in choice_field_map:
                metric = choice_field_map[from_name]
                for c in choices:
                    if c == "Yes":
                        metrics[metric] = 1
                    elif c == "No":
                        metrics[metric] = 0
            else:
                # Handle global Yes/No or multi-choice format
                for c in choices:
                    # Handle Yes/No format (global - applies to all active metrics)
                    if c == "Yes":
                        for m in active_metrics:
                            metrics[m] = 1
                    elif c == "No":
                        for m in active_metrics:
                            metrics[m] = 0
                    # Handle multi-choice format (original)
                    elif c == "Analogy":
                        metrics["Analogy"] = 1
                    elif c == "Metaphor":
                        metrics["Metaphor"] = 1
                    elif c == "Humor":
                        metrics["Humor"] = 1
                    elif c == "Connection to everyday life":
                        metrics["Connection"] = 1
                    elif c == "Scaffolding":
                        metrics["Scaffolding"] = 1
        elif r.get("type") == "textarea":
            from_name = r.get("from_name", "")
            text_list = r.get("value", {}).get("text", [])
            text = ". ".join(text_list) if text_list else ""
            
            if from_name in reasoning_field_map:
                metric = reasoning_field_map[from_name]
                if metric is None:
                    # Generic reasoning - apply to all active metrics
                    for m in active_metrics:
                        if not reasoning[m]:  # Don't overwrite specific reasoning
                            reasoning[m] = text
                else:
                    reasoning[metric] = text
    
    return metrics, reasoning


def build_item_table(
    tasks: List[Dict[str, Any]], 
    v2_scores_df: pd.DataFrame,
    active_metrics: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a DataFrame with one row per (Index, model) where all annotators are present.
    Uses v2 scores from the provided DataFrame instead of from JSON.
    
    Args:
        tasks: Label Studio tasks with annotations
        v2_scores_df: DataFrame with v2 metric scores
        active_metrics: For Yes/No format labeling, which metrics to set. 
                       Defaults to all METRIC_NAMES.
    """
    if active_metrics is None:
        active_metrics = METRIC_NAMES
        
    annotators = extract_annotators(tasks)
    rows = []
    
    # Determine if we're using Index or question-based matching
    use_index = "Index" in v2_scores_df.columns
    if not use_index:
        print("  Note: No 'Index' column found, using 'question' for matching")

    for t in tasks:
        data = t["data"]
        # Try Index first, fall back to question
        if "Index" in data:
            qid = data["Index"]
        else:
            qid = data.get("question", "")
        
        model = "human"

        # Latest annotation per annotator
        latest: Dict[str, Dict[str, Any]] = {}
        for ann in t.get("annotations", []):
            email = ann["completed_by"]["email"]
            ts = ann["updated_at"]
            if email not in latest or ts > latest[email]["ts"]:
                metrics, reasoning = extract_metrics_from_result(ann.get("result", []), active_metrics)
                latest[email] = {
                    "ts": ts,
                    "metrics": metrics,
                    "reasoning": reasoning,  # Now a dict: {metric: reasoning_text}
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
            # Store per-annotator per-metric reasoning
            row[f"{metric}_reasoning_per_annotator"] = [
                latest[email]["reasoning"].get(metric, "") for email in annotators
            ]

        # Store combined reasoning (legacy, for backward compatibility)
        all_reasoning = []
        for email in annotators:
            for metric in active_metrics:
                reason = latest[email]["reasoning"].get(metric, "")
                if reason:
                    all_reasoning.append(reason)
        row["reasoning"] = ". ".join(all_reasoning) if all_reasoning else ""

        # Human mean metrics
        for metric in METRIC_NAMES:
            arr = np.array(row[f"{metric}_per_annotator"], dtype=float)
            row[f"human_mean_{metric}"] = float(arr.mean())

        # Get v2 LLM scores from the CSV DataFrame
        if use_index:
            v2_row = v2_scores_df[v2_scores_df["Index"] == qid]
        else:
            v2_row = v2_scores_df[v2_scores_df["question"] == qid]
        
        if len(v2_row) == 0:
            print(f"  Warning: No v2 scores found for qid={str(qid)[:50]}...")
            continue
            
        v2_row = v2_row.iloc[0]
        
        for metric in METRIC_NAMES:
            score_col = find_score_column(v2_scores_df, metric)
            if score_col:
                row[f"LLM_{metric}"] = v2_row.get(score_col, np.nan)
            else:
                row[f"LLM_{metric}"] = np.nan

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
    df: pd.DataFrame, label_type: str = "mean", threshold: float = 0.5
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
            binary_agreement = np.nan
        else:
            # Binarize both human and LLM for correlation calculations
            human_binary = (x_valid >= 0.5).astype(int)
            llm_binary = (y_valid >= threshold).astype(int)
            
            # Check if binarized data has variance (needed for correlations)
            if np.std(human_binary) == 0 or np.std(llm_binary) == 0:
                pearson = np.nan
                spearman = np.nan
                kendall = np.nan
            else:
                pearson = np.corrcoef(human_binary, llm_binary)[0, 1]
                spearman, _ = stats.spearmanr(human_binary, llm_binary)
                kendall, _ = stats.kendalltau(human_binary, llm_binary)
            
            # Compute binary agreement (flat accuracy)
            binary_agreement = (human_binary == llm_binary).mean()

        rows.append({
            "metric": metric,
            f"pearson_{label_type}": float(pearson),
            f"spearman_{label_type}": float(spearman),
            f"kendall_{label_type}": float(kendall),
            f"binary_agreement_{label_type}": float(binary_agreement),
        })
    return pd.DataFrame(rows)


def compute_majority_vote_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add majority-vote binary label columns. Ties are broken to 0."""
    df = df.copy()
    for metric in METRIC_NAMES:
        per_annotator = df[f"{metric}_per_annotator"]
        majority_labels = []
        for row in per_annotator:
            arr = np.array(row, dtype=int)
            # Majority requires strictly more than half; ties go to 0
            majority = 1 if arr.sum() > len(arr) / 2 else 0
            majority_labels.append(majority)
        df[f"human_majority_{metric}"] = majority_labels
    return df


def _plot_intercoder_reliability_on_ax(
    ax: plt.Axes, 
    icr_df: pd.DataFrame,
    title: str = "Inter-coder Reliability: Human Annotator Agreement (v2)",
    fontsize: int = 12,
    rotation: int = 0
) -> None:
    """Plot intercoder reliability bar chart on given axes."""
    metrics = icr_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    pa_vals = icr_df["percent_agreement"].values
    ac1_vals = icr_df["ac1"].values

    bars1 = ax.bar(x - width / 2, pa_vals, width, label="Percent Agreement", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, ac1_vals, width, label="Gwet's AC1", color="#55A868")

    ax.set_xlabel("Metric", fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=fontsize - 1, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=fontsize - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize - 3)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize - 3)


def plot_intercoder_reliability(icr_df: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart showing percent agreement and Gwet's AC1."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_intercoder_reliability_on_ax(ax, icr_df)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _plot_confusion_matrix_on_ax(
    ax: plt.Axes,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    metric_name: str,
    row_axis_label: str,
    col_axis_label: str,
    cmap: str = "Blues",
    fontsize: int = 11,
    tick_labels: Tuple[str, str] = ("Absent", "Present"),
    pct_decimals: int = 1
) -> float:
    """
    Plot a single confusion matrix on the given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        row_labels: Binary labels for rows (0/1)
        col_labels: Binary labels for columns (0/1)
        metric_name: Name of the metric for title
        row_axis_label: Label for y-axis
        col_axis_label: Label for x-axis
        cmap: Colormap name
        fontsize: Font size for cell annotations
        tick_labels: Labels for tick marks (default: "Absent", "Present")
        pct_decimals: Decimal places for percentage display
        
    Returns:
        Agreement rate (accuracy)
    """
    # Compute confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for r, c in zip(row_labels, col_labels):
        cm[r, c] += 1
    
    total = cm.sum()
    agreement = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0
    
    # Plot
    ax.imshow(cm, cmap=cmap, aspect="auto")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100 if total > 0 else 0
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.{pct_decimals}f}%)", 
                   ha="center", va="center", color=text_color, fontsize=fontsize)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(list(tick_labels), fontsize=fontsize - 1)
    ax.set_yticklabels(list(tick_labels), fontsize=fontsize - 1)
    ax.set_xlabel(col_axis_label, fontsize=fontsize)
    ax.set_ylabel(row_axis_label, fontsize=fontsize)
    ax.set_title(f"{metric_name}\n({agreement:.{pct_decimals}%})", fontsize=fontsize + 1, fontweight="bold")
    
    return agreement


def _plot_confusion_matrices_on_axes(
    axes: List[plt.Axes],
    df: pd.DataFrame,
    mode: str = "llm_vs_human",
    annotators: List[str] = None,
    threshold: float = 0.5,
    fontsize: int = 11,
    tick_labels: Tuple[str, str] = ("Absent", "Present"),
    pct_decimals: int = 1,
    show_first_ylabel_only: bool = False
) -> None:
    """
    Plot confusion matrices on provided axes.
    
    Args:
        axes: List of matplotlib axes (one per metric)
        df: DataFrame with metric data
        mode: "llm_vs_human" or "intercoder"
        annotators: List of annotator names (required for intercoder mode)
        threshold: Threshold for binarizing LLM scores (llm_vs_human mode only)
        fontsize: Base font size
        tick_labels: Labels for tick marks
        pct_decimals: Decimal places for percentage display
        show_first_ylabel_only: If True, only show ylabel on first subplot
    """
    # Configure based on mode
    if mode == "intercoder":
        cmap = "Greens"
        short_names = []
        for a in (annotators or []):
            if "@" in a:
                short_names.append(a.split("@")[0][:10])
            else:
                short_names.append(a[:10])
        row_label = f"Coder 1 ({short_names[0]})" if short_names else "Coder 1"
        col_label = f"Coder 2 ({short_names[1]})" if len(short_names) > 1 else "Coder 2"
    else:  # llm_vs_human
        cmap = "Blues"
        row_label = "Human Label"
        col_label = "LLM Prediction"
    
    for idx, metric in enumerate(METRIC_NAMES):
        ax = axes[idx]
        
        if mode == "intercoder":
            per_annotator = df[f"{metric}_per_annotator"].tolist()
            row_labels = np.array([row[0] for row in per_annotator], dtype=int)
            col_labels = np.array([row[1] for row in per_annotator], dtype=int)
        else:  # llm_vs_human
            human_col = f"human_mean_{metric}"
            llm_col = f"LLM_{metric}"
            mask = ~(df[human_col].isna() | df[llm_col].isna())
            human_vals = df.loc[mask, human_col].values
            llm_vals = df.loc[mask, llm_col].values
            row_labels = (human_vals >= 0.5).astype(int)
            col_labels = (llm_vals >= threshold).astype(int)
        
        ylabel = row_label if (not show_first_ylabel_only or idx == 0) else ""
        _plot_confusion_matrix_on_ax(
            ax, row_labels, col_labels, metric,
            ylabel, col_label, cmap=cmap, fontsize=fontsize,
            tick_labels=tick_labels, pct_decimals=pct_decimals
        )


def plot_confusion_matrices_grid(
    df: pd.DataFrame,
    output_path: Path,
    mode: str = "llm_vs_human",
    annotators: List[str] = None,
    threshold: float = 0.5
) -> None:
    """
    Create confusion matrices for all metrics.
    
    Args:
        df: DataFrame with metric data
        output_path: Path to save the plot
        mode: "llm_vs_human" or "intercoder"
        annotators: List of annotator names (required for intercoder mode)
        threshold: Threshold for binarizing LLM scores (llm_vs_human mode only)
    """
    n_metrics = len(METRIC_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()
    
    # Configure title based on mode
    if mode == "intercoder":
        title = "Inter-coder Agreement - Confusion Matrices"
    else:
        title = f"LLM vs Human Agreement - Confusion Matrices\n(LLM threshold={threshold})"
    
    _plot_confusion_matrices_on_axes(
        axes_flat, df, mode=mode, annotators=annotators, threshold=threshold
    )
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_intercoder_confusion_matrices(
    df: pd.DataFrame, annotators: List[str], output_path: Path
) -> None:
    """Create confusion matrices comparing agreement between two human coders."""
    plot_confusion_matrices_grid(df, output_path, mode="intercoder", annotators=annotators)


def plot_confusion_matrices(
    df: pd.DataFrame, output_path: Path, threshold: float = 0.5
) -> None:
    """Create confusion matrices comparing LLM predictions vs human labels."""
    plot_confusion_matrices_grid(df, output_path, mode="llm_vs_human", threshold=threshold)


def plot_per_annotator_llm_confusion_matrices(
    df: pd.DataFrame,
    annotators: List[str],
    output_path: Path,
    threshold: float = 0.5
) -> None:
    """
    Create confusion matrices comparing each individual annotator vs LLM predictions.
    
    Args:
        df: DataFrame with {metric}_per_annotator and LLM_{metric} columns
        annotators: List of annotator names/emails
        output_path: Path to save the plot
        threshold: Threshold for binarizing LLM scores
    """
    n_annotators = len(annotators)
    n_metrics = len(METRIC_NAMES)
    
    # Get short names for annotators
    short_names = []
    for a in annotators:
        if "@" in a:
            short_names.append(a.split("@")[0])
        else:
            short_names.append(a[:15])
    
    fig, axes = plt.subplots(n_annotators, n_metrics, figsize=(3 * n_metrics, 3.5 * n_annotators))
    
    # Handle single annotator case
    if n_annotators == 1:
        axes = axes.reshape(1, -1)
    
    for ann_idx, (annotator, short_name) in enumerate(zip(annotators, short_names)):
        for metric_idx, metric in enumerate(METRIC_NAMES):
            ax = axes[ann_idx, metric_idx]
            
            # Get annotator's labels
            per_annotator = df[f"{metric}_per_annotator"].tolist()
            annotator_labels = np.array([row[ann_idx] for row in per_annotator], dtype=int)
            
            # Get LLM predictions
            llm_col = f"LLM_{metric}"
            mask = ~df[llm_col].isna()
            llm_vals = df.loc[mask, llm_col].values
            llm_binary = (llm_vals >= threshold).astype(int)
            annotator_labels_valid = annotator_labels[mask]
            
            # Plot confusion matrix
            _plot_confusion_matrix_on_ax(
                ax, annotator_labels_valid, llm_binary, metric,
                row_axis_label=short_name if metric_idx == 0 else "",
                col_axis_label="LLM" if ann_idx == n_annotators - 1 else "",
                cmap="Purples", fontsize=9,
                tick_labels=("0", "1"), pct_decimals=0
            )
            
            # Only show metric name in title for first row
            if ann_idx > 0:
                # Get agreement from title and update
                title_text = ax.get_title()
                # Extract just the percentage part
                pct_part = title_text.split('\n')[-1] if '\n' in title_text else title_text
                ax.set_title(pct_part, fontsize=10, fontweight="bold")
    
    fig.suptitle(f"Per-Annotator LLM Agreement (threshold={threshold})", 
                 fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def _plot_human_llm_correlations_on_ax(
    ax: plt.Axes,
    corr_mean_df: pd.DataFrame,
    corr_maj_df: pd.DataFrame,
    title: str = "Human–LLM Agreement (Spearman ρ) - v2 Metrics",
    fontsize: int = 12,
    rotation: int = 0
) -> None:
    """Plot human-LLM correlations bar chart on given axes."""
    metrics = corr_mean_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.25

    spearman_mean = corr_mean_df["spearman_mean"].values
    spearman_majority = corr_maj_df["spearman_majority"].values
    binary_agreement_majority = corr_maj_df["binary_agreement_majority"].values

    bars1 = ax.bar(x - width, spearman_mean, width, label="Spearman (Mean)", color="#4C72B0")
    bars2 = ax.bar(x, spearman_majority, width, label="Spearman (Majority)", color="#55A868")
    bars3 = ax.bar(x + width, binary_agreement_majority, width, label="Binary Agreement", color="#C44E52")

    ax.set_xlabel("Metric", fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=fontsize - 1, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(-0.3, 1.15)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=fontsize - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            va = "bottom" if height >= 0 else "top"
            offset = 2 if height >= 0 else -2
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points", ha="center", va=va, fontsize=fontsize - 3)


def plot_human_llm_correlations(
    corr_mean_df: pd.DataFrame, corr_maj_df: pd.DataFrame, output_path: Path
) -> None:
    """Create a grouped bar chart showing Spearman correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_human_llm_correlations_on_ax(ax, corr_mean_df, corr_maj_df)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_combined_analysis(
    df: pd.DataFrame,
    annotators: List[str],
    icr_df: pd.DataFrame,
    corr_mean_df: pd.DataFrame,
    corr_maj_df: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.5
) -> None:
    """
    Create a combined figure with all analysis plots.
    
    Layout:
    - Row 0: Intercoder reliability bar chart | Human-LLM correlations bar chart
    - Row 1: Inter-coder confusion matrices (5 metrics)
    - Row 2: LLM vs Human (mean) confusion matrices (5 metrics)
    - Rows 3+: Per-annotator LLM confusion matrices (one row per annotator)
    """
    n_annotators = len(annotators)
    n_rows = 3 + n_annotators  # bar charts + intercoder + mean LLM + per-annotator rows
    
    # Get short names for annotators
    short_names = []
    for a in annotators:
        if "@" in a:
            short_names.append(a.split("@")[0])
        else:
            short_names.append(a[:15])
    
    fig = plt.figure(figsize=(18, 4 + 3.2 * n_rows))
    
    # Use GridSpec for flexible layout with space for titles
    height_ratios = [1.2, 1, 1] + [1] * n_annotators
    gs = fig.add_gridspec(n_rows, 5, height_ratios=height_ratios, hspace=0.5, wspace=0.3)
    
    # === Row 0: Bar charts (spanning columns) ===
    ax_icr = fig.add_subplot(gs[0, :2])
    ax_corr = fig.add_subplot(gs[0, 3:])
    
    _plot_intercoder_reliability_on_ax(
        ax_icr, icr_df, title="A) Inter-coder Reliability", fontsize=10, rotation=15
    )
    _plot_human_llm_correlations_on_ax(
        ax_corr, corr_mean_df, corr_maj_df, title="B) Human–LLM Correlation", fontsize=10, rotation=15
    )
    
    # === Row 1: Inter-coder confusion matrices ===
    intercoder_axes = [fig.add_subplot(gs[1, idx]) for idx in range(len(METRIC_NAMES))]
    _plot_confusion_matrices_on_axes(
        intercoder_axes, df, mode="intercoder", annotators=annotators,
        fontsize=9, tick_labels=("0", "1"), pct_decimals=0, show_first_ylabel_only=True
    )
    # Add title above the middle axis
    intercoder_axes[2].annotate(
        "C) Inter-coder Agreement Matrices", xy=(0.5, 1.25), xycoords="axes fraction",
        ha="center", fontsize=12, fontweight="bold"
    )
    
    # === Row 2: LLM vs Human (mean) confusion matrices ===
    llm_axes = [fig.add_subplot(gs[2, idx]) for idx in range(len(METRIC_NAMES))]
    _plot_confusion_matrices_on_axes(
        llm_axes, df, mode="llm_vs_human", threshold=threshold,
        fontsize=9, tick_labels=("0", "1"), pct_decimals=0, show_first_ylabel_only=True
    )
    llm_axes[2].annotate(
        f"D) LLM vs Mean Human (threshold={threshold})", xy=(0.5, 1.25), xycoords="axes fraction",
        ha="center", fontsize=12, fontweight="bold"
    )
    
    # === Rows 3+: Per-annotator LLM confusion matrices ===
    first_per_annotator_axes = None
    for ann_idx, short_name in enumerate(short_names):
        row_idx = 3 + ann_idx
        axes = [fig.add_subplot(gs[row_idx, metric_idx]) for metric_idx in range(len(METRIC_NAMES))]
        
        if ann_idx == 0:
            first_per_annotator_axes = axes
        
        for metric_idx, metric in enumerate(METRIC_NAMES):
            ax = axes[metric_idx]
            
            # Get annotator's labels
            per_annotator = df[f"{metric}_per_annotator"].tolist()
            annotator_labels = np.array([row[ann_idx] for row in per_annotator], dtype=int)
            
            # Get LLM predictions
            llm_col = f"LLM_{metric}"
            mask = ~df[llm_col].isna()
            llm_vals = df.loc[mask, llm_col].values
            llm_binary = (llm_vals >= threshold).astype(int)
            annotator_labels_valid = annotator_labels[mask]
            
            _plot_confusion_matrix_on_ax(
                ax, annotator_labels_valid, llm_binary, metric,
                row_axis_label=short_name if metric_idx == 0 else "",
                col_axis_label="LLM" if ann_idx == n_annotators - 1 else "",
                cmap="Purples", fontsize=9,
                tick_labels=("0", "1"), pct_decimals=0
            )
            
            # Only show metric name for first per-annotator row
            if ann_idx > 0:
                title_text = ax.get_title()
                pct_part = title_text.split('\n')[-1] if '\n' in title_text else title_text
                ax.set_title(pct_part, fontsize=10, fontweight="bold")
    
    # Add section title above first per-annotator row
    if first_per_annotator_axes:
        first_per_annotator_axes[2].annotate(
            f"E) Per-Annotator LLM Agreement (threshold={threshold})", 
            xy=(0.5, 1.25), xycoords="axes fraction",
            ha="center", fontsize=12, fontweight="bold"
        )
    
    # Adjust layout and save
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def add_annotations_to_csv(
    csv_path: str,
    annotations_df: pd.DataFrame,
    active_metrics: List[str],
    annotators: List[str],
    metric_version: str = "v6"
) -> None:
    """
    Add human annotation columns to the original CSV file.
    
    Args:
        csv_path: Path to the CSV file to update
        annotations_df: DataFrame with annotation data (from build_item_table)
        active_metrics: List of metrics that were annotated
        annotators: List of annotator emails
        metric_version: Version string for column naming (e.g., "v6")
    """
    csv_df = pd.read_csv(csv_path)
    
    # Determine if we're matching by Index or question
    use_index = "Index" in csv_df.columns
    match_col = "Index" if use_index else "question"
    
    # Get short annotator names for column headers (e.g., "john.smith" from "john.smith@gmail.com")
    short_names = []
    for a in annotators:
        if "@" in a:
            short_names.append(a.split("@")[0])
        else:
            short_names.append(a[:15])
    
    added_cols = []
    
    # Add reasoning column (legacy combined)
    reasoning_col = f"reasoning_{metric_version}"
    csv_df[reasoning_col] = ""
    added_cols.append(reasoning_col)
    
    # Add annotation columns for each active metric
    for metric in active_metrics:
        metric_lower = metric.lower()
        
        # Add per-annotator score columns
        for short_name in short_names:
            col_name = f"{short_name}_{metric_lower}_{metric_version}"
            csv_df[col_name] = np.nan
            added_cols.append(col_name)
        
        # Add per-annotator reasoning columns for this metric
        for short_name in short_names:
            reason_col_name = f"{short_name}_{metric_lower}_{metric_version}_reason"
            csv_df[reason_col_name] = ""
            added_cols.append(reason_col_name)
    
    # Fill in values from annotations_df
    for _, row in annotations_df.iterrows():
        qid = row["qid"]
        mask = csv_df[match_col] == qid
        
        if mask.any():
            # Add combined reasoning (legacy)
            csv_df.loc[mask, reasoning_col] = row.get("reasoning", "")
            
            # Add per-annotator scores and reasoning for each metric
            for metric in active_metrics:
                metric_lower = metric.lower()
                per_annotator = row[f"{metric}_per_annotator"]
                per_annotator_reasoning = row.get(f"{metric}_reasoning_per_annotator", [""] * len(short_names))
                
                for i, short_name in enumerate(short_names):
                    # Score column
                    col_name = f"{short_name}_{metric_lower}_{metric_version}"
                    csv_df.loc[mask, col_name] = per_annotator[i]
                    
                    # Reasoning column
                    reason_col_name = f"{short_name}_{metric_lower}_{metric_version}_reason"
                    if i < len(per_annotator_reasoning):
                        csv_df.loc[mask, reason_col_name] = per_annotator_reasoning[i]
    
    # Save updated CSV
    csv_df.to_csv(csv_path, index=False)
    
    print(f"  Added columns: {added_cols}")
    print(f"  Updated: {csv_path}")


def main(v2_csv_path: str, json_path: str, active_metrics: List[str] = None, metric_version: str = "v6") -> None:
    """
    Main analysis function.
    
    Args:
        v2_csv_path: Path to balanced_dataset_v2.csv with v2 metric scores
        json_path: Path to labelstudio_output.json with human annotations
        active_metrics: For Yes/No format labeling, which metrics to analyze.
                       Defaults to all METRIC_NAMES.
        metric_version: Version string for column naming (e.g., "v6")
    """
    if active_metrics is None:
        active_metrics = METRIC_NAMES
        
    print(f"Loading v2 scores from: {v2_csv_path}")
    v2_df = pd.read_csv(v2_csv_path)
    
    print(f"Loading human annotations from: {json_path}")
    tasks = load_tasks(json_path)
    
    df, annotators = build_item_table(tasks, v2_df, active_metrics)

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
    
    # Add annotation columns to the original CSV
    print("\nAdding annotation columns to CSV...")
    add_annotations_to_csv(v2_csv_path, df, active_metrics, annotators, metric_version)

    # Generate combined plot with all analyses
    if len(annotators) >= 2:
        print("\nGenerating plot...")
        combined_plot_path = output_dir / "combined_analysis.png"
        plot_combined_analysis(df, annotators, icr_df, corr_mean_df, corr_maj_df, combined_plot_path)
    else:
        print(f"\nSkipping combined plot (requires at least 2 annotators, found {len(annotators)})")
        # Generate simpler LLM correlation plot only
        print("Generating LLM correlation plot...")
        corr_plot_path = output_dir / "human_llm_correlations.png"
        plot_human_llm_correlations(corr_mean_df, corr_maj_df, corr_plot_path)


if __name__ == "__main__":
    import argparse
    
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(
        description="Inter-coder reliability analysis for v2 metrics."
    )
    parser.add_argument(
        "v2_csv", 
        nargs="?",
        default=str(script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv"),
        help="Path to CSV with v2 metric scores"
    )
    parser.add_argument(
        "json_file",
        nargs="?", 
        default=str(script_dir / "balanced_dataset" / "labelstudio_output.json"),
        help="Path to Label Studio JSON export"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=None,
        help="Active metrics for Yes/No format labeling (default: all)"
    )
    parser.add_argument(
        "--version",
        default="v6",
        help="Metric version for column naming, e.g., 'v6' creates 'john.smith_metaphor_v6' (default: v6)"
    )
    
    args = parser.parse_args()
    
    main(args.v2_csv, args.json_file, args.metrics, args.version)

