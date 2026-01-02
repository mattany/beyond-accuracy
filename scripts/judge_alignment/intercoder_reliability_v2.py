#!/usr/bin/env python3
"""
Inter-coder reliability analysis for v2 metrics.

This script analyzes agreement between human annotators and compares 
human labels with LLM-generated v2 scores from the balanced_dataset_v2.csv.

Supports two modes:
1. JSON mode: Load annotations from Label Studio JSON export
2. CSV mode: Load annotations directly from formatted CSV files (--csv-mode)
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


METRIC_NAMES = ["Analogy", "Metaphor", "Humor", "Connection", "Scaffolding"]

# Mapping from metric display names to possible column prefixes
METRIC_TO_COL_PREFIX = {
    "Analogy": ["analogy"],
    "Metaphor": ["metaphor"],
    "Humor": ["humor"],
    "Connection": ["connection", "connection_to_everyday_life", "conn"],
    "Scaffolding": ["scaffolding"],
}

# Mapping from METRIC_NAMES to CSV column names for v2 scores (with fallbacks)
V2_SCORE_COLUMNS = {
    "Analogy": ["analogy_v2_score", "analogy_v6_score", "analogy_v8_score", "analogy_score", "analogy_explicit_score"],
    "Metaphor": ["metaphor_v2_score", "metaphor_v6_score", "metaphor_v8_score", "metaphor_v12_score", "metaphor_v11_score", "metaphor_v10_score", "metaphor_v9_score", "metaphor_v7_score", "metaphor_v5_score", "metaphor_v4_score", "metaphor_v3_score", "metaphor_score", "metaphor_explicit_score"],
    "Humor": ["humor_v5_score", "humor_v4_score", "humor_v2_score", "humor_v6_score", "humor_v8_score", "humor_score", "humor_explicit_score"],
    "Connection": ["connection_to_everyday_life_v4_score", "connection_to_everyday_life_v3_score", "connection_to_everyday_life_v2_score", "connection_v6_score", "connection_v8_score", "connection_score", "connection_to_everyday_life_score"],
    "Scaffolding": ["scaffolding_v2_score", "scaffolding_score", "scaffolding_v6_score", "scaffolding_v8_score"],
}


def find_score_column(df: pd.DataFrame, metric: str) -> str:
    """Find the first matching score column for a metric."""
    candidates = V2_SCORE_COLUMNS.get(metric, [])
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_version_from_column(col_name: str) -> str:
    """
    Extract version string from a score column name.
    
    Examples:
        'humor_v5_score' -> 'v5'
        'connection_to_everyday_life_v4_score' -> 'v4'
        'metaphor_score' -> None (no version)
        'analogy_explicit_score' -> 'v1' (v1 explicit format)
    """
    # Check for v1 "explicit" format first
    if '_explicit_score' in col_name or col_name.endswith('_explicit_score'):
        return 'v1'
    match = re.search(r'_(v\d+)_', col_name)
    if match:
        return match.group(1)
    return None


def find_annotator_columns(df: pd.DataFrame, metric: str) -> Tuple[List[str], List[str], str]:
    """
    Find annotator score and reason columns for a metric in the DataFrame.
    
    Returns:
        Tuple of (score_column_names, reason_column_names, version_string)
        
    Looks for patterns like:
        - mattany_metaphor_v8, nirgrn_metaphor_v8
        - mattany_humor_v5_reason, nirgrn_humor_v5_reason
    """
    prefixes = METRIC_TO_COL_PREFIX.get(metric, [metric.lower()])
    
    score_cols = []
    reason_cols = []
    version = None
    
    for col in df.columns:
        col_lower = col.lower()
        for prefix in prefixes:
            # Match pattern: {annotator}_{prefix}_{version} or {annotator}_{prefix}
            pattern = rf'^([a-z_]+)_({prefix})(?:_(v\d+))?$'
            match = re.match(pattern, col_lower)
            if match:
                annotator = match.group(1)
                found_version = match.group(3)
                score_cols.append(col)
                if found_version:
                    version = found_version
                break
            
            # Match reason pattern: {annotator}_{prefix}_{version}_reason
            reason_pattern = rf'^([a-z_]+)_({prefix})(?:_(v\d+))?_reason$'
            reason_match = re.match(reason_pattern, col_lower)
            if reason_match:
                reason_cols.append(col)
                if reason_match.group(3):
                    version = reason_match.group(3)
                break
    
    return sorted(score_cols), sorted(reason_cols), version


def build_item_table_from_csv(
    csv_paths: List[str],
    active_metrics: List[str] = None
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Build a DataFrame with one row per item, loading annotations directly from CSV files.
    
    Supports multiple CSV files, each potentially containing different metrics.
    
    Args:
        csv_paths: List of paths to formatted CSV files with embedded annotations
        active_metrics: Which metrics to extract. Defaults to all found in the CSVs.
        
    Returns:
        Tuple of (DataFrame, annotator_list, metric_versions_dict)
    """
    if active_metrics is None:
        active_metrics = METRIC_NAMES
    
    # Load all CSVs
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append((path, df))
        print(f"  Loaded {len(df)} rows from {Path(path).name}")
    
    # Determine which metrics are available in which CSV
    metric_to_csv: Dict[str, Tuple[str, pd.DataFrame]] = {}
    metric_versions: Dict[str, str] = {}
    all_annotators = set()
    
    for path, df in dfs:
        for metric in active_metrics:
            score_cols, reason_cols, version = find_annotator_columns(df, metric)
            if score_cols:
                metric_to_csv[metric] = (path, df)
                if version:
                    metric_versions[metric] = version
                # Extract annotator names from columns
                for col in score_cols:
                    # Extract annotator name (everything before the metric prefix)
                    for prefix in METRIC_TO_COL_PREFIX.get(metric, [metric.lower()]):
                        if f"_{prefix}" in col.lower():
                            annotator = col.lower().split(f"_{prefix}")[0]
                            all_annotators.add(annotator)
                            break
    
    annotators = sorted(all_annotators)
    print(f"  Found annotators: {annotators}")
    print(f"  Found metrics: {list(metric_to_csv.keys())}")
    
    # Use the first CSV as the base (for Index/question matching)
    base_path, base_df = dfs[0]
    
    # Determine match column
    if "Index" in base_df.columns:
        match_col = "Index"
    else:
        match_col = "question"
    
    rows = []
    for idx, base_row in base_df.iterrows():
        qid = base_row[match_col]
        row: Dict[str, Any] = {"qid": qid, "model": "human"}
        
        # Process each metric
        for metric in active_metrics:
            if metric not in metric_to_csv:
                # Metric not found in any CSV - set defaults
                row[f"{metric}_per_annotator"] = [0] * len(annotators)
                row[f"{metric}_reasoning_per_annotator"] = [""] * len(annotators)
                row[f"human_mean_{metric}"] = 0.0
                row[f"LLM_{metric}"] = np.nan
                continue
            
            csv_path, csv_df = metric_to_csv[metric]
            
            # Find the matching row in this CSV
            if match_col in csv_df.columns:
                matching_rows = csv_df[csv_df[match_col] == qid]
            else:
                matching_rows = pd.DataFrame()
            
            if len(matching_rows) == 0:
                # Try matching from base_df index
                if idx < len(csv_df):
                    metric_row = csv_df.iloc[idx]
                else:
                    row[f"{metric}_per_annotator"] = [0] * len(annotators)
                    row[f"{metric}_reasoning_per_annotator"] = [""] * len(annotators)
                    row[f"human_mean_{metric}"] = 0.0
                    row[f"LLM_{metric}"] = np.nan
                    continue
            else:
                metric_row = matching_rows.iloc[0]
            
            # Get annotator columns for this metric
            score_cols, reason_cols, version = find_annotator_columns(csv_df, metric)
            
            # Extract per-annotator values
            per_annotator_scores = []
            per_annotator_reasons = []
            
            for annotator in annotators:
                # Find the score column for this annotator
                score_val = 0
                reason_val = ""
                
                for col in score_cols:
                    if col.lower().startswith(annotator):
                        val = metric_row.get(col, np.nan)
                        if pd.notna(val):
                            score_val = int(float(val))
                        break
                
                for col in reason_cols:
                    if col.lower().startswith(annotator):
                        val = metric_row.get(col, "")
                        if pd.notna(val):
                            reason_val = str(val)
                        break
                
                per_annotator_scores.append(score_val)
                per_annotator_reasons.append(reason_val)
            
            row[f"{metric}_per_annotator"] = per_annotator_scores
            row[f"{metric}_reasoning_per_annotator"] = per_annotator_reasons
            row[f"human_mean_{metric}"] = float(np.mean(per_annotator_scores))
            
            # Get LLM score
            llm_score_col = find_score_column(csv_df, metric)
            if llm_score_col and llm_score_col in csv_df.columns:
                row[f"LLM_{metric}"] = metric_row.get(llm_score_col, np.nan)
            else:
                row[f"LLM_{metric}"] = np.nan
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    # Filter to only metrics that were found
    found_metrics = [m for m in active_metrics if m in metric_to_csv]
    
    return result_df, annotators, metric_versions, found_metrics


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


def compute_intercoder_reliability(df: pd.DataFrame, active_metrics: List[str] = None) -> pd.DataFrame:
    """Compute percent agreement, Fleiss' kappa, and Gwet's AC1 for each metric."""
    if active_metrics is None:
        active_metrics = METRIC_NAMES
    
    rows = []
    for metric in active_metrics:
        col_name = f"{metric}_per_annotator"
        if col_name not in df.columns:
            continue
        mat = df[col_name].tolist()
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


def _compute_precision_recall_f1(human_binary: np.ndarray, llm_binary: np.ndarray) -> tuple:
    """Compute precision, recall, and F1 score.
    
    Human labels are ground truth, LLM predictions are being evaluated.
    Positive class = 1 (metric is present).
    """
    tp = ((human_binary == 1) & (llm_binary == 1)).sum()
    fp = ((human_binary == 0) & (llm_binary == 1)).sum()
    fn = ((human_binary == 1) & (llm_binary == 0)).sum()
    tn = ((human_binary == 0) & (llm_binary == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    
    return float(precision), float(recall), float(f1), int(tp), int(fp), int(fn), int(tn)


def compute_human_llm_correlations(
    df: pd.DataFrame, label_type: str = "mean", threshold: float = 0.5,
    active_metrics: List[str] = None
) -> pd.DataFrame:
    """Compute correlations between human labels and LLM v2 scores."""
    if active_metrics is None:
        active_metrics = METRIC_NAMES
    
    rows = []
    for metric in active_metrics:
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
            spearman_excl_ties = np.nan
            kendall = np.nan
            binary_agreement = np.nan
            binary_agreement_excl_ties = np.nan
            precision, recall, f1 = np.nan, np.nan, np.nan
            precision_excl, recall_excl, f1_excl = np.nan, np.nan, np.nan
            tp, fp, fn, tn = 0, 0, 0, 0
            tp_excl, fp_excl, fn_excl, tn_excl = 0, 0, 0, 0
            n_ties = 0
            n_consensus = 0
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
            
            # Compute precision, recall, F1
            precision, recall, f1, tp, fp, fn, tn = _compute_precision_recall_f1(human_binary, llm_binary)
            
            # Compute metrics excluding ties (where human mean == 0.5)
            # Ties occur when annotators disagree (one says 0, one says 1)
            consensus_mask = (x_valid != 0.5)
            n_ties = (~consensus_mask).sum()
            n_consensus = consensus_mask.sum()
            
            # Total positives and negatives in full dataset (for adjusted metrics)
            total_positives = (human_binary == 1).sum()
            total_negatives = (human_binary == 0).sum()
            
            if n_consensus > 0:
                human_consensus = human_binary[consensus_mask]
                llm_consensus = llm_binary[consensus_mask]
                binary_agreement_excl_ties = (human_consensus == llm_consensus).mean()
                # Compute Spearman excluding ties
                if np.std(human_consensus) == 0 or np.std(llm_consensus) == 0:
                    spearman_excl_ties = np.nan
                else:
                    spearman_excl_ties, _ = stats.spearmanr(human_consensus, llm_consensus)
                # Compute precision, recall, F1 excluding ties (on subset)
                precision_excl, recall_excl_subset, f1_excl_subset, tp_excl, fp_excl, fn_excl, tn_excl = \
                    _compute_precision_recall_f1(human_consensus, llm_consensus)
                
                # Compute ADJUSTED recall: TP_consensus / Total_Positives_Full
                # This accounts for positives in ties that we couldn't evaluate
                recall_excl = tp_excl / total_positives if total_positives > 0 else np.nan
                
                # Recompute F1 with adjusted recall
                if precision_excl > 0 and recall_excl > 0:
                    f1_excl = 2 * precision_excl * recall_excl / (precision_excl + recall_excl)
                else:
                    f1_excl = np.nan
            else:
                binary_agreement_excl_ties = np.nan
                spearman_excl_ties = np.nan
                precision_excl, recall_excl, f1_excl = np.nan, np.nan, np.nan
                tp_excl, fp_excl, fn_excl, tn_excl = 0, 0, 0, 0

        rows.append({
            "metric": metric,
            f"pearson_{label_type}": float(pearson),
            f"spearman_{label_type}": float(spearman),
            f"spearman_excl_ties_{label_type}": float(spearman_excl_ties),
            f"kendall_{label_type}": float(kendall),
            f"binary_agreement_{label_type}": float(binary_agreement),
            f"binary_agr_excl_ties_{label_type}": float(binary_agreement_excl_ties),
            f"precision_{label_type}": float(precision),
            f"recall_{label_type}": float(recall),
            f"f1_{label_type}": float(f1),
            f"precision_excl_ties_{label_type}": float(precision_excl),
            f"recall_excl_ties_{label_type}": float(recall_excl),
            f"f1_excl_ties_{label_type}": float(f1_excl),
            f"tp_{label_type}": tp,
            f"fp_{label_type}": fp,
            f"fn_{label_type}": fn,
            f"tn_{label_type}": tn,
            f"n_ties_{label_type}": int(n_ties),
            f"n_consensus_{label_type}": int(n_consensus),
        })
    return pd.DataFrame(rows)


def compute_population_adjusted_metrics(
    sample_df: pd.DataFrame,
    full_dataset_path: str,
    active_metrics: List[str],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute population-adjusted metrics under a stratified sampling design.

    This estimator assumes the *labeled sample* was constructed by stratified
    sampling on the LLM-as-judge prediction (at the same `threshold`):

      - Stratum 1: LLM predicts positive (LLM score >= threshold)
      - Stratum 0: LLM predicts negative (LLM score < threshold)

    A common "balanced" design is 50/50 sampling from these two strata, but
    the estimator below does not require the sample to be exactly balanced;
    it uses the actual sample stratum sizes (n_h) and population stratum sizes
    (N_h) and applies sampling-weight correction (a.k.a. post-stratification):

      weight_h = N_h / n_h

    Weighted confusion counts are then:
      TP_pop = weight_1 * TP_sample
      FP_pop = weight_1 * FP_sample
      FN_pop = weight_0 * FN_sample
      TN_pop = weight_0 * TN_sample

    From those, compute population-level precision/recall/F1.

    Caveat / assumption:
      This is valid when the labeled sample is representative *within each
      prediction stratum* (i.e., the conditional label distributions
      P(Y | Ŷ=1) and P(Y | Ŷ=0) in the sample match the population).
    
    Args:
        sample_df: DataFrame with human annotations (balanced sample)
        full_dataset_path: Path to full dataset CSV with LLM scores
        active_metrics: List of metrics to compute
        threshold: Threshold for binarizing LLM scores
        
    Returns:
        DataFrame with population-adjusted metrics per metric
    """
    full_df = pd.read_csv(full_dataset_path)
    
    rows = []
    for metric in active_metrics:
        # Find score column in full dataset
        score_col = None
        for col_name in V2_SCORE_COLUMNS.get(metric, []):
            if col_name in full_df.columns:
                score_col = col_name
                break
        
        if score_col is None:
            print(f"  Warning: No score column found for {metric} in full dataset")
            continue
        
        # Get full dataset distribution
        full_scores = full_df[score_col].dropna()
        n_full_total = len(full_scores)
        n_full_llm_pos = (full_scores >= threshold).sum()
        n_full_llm_neg = (full_scores < threshold).sum()
        
        # Get sample confusion matrix values
        # Need to compute from sample_df
        human_col = f"human_majority_{metric}"
        llm_col = f"LLM_{metric}"
        
        if human_col not in sample_df.columns or llm_col not in sample_df.columns:
            print(f"  Warning: Missing columns for {metric} in sample")
            continue
            
        mask = ~(sample_df[human_col].isna() | sample_df[llm_col].isna())
        human_vals = sample_df.loc[mask, human_col].values.astype(int)
        llm_vals = (sample_df.loc[mask, llm_col].values >= threshold).astype(int)
        
        # Compute sample confusion matrix
        tp_sample = ((human_vals == 1) & (llm_vals == 1)).sum()
        fp_sample = ((human_vals == 0) & (llm_vals == 1)).sum()
        fn_sample = ((human_vals == 1) & (llm_vals == 0)).sum()
        tn_sample = ((human_vals == 0) & (llm_vals == 0)).sum()
        
        n_sample_llm_pos = tp_sample + fp_sample
        n_sample_llm_neg = fn_sample + tn_sample

        # --- Sampling-weight correction (stratified by LLM prediction) ---
        # weights map sample stratum counts -> population stratum counts
        weight_llm_pos = (n_full_llm_pos / n_sample_llm_pos) if n_sample_llm_pos > 0 else np.nan
        weight_llm_neg = (n_full_llm_neg / n_sample_llm_neg) if n_sample_llm_neg > 0 else np.nan

        # Weighted (estimated population) confusion matrix counts
        est_tp_pop = float(tp_sample) * weight_llm_pos if not np.isnan(weight_llm_pos) else np.nan
        est_fp_pop = float(fp_sample) * weight_llm_pos if not np.isnan(weight_llm_pos) else np.nan
        est_fn_pop = float(fn_sample) * weight_llm_neg if not np.isnan(weight_llm_neg) else np.nan
        est_tn_pop = float(tn_sample) * weight_llm_neg if not np.isnan(weight_llm_neg) else np.nan

        # Population metrics derived from weighted counts
        adj_precision = (
            est_tp_pop / (est_tp_pop + est_fp_pop)
            if (not np.isnan(est_tp_pop) and not np.isnan(est_fp_pop) and (est_tp_pop + est_fp_pop) > 0)
            else np.nan
        )
        adj_recall = (
            est_tp_pop / (est_tp_pop + est_fn_pop)
            if (not np.isnan(est_tp_pop) and not np.isnan(est_fn_pop) and (est_tp_pop + est_fn_pop) > 0)
            else np.nan
        )
        adj_f1 = (
            2 * adj_precision * adj_recall / (adj_precision + adj_recall)
            if (not np.isnan(adj_precision) and not np.isnan(adj_recall) and (adj_precision + adj_recall) > 0)
            else np.nan
        )
        est_total_positives = (est_tp_pop + est_fn_pop) if (not np.isnan(est_tp_pop) and not np.isnan(est_fn_pop)) else np.nan

        # Sample metrics (unweighted, within the labeled sample)
        precision_sample = tp_sample / n_sample_llm_pos if n_sample_llm_pos > 0 else np.nan
        sample_recall = tp_sample / (tp_sample + fn_sample) if (tp_sample + fn_sample) > 0 else np.nan
        
        rows.append({
            "metric": metric,
            "n_full_dataset": n_full_total,
            "n_full_llm_pos": int(n_full_llm_pos),
            "n_full_llm_neg": int(n_full_llm_neg),
            "n_sample_llm_pos": int(n_sample_llm_pos),
            "n_sample_llm_neg": int(n_sample_llm_neg),
            "weight_llm_pos": float(weight_llm_pos) if not np.isnan(weight_llm_pos) else np.nan,
            "weight_llm_neg": float(weight_llm_neg) if not np.isnan(weight_llm_neg) else np.nan,
            "sample_precision": float(precision_sample) if not np.isnan(precision_sample) else np.nan,
            "est_total_positives": est_total_positives,
            "est_tp_population": est_tp_pop,
            "est_fp_population": est_fp_pop,
            "est_fn_population": est_fn_pop,
            "est_tn_population": est_tn_pop,
            "adj_precision": adj_precision,
            "adj_recall": adj_recall,
            "adj_f1": adj_f1,
            "sample_recall": float(sample_recall) if not np.isnan(sample_recall) else np.nan,
        })
    
    return pd.DataFrame(rows)


def compute_majority_vote_labels(df: pd.DataFrame, active_metrics: List[str] = None) -> pd.DataFrame:
    """Add majority-vote binary label columns. Ties are broken to 0."""
    if active_metrics is None:
        active_metrics = METRIC_NAMES
    
    df = df.copy()
    for metric in active_metrics:
        col_name = f"{metric}_per_annotator"
        if col_name not in df.columns:
            continue
        per_annotator = df[col_name]
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
    rotation: int = 0,
    active_metrics: List[str] = None
) -> None:
    """Plot intercoder reliability bar chart on given axes."""
    # Filter to active metrics if specified
    if active_metrics:
        icr_df = icr_df[icr_df["metric"].isin(active_metrics)]
    
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
    show_first_ylabel_only: bool = False,
    human_label_type: str = "mean",
    active_metrics: List[str] = None
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
        human_label_type: "mean" or "majority" for llm_vs_human mode
        active_metrics: List of metrics to plot (defaults to METRIC_NAMES)
    """
    metrics_to_plot = active_metrics if active_metrics else METRIC_NAMES
    
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
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        if mode == "intercoder":
            per_annotator = df[f"{metric}_per_annotator"].tolist()
            row_labels = np.array([row[0] for row in per_annotator], dtype=int)
            col_labels = np.array([row[1] for row in per_annotator], dtype=int)
        else:  # llm_vs_human
            human_col = f"human_{human_label_type}_{metric}"
            llm_col = f"LLM_{metric}"
            mask = ~(df[human_col].isna() | df[llm_col].isna())
            human_vals = df.loc[mask, human_col].values
            llm_vals = df.loc[mask, llm_col].values
            # For majority, values are already 0/1; for mean, threshold at 0.5
            if human_label_type == "majority":
                row_labels = human_vals.astype(int)
            else:
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
    title: str = "Human–LLM Agreement - v2 Metrics",
    fontsize: int = 12,
    rotation: int = 0,
    active_metrics: List[str] = None
) -> None:
    """Plot human-LLM correlations bar chart on given axes."""
    # Filter to active metrics if specified
    if active_metrics:
        corr_mean_df = corr_mean_df[corr_mean_df["metric"].isin(active_metrics)]
        corr_maj_df = corr_maj_df[corr_maj_df["metric"].isin(active_metrics)]
    
    metrics = corr_mean_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.18

    spearman_majority = corr_maj_df["spearman_majority"].values
    spearman_excl_ties = corr_mean_df["spearman_excl_ties_mean"].values
    binary_agreement_majority = corr_maj_df["binary_agreement_majority"].values
    binary_agr_excl_ties = corr_mean_df["binary_agr_excl_ties_mean"].values

    bars1 = ax.bar(x - 1.5*width, spearman_majority, width, label="Spearman (Majority)", color="#4C72B0")
    bars2 = ax.bar(x - 0.5*width, spearman_excl_ties, width, label="Spearman (Excl. Ties)", color="#7EB0D5")
    bars3 = ax.bar(x + 0.5*width, binary_agreement_majority, width, label="Accuracy (Majority)", color="#C44E52")
    bars4 = ax.bar(x + 1.5*width, binary_agr_excl_ties, width, label="Accuracy (Excl. Ties)", color="#55A868")

    ax.set_xlabel("Metric", fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=fontsize - 1, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(-0.3, 1.35)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=fontsize - 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            va = "bottom" if height >= 0 else "top"
            offset = 2 if height >= 0 else -2
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points", ha="center", va=va, fontsize=fontsize - 3)


def _plot_precision_recall_on_ax(
    ax: plt.Axes,
    corr_mean_df: pd.DataFrame,
    title: str = "Precision & Recall (Excl. Ties)",
    fontsize: int = 12,
    rotation: int = 0,
    active_metrics: List[str] = None,
    pop_adj_df: pd.DataFrame = None
) -> None:
    """Plot precision, recall, and F1 bar chart on given axes.
    
    If pop_adj_df is provided, shows both sample and estimated (population-adjusted) metrics.
    """
    # Filter to active metrics if specified
    if active_metrics:
        corr_mean_df = corr_mean_df[corr_mean_df["metric"].isin(active_metrics)]
        if pop_adj_df is not None:
            pop_adj_df = pop_adj_df[pop_adj_df["metric"].isin(active_metrics)]
    
    metrics = corr_mean_df["metric"].tolist()
    x = np.arange(len(metrics))
    
    # If we have population-adjusted data, show sample metrics and estimated recall/F1
    if pop_adj_df is not None and len(pop_adj_df) > 0:
        width = 0.12
        
        # Use sample_precision and sample_recall from pop_adj_df for consistency
        # (both are based on majority vote labels)
        # Note: Est. Precision = Sample Precision always, so we only show one
        pop_adj_merged = pop_adj_df.set_index("metric").reindex(metrics)
        precision_sample = pop_adj_merged["sample_precision"].values
        recall_sample = pop_adj_merged["sample_recall"].values
        recall_est = pop_adj_merged["adj_recall"].values
        
        # Calculate sample F1 from sample precision and recall
        f1_sample = 2 * precision_sample * recall_sample / (precision_sample + recall_sample)
        f1_sample = np.where(np.isnan(f1_sample), 0, f1_sample)
        f1_est = pop_adj_merged["adj_f1"].values
        
        bars1 = ax.bar(x - 2*width, precision_sample, width, label="Precision", color="#C44E52")
        bars2 = ax.bar(x - width, recall_sample, width, label="Sample Recall", color="#55A868")
        bars3 = ax.bar(x, recall_est, width, label="Est. Recall", color="#A8D5A8", edgecolor="#55A868", linewidth=1.5)
        bars4 = ax.bar(x + width, f1_sample, width, label="Sample F1", color="#8B5CF6")
        bars5 = ax.bar(x + 2*width, f1_est, width, label="Est. F1", color="#C4B5FD", edgecolor="#8B5CF6", linewidth=1.5)
        
        all_bars = [bars1, bars2, bars3, bars4, bars5]
    else:
        width = 0.2
        precision_excl = corr_mean_df["precision_excl_ties_mean"].values
        recall_excl = corr_mean_df["recall_excl_ties_mean"].values
        f1_excl = corr_mean_df["f1_excl_ties_mean"].values

        bars1 = ax.bar(x - width, precision_excl, width, label="Precision", color="#C44E52")
        bars2 = ax.bar(x, recall_excl, width, label="Recall", color="#55A868")
        bars3 = ax.bar(x + width, f1_excl, width, label="F1", color="#8B5CF6")
        
        all_bars = [bars1, bars2, bars3]

    ax.set_xlabel("Metric", fontsize=fontsize)
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=fontsize - 1, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(0, 1.25)
    n_legend_cols = 3 if len(all_bars) > 3 else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=n_legend_cols, fontsize=fontsize - 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize - 3)


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
    threshold: float = 0.5,
    active_metrics: List[str] = None,
    pop_adj_df: pd.DataFrame = None
) -> None:
    """
    Create a combined figure with all analysis plots.
    
    Layout:
    - Row 0: Intercoder reliability bar chart | Human-LLM correlations bar chart
    - Row 1: Inter-coder confusion matrices (active metrics only)
    - Row 2: LLM vs Human (majority) confusion matrices (active metrics only)
    - Rows 3+: Per-annotator LLM confusion matrices (one row per annotator)
    
    Args:
        active_metrics: List of metrics to show. Defaults to METRIC_NAMES.
        pop_adj_df: Population-adjusted metrics DataFrame (optional).
    """
    metrics_to_plot = active_metrics if active_metrics else METRIC_NAMES
    n_metrics = len(metrics_to_plot)
    n_annotators = len(annotators)
    n_rows = 3 + n_annotators  # bar charts + intercoder + majority LLM + per-annotator rows
    
    # Add majority vote columns if not already present
    df = compute_majority_vote_labels(df)
    
    # Get short names for annotators
    short_names = []
    for a in annotators:
        if "@" in a:
            short_names.append(a.split("@")[0])
        else:
            short_names.append(a[:15])
    
    # Adjust figure size based on number of metrics
    fig_width = max(12, 4 * n_metrics)
    fig = plt.figure(figsize=(fig_width, 5 + 2.5 * n_rows))
    
    # Use GridSpec for flexible layout with space for titles
    # Bar charts get 1.8, confusion matrix rows get 0.8 each
    height_ratios = [1.8, 0.8, 0.8] + [0.8] * n_annotators
    gs = fig.add_gridspec(n_rows, max(n_metrics, 3), height_ratios=height_ratios, hspace=0.6, wspace=0.35)
    
    # === Row 0: Bar charts (spanning columns) ===
    n_cols = max(n_metrics, 3)
    col_span = n_cols // 3
    ax_icr = fig.add_subplot(gs[0, :col_span])
    ax_corr = fig.add_subplot(gs[0, col_span:2*col_span])
    ax_prec = fig.add_subplot(gs[0, 2*col_span:])
    
    _plot_intercoder_reliability_on_ax(
        ax_icr, icr_df, title="A) Inter-coder Reliability", fontsize=10, rotation=15,
        active_metrics=metrics_to_plot
    )
    _plot_human_llm_correlations_on_ax(
        ax_corr, corr_mean_df, corr_maj_df, title="B) Human–LLM Agreement", fontsize=10, rotation=15,
        active_metrics=metrics_to_plot
    )
    title_c = "C) Precision, Recall & F1" if pop_adj_df is not None else "C) Precision & Recall"
    _plot_precision_recall_on_ax(
        ax_prec, corr_mean_df, title=title_c, fontsize=10, rotation=15,
        active_metrics=metrics_to_plot, pop_adj_df=pop_adj_df
    )
    
    # === Row 1: Inter-coder confusion matrices ===
    intercoder_axes = [fig.add_subplot(gs[1, idx]) for idx in range(n_metrics)]
    _plot_confusion_matrices_on_axes(
        intercoder_axes, df, mode="intercoder", annotators=annotators,
        fontsize=9, tick_labels=("0", "1"), pct_decimals=0, show_first_ylabel_only=True,
        active_metrics=metrics_to_plot
    )
    # Add title above the middle axis
    title_idx = n_metrics // 2
    intercoder_axes[title_idx].annotate(
        "D) Inter-coder Agreement Matrices", xy=(0.5, 1.25), xycoords="axes fraction",
        ha="center", fontsize=12, fontweight="bold"
    )
    
    # === Row 2: LLM vs Human (majority) confusion matrices ===
    llm_axes = [fig.add_subplot(gs[2, idx]) for idx in range(n_metrics)]
    _plot_confusion_matrices_on_axes(
        llm_axes, df, mode="llm_vs_human", threshold=threshold,
        fontsize=9, tick_labels=("0", "1"), pct_decimals=0, show_first_ylabel_only=True,
        human_label_type="majority", active_metrics=metrics_to_plot
    )
    llm_axes[title_idx].annotate(
        f"E) LLM vs Majority Human (threshold={threshold})", xy=(0.5, 1.25), xycoords="axes fraction",
        ha="center", fontsize=12, fontweight="bold"
    )
    
    # === Rows 3+: Per-annotator LLM confusion matrices ===
    first_per_annotator_axes = None
    for ann_idx, short_name in enumerate(short_names):
        row_idx = 3 + ann_idx
        axes = [fig.add_subplot(gs[row_idx, metric_idx]) for metric_idx in range(n_metrics)]
        
        if ann_idx == 0:
            first_per_annotator_axes = axes
        
        for metric_idx, metric in enumerate(metrics_to_plot):
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
        first_per_annotator_axes[title_idx].annotate(
            f"F) Per-Annotator LLM Agreement (threshold={threshold})", 
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
    metric_versions: Dict[str, str] = None
) -> None:
    """
    Add human annotation columns to the original CSV file.
    
    Args:
        csv_path: Path to the CSV file to update
        annotations_df: DataFrame with annotation data (from build_item_table)
        active_metrics: List of metrics that were annotated
        annotators: List of annotator emails
        metric_versions: Dict mapping metric name to version string (e.g., {"Humor": "v5", "Connection": "v4"})
    """
    if metric_versions is None:
        metric_versions = {}
    
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
    
    # Add annotation columns for each active metric
    for metric in active_metrics:
        metric_lower = metric.lower()
        version = metric_versions.get(metric, "v1")  # Default to v1 if no version found
        
        # Add per-annotator score columns
        for short_name in short_names:
            col_name = f"{short_name}_{metric_lower}_{version}"
            csv_df[col_name] = np.nan
            added_cols.append(col_name)
        
        # Add per-annotator reasoning columns for this metric
        for short_name in short_names:
            reason_col_name = f"{short_name}_{metric_lower}_{version}_reason"
            csv_df[reason_col_name] = ""
            added_cols.append(reason_col_name)
    
    # Fill in values from annotations_df
    for _, row in annotations_df.iterrows():
        qid = row["qid"]
        mask = csv_df[match_col] == qid
        
        if mask.any():
            # Add per-annotator scores and reasoning for each metric
            for metric in active_metrics:
                metric_lower = metric.lower()
                version = metric_versions.get(metric, "v1")
                per_annotator = row[f"{metric}_per_annotator"]
                per_annotator_reasoning = row.get(f"{metric}_reasoning_per_annotator", [""] * len(short_names))
                
                for i, short_name in enumerate(short_names):
                    # Score column
                    col_name = f"{short_name}_{metric_lower}_{version}"
                    csv_df.loc[mask, col_name] = per_annotator[i]
                    
                    # Reasoning column
                    reason_col_name = f"{short_name}_{metric_lower}_{version}_reason"
                    if i < len(per_annotator_reasoning):
                        csv_df.loc[mask, reason_col_name] = per_annotator_reasoning[i]
    
    # Save updated CSV
    csv_df.to_csv(csv_path, index=False)
    
    print(f"  Added columns: {added_cols}")
    print(f"  Updated: {csv_path}")


def main_csv_mode(
    csv_paths: List[str],
    output_dir: str,
    active_metrics: List[str] = None,
    full_dataset_path: str = None
) -> None:
    """
    Main analysis function for CSV mode (annotations embedded in CSV files).
    
    Args:
        csv_paths: List of paths to formatted CSV files with embedded annotations
        output_dir: Directory to save output files
        active_metrics: Which metrics to analyze. Defaults to all found in CSVs.
        full_dataset_path: Path to full dataset CSV for population-adjusted metrics.
    """
    print(f"\n=== Loading annotations from {len(csv_paths)} CSV file(s) ===")
    
    df, annotators, metric_versions, found_metrics = build_item_table_from_csv(
        csv_paths, active_metrics
    )
    
    # Use found metrics if no active_metrics specified
    if active_metrics is None:
        active_metrics = found_metrics
    else:
        # Filter to only metrics that were found
        active_metrics = [m for m in active_metrics if m in found_metrics]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nAnnotators:")
    for a in annotators:
        print("  ", a)
    
    print(f"\nMetric versions: {metric_versions}")
    print(f"Active metrics: {active_metrics}")
    print(f"\nNumber of items: {len(df)}")
    
    # Inter-coder reliability (human-only)
    print("\n=== Inter-coder Reliability (Humans Only) ===")
    icr_df = compute_intercoder_reliability(df, active_metrics=active_metrics)
    print(icr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Human–LLM correlations (mean)
    print("\n=== Human–LLM Correlations (Mean Human Label) ===")
    corr_mean_df = compute_human_llm_correlations(df, label_type="mean", active_metrics=active_metrics)
    print(corr_mean_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Majority vote and Human–LLM correlations (majority)
    df_majority = compute_majority_vote_labels(df, active_metrics=active_metrics)
    print("\n=== Human–LLM Correlations (Majority-vote Human Label) ===")
    corr_maj_df = compute_human_llm_correlations(df_majority, label_type="majority", active_metrics=active_metrics)
    print(corr_maj_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Population-adjusted metrics (if full dataset provided)
    pop_adj_df = None
    if full_dataset_path:
        print(f"\n=== Population-Adjusted Metrics (based on {full_dataset_path}) ===")
        pop_adj_df = compute_population_adjusted_metrics(
            df_majority, full_dataset_path, active_metrics
        )
        if len(pop_adj_df) > 0:
            display_cols = ["metric", "n_full_dataset", "n_full_llm_pos", "n_full_llm_neg",
                           "n_sample_llm_pos", "n_sample_llm_neg",
                           "sample_precision", "adj_recall", "adj_f1"]
            available_cols = [c for c in display_cols if c in pop_adj_df.columns]
            print(pop_adj_df[available_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Save tables to CSV
    icr_path = output_path / "intercoder_reliability.csv"
    corr_mean_path = output_path / "human_llm_corr_mean.csv"
    corr_maj_path = output_path / "human_llm_corr_majority.csv"
    
    icr_df.to_csv(icr_path, index=False)
    corr_mean_df.to_csv(corr_mean_path, index=False)
    corr_maj_df.to_csv(corr_maj_path, index=False)
    
    if pop_adj_df is not None and len(pop_adj_df) > 0:
        pop_adj_path = output_path / "population_adjusted_metrics.csv"
        pop_adj_df.to_csv(pop_adj_path, index=False)
        print(f"  {pop_adj_path.name}")
    
    print(f"\nResults saved to {output_path}:")
    print(f"  {icr_path.name}")
    print(f"  {corr_mean_path.name}")
    print(f"  {corr_maj_path.name}")
    
    # Generate combined plot with all analyses
    if len(annotators) >= 2:
        print("\nGenerating plot...")
        combined_plot_path = output_path / "combined_analysis.png"
        plot_combined_analysis(
            df, annotators, icr_df, corr_mean_df, corr_maj_df, combined_plot_path,
            active_metrics=active_metrics, pop_adj_df=pop_adj_df
        )
    else:
        print(f"\nSkipping combined plot (requires at least 2 annotators, found {len(annotators)})")


def main(v2_csv_path: str, json_path: str, active_metrics: List[str] = None, 
         full_dataset_path: str = None) -> None:
    """
    Main analysis function (JSON mode - annotations from Label Studio).
    
    Args:
        v2_csv_path: Path to balanced_dataset_v2.csv with v2 metric scores
        json_path: Path to labelstudio_output.json with human annotations
        active_metrics: For Yes/No format labeling, which metrics to analyze.
                       Defaults to all METRIC_NAMES.
        full_dataset_path: Path to full dataset CSV for population-adjusted metrics.
    """
    if active_metrics is None:
        active_metrics = METRIC_NAMES
        
    print(f"Loading v2 scores from: {v2_csv_path}")
    v2_df = pd.read_csv(v2_csv_path)
    
    # Build metric_versions dict from found score columns
    metric_versions: Dict[str, str] = {}
    for metric in active_metrics:
        score_col = find_score_column(v2_df, metric)
        if score_col:
            version = extract_version_from_column(score_col)
            if version:
                metric_versions[metric] = version
                print(f"  Found {metric}: {score_col} -> {version}")
            else:
                metric_versions[metric] = "v1"
                print(f"  Found {metric}: {score_col} -> v1 (no version in name)")
    
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

    # Population-adjusted metrics (if full dataset provided)
    pop_adj_df = None
    if full_dataset_path:
        print(f"\n=== Population-Adjusted Metrics (based on {full_dataset_path}) ===")
        pop_adj_df = compute_population_adjusted_metrics(
            df_majority, full_dataset_path, active_metrics
        )
        if len(pop_adj_df) > 0:
            # Print key columns
            display_cols = ["metric", "n_full_dataset", "n_full_llm_pos", "n_full_llm_neg",
                           "n_sample_llm_pos", "n_sample_llm_neg",
                           "weight_llm_pos", "weight_llm_neg",
                           "sample_precision", "est_total_positives",
                           "sample_recall", "adj_recall", "adj_f1"]
            print(pop_adj_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print("\n  Note: adj_* metrics use sampling-weight correction stratified by LLM prediction.")
            print("        sample_* metrics are within the labeled (often balanced) sample only.")

    # Save tables to CSV
    icr_path = output_dir / "intercoder_reliability.csv"
    corr_mean_path = output_dir / "human_llm_corr_mean.csv"
    corr_maj_path = output_dir / "human_llm_corr_majority.csv"

    icr_df.to_csv(icr_path, index=False)
    corr_mean_df.to_csv(corr_mean_path, index=False)
    corr_maj_df.to_csv(corr_maj_path, index=False)
    
    if pop_adj_df is not None and len(pop_adj_df) > 0:
        pop_adj_path = output_dir / "population_adjusted_metrics.csv"
        pop_adj_df.to_csv(pop_adj_path, index=False)
        print(f"  {pop_adj_path.name}")

    print(f"\nResults saved to {output_dir}:")
    print(f"  {icr_path.name}")
    print(f"  {corr_mean_path.name}")
    print(f"  {corr_maj_path.name}")
    
    # Add annotation columns to the original CSV
    print("\nAdding annotation columns to CSV...")
    add_annotations_to_csv(v2_csv_path, df, active_metrics, annotators, metric_versions)

    # Generate combined plot with all analyses
    if len(annotators) >= 2:
        print("\nGenerating plot...")
        combined_plot_path = output_dir / "combined_analysis.png"
        plot_combined_analysis(df, annotators, icr_df, corr_mean_df, corr_maj_df, combined_plot_path,
                               active_metrics=active_metrics, pop_adj_df=pop_adj_df)
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
        description="Inter-coder reliability analysis for v2 metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JSON mode (annotations from Label Studio):
  python intercoder_reliability_v2.py scores.csv annotations.json
  
  # CSV mode (annotations embedded in CSV files):
  python intercoder_reliability_v2.py --csv-mode file1.csv file2.csv --output-dir ./results
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--csv-mode",
        action="store_true",
        help="Use CSV mode: load annotations directly from formatted CSV files"
    )
    
    # Positional arguments (interpretation depends on mode)
    parser.add_argument(
        "input_files", 
        nargs="*",
        help="In JSON mode: v2_csv json_file. In CSV mode: one or more CSV files with embedded annotations"
    )
    
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for results (CSV mode). Defaults to parent of first input file."
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=None,
        help="Active metrics to analyze (default: all found in input)"
    )
    
    parser.add_argument(
        "--full-dataset",
        type=str,
        default=None,
        help="Path to full dataset CSV for population-adjusted metrics"
    )
    
    args = parser.parse_args()
    
    if args.csv_mode:
        # CSV mode: annotations embedded in CSV files
        if not args.input_files:
            parser.error("CSV mode requires at least one input CSV file")
        
        csv_paths = args.input_files
        
        # Default output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Default to parent of first input file
            output_dir = str(Path(csv_paths[0]).parent.parent)
        
        main_csv_mode(csv_paths, output_dir, args.metrics, args.full_dataset)
    else:
        # JSON mode (legacy): annotations from Label Studio
        if len(args.input_files) == 0:
            v2_csv = str(script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv")
            json_file = str(script_dir / "balanced_dataset" / "labelstudio_output.json")
        elif len(args.input_files) == 1:
            v2_csv = args.input_files[0]
            json_file = str(script_dir / "balanced_dataset" / "labelstudio_output.json")
        elif len(args.input_files) >= 2:
            v2_csv = args.input_files[0]
            json_file = args.input_files[1]
        
        main(v2_csv, json_file, args.metrics, args.full_dataset)

