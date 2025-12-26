#!/usr/bin/env python3
"""
Add individual coder labels from Label Studio JSON to the dataset CSVs.

Supports both v1 (question_id + model) and v2 (Index only) formats.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# V1 metrics (original format)
METRIC_NAMES_V1 = ["Analogy", "Metaphor", "Humor", "Connection"]

# V2 metrics (includes Scaffolding)
METRIC_NAMES_V2 = ["Analogy", "Metaphor", "Humor", "Connection", "Scaffolding"]

# Mapping from Label Studio choices to column names
CHOICE_TO_METRIC = {
    "Analogy": "analogy",
    "Metaphor": "metaphor", 
    "Humor": "humor",
    "Connection to everyday life": "connection",
    "Scaffolding": "scaffolding",
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


def detect_format_version(tasks: List[Dict[str, Any]]) -> str:
    """Detect whether the data is v1 (question_id + model) or v2 (Index) format."""
    if not tasks:
        return "v1"
    
    first_data = tasks[0].get("data", {})
    if "Index" in first_data and "question_id" not in first_data:
        return "v2"
    return "v1"


def extract_metrics_from_result(result: List[Dict[str, Any]], version: str = "v1") -> Dict[str, int]:
    """Extract binary metrics from annotation result."""
    metric_names = METRIC_NAMES_V2 if version == "v2" else METRIC_NAMES_V1
    metric_columns = [CHOICE_TO_METRIC.get(m, m.lower()) for m in metric_names]
    metrics = {col: 0 for col in metric_columns}
    
    for r in result:
        if r.get("type") == "choices":
            for c in r.get("value", {}).get("choices", []):
                if c in CHOICE_TO_METRIC:
                    col_name = CHOICE_TO_METRIC[c]
                    if col_name in metrics:
                        metrics[col_name] = 1
    return metrics


def build_coder_labels(tasks: List[Dict[str, Any]], version: str = None) -> pd.DataFrame:
    """
    Build a DataFrame with coder labels for each sample.
    
    For v1 format (question_id + model):
        Returns DataFrame with columns:
            - question_id, model
            - {metric}_{coder_name} for each metric and coder
    
    For v2 format (Index only):
        Returns DataFrame with columns:
            - Index
            - {metric}_{coder_name} for each metric and coder (includes scaffolding)
    """
    if version is None:
        version = detect_format_version(tasks)
    
    print(f"Detected format version: {version}")
    
    metric_names = METRIC_NAMES_V2 if version == "v2" else METRIC_NAMES_V1
    metric_columns = [CHOICE_TO_METRIC.get(m, m.lower()) for m in metric_names]
    
    annotators = extract_annotators(tasks)
    
    # Create short names for annotators (use first part of email)
    annotator_names = {}
    for email in annotators:
        name = email.split("@")[0].replace(".", "_")
        annotator_names[email] = name
    
    print(f"Annotators found: {annotator_names}")
    print(f"Metrics: {metric_columns}")
    
    rows = []
    for t in tasks:
        # Build row identifier based on version
        if version == "v2":
            row = {"Index": t["data"]["Index"]}
        else:
            row = {
                "question_id": t["data"]["question_id"],
                "model": t["data"]["model"]
            }
        
        # Get latest annotation per annotator
        latest: Dict[str, Dict[str, Any]] = {}
        for ann in t.get("annotations", []):
            email = ann["completed_by"]["email"]
            ts = ann["updated_at"]
            if email not in latest or ts > latest[email]["ts"]:
                latest[email] = {
                    "ts": ts,
                    "metrics": extract_metrics_from_result(ann.get("result", []), version),
                }
        
        # Add each annotator's labels for each metric
        for email in annotators:
            name = annotator_names[email]
            if email in latest:
                metrics = latest[email]["metrics"]
                for metric_col in metric_columns:
                    row[f"{metric_col}_{name}"] = metrics.get(metric_col, None)
            else:
                # Annotator didn't label this item
                for metric_col in metric_columns:
                    row[f"{metric_col}_{name}"] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def add_coder_labels_to_csv(csv_path: str, json_path: str, output_path: str = None, version: str = None):
    """
    Add coder labels from JSON to the CSV dataset.
    
    Args:
        csv_path: Path to the CSV dataset
        json_path: Path to the Label Studio JSON export
        output_path: Optional output path (defaults to csv_path)
        version: Optional version ('v1' or 'v2'). Auto-detected if not specified.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loading JSON: {json_path}")
    tasks = load_tasks(json_path)
    
    # Auto-detect version if not specified
    if version is None:
        version = detect_format_version(tasks)
    
    # Build coder labels DataFrame
    coder_df = build_coder_labels(tasks, version)
    
    # Merge with original CSV based on version
    if version == "v2":
        merge_keys = ["Index"]
    else:
        merge_keys = ["question_id", "model"]
    
    print(f"Merging on: {merge_keys}")
    df_merged = df.merge(coder_df, on=merge_keys, how="left")
    
    # Save
    if output_path is None:
        output_path = csv_path
    
    df_merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Added columns: {[c for c in coder_df.columns if c not in merge_keys]}")
    
    return df_merged


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add coder labels from Label Studio JSON to CSV dataset"
    )
    parser.add_argument(
        "--csv", "-c",
        type=str,
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--json", "-j", 
        type=str,
        help="Path to Label Studio JSON export"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (defaults to overwriting input CSV)"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        choices=["v1", "v2"],
        default=None,
        help="Format version (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # If specific files are provided, use them
    if args.csv and args.json:
        add_coder_labels_to_csv(args.csv, args.json, args.output, args.version)
        return
    
    # Otherwise, run on default locations
    script_dir = Path(__file__).parent
    
    # V1 format: balanced_dataset
    json_v1 = script_dir / "balanced_dataset" / "labelstudio_output.json"
    csv_v1 = script_dir / "balanced_dataset" / "balanced_dataset.csv"
    if json_v1.exists() and csv_v1.exists():
        print("\n=== Processing V1 balanced_dataset ===")
        add_coder_labels_to_csv(str(csv_v1), str(json_v1), version="v1")
    
    # V2 format: balanced_dataset_v2 with v1 labelstudio export
    csv_v2 = script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv"
    if csv_v2.exists() and json_v1.exists():
        print("\n=== Processing V2 balanced_dataset_v2 (with v1 labels) ===")
        add_coder_labels_to_csv(str(csv_v2), str(json_v1), version="v1")
    
    # V2 format: balanced_dataset_v2_human
    json_v2_human = script_dir / "balanced_dataset_v2_human" / "labelstudio_output.json"
    csv_v2_human = script_dir / "balanced_dataset_v2_human" / "balanced_30_formatted.csv"
    if json_v2_human.exists() and csv_v2_human.exists():
        print("\n=== Processing V2 balanced_dataset_v2_human ===")
        add_coder_labels_to_csv(str(csv_v2_human), str(json_v2_human), version="v2")


if __name__ == "__main__":
    main()

