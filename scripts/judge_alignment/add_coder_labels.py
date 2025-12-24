#!/usr/bin/env python3
"""
Add individual coder labels from Label Studio JSON to the dataset CSVs.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


METRIC_NAMES = ["Analogy", "Metaphor", "Humor", "Connection"]

# Mapping from Label Studio choices to column names
CHOICE_TO_METRIC = {
    "Analogy": "analogy",
    "Metaphor": "metaphor", 
    "Humor": "humor",
    "Connection to everyday life": "connection",
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
    """Extract binary metrics from annotation result."""
    metrics = {m: 0 for m in CHOICE_TO_METRIC.values()}
    for r in result:
        if r.get("type") == "choices":
            for c in r.get("value", {}).get("choices", []):
                if c in CHOICE_TO_METRIC:
                    metrics[CHOICE_TO_METRIC[c]] = 1
    return metrics


def build_coder_labels(tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame with coder labels for each (question_id, model).
    
    Returns DataFrame with columns:
        - question_id, model
        - {metric}_{coder_name} for each metric and coder
    """
    annotators = extract_annotators(tasks)
    
    # Create short names for annotators (use first part of email)
    annotator_names = {}
    for email in annotators:
        name = email.split("@")[0].replace(".", "_")
        annotator_names[email] = name
    
    print(f"Annotators found: {annotator_names}")
    
    rows = []
    for t in tasks:
        qid = t["data"]["question_id"]
        model = t["data"]["model"]
        
        # Get latest annotation per annotator
        latest: Dict[str, Dict[str, Any]] = {}
        for ann in t.get("annotations", []):
            email = ann["completed_by"]["email"]
            ts = ann["updated_at"]
            if email not in latest or ts > latest[email]["ts"]:
                latest[email] = {
                    "ts": ts,
                    "metrics": extract_metrics_from_result(ann.get("result", [])),
                }
        
        row = {"question_id": qid, "model": model}
        
        # Add each annotator's labels for each metric
        for email in annotators:
            name = annotator_names[email]
            if email in latest:
                metrics = latest[email]["metrics"]
                for metric in CHOICE_TO_METRIC.values():
                    row[f"{metric}_{name}"] = metrics.get(metric, None)
            else:
                # Annotator didn't label this item
                for metric in CHOICE_TO_METRIC.values():
                    row[f"{metric}_{name}"] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def add_coder_labels_to_csv(csv_path: str, json_path: str, output_path: str = None):
    """
    Add coder labels from JSON to the CSV dataset.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loading JSON: {json_path}")
    tasks = load_tasks(json_path)
    
    # Build coder labels DataFrame
    coder_df = build_coder_labels(tasks)
    
    # Merge with original CSV
    df_merged = df.merge(coder_df, on=["question_id", "model"], how="left")
    
    # Save
    if output_path is None:
        output_path = csv_path
    
    df_merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Added columns: {[c for c in coder_df.columns if c not in ['question_id', 'model']]}")
    
    return df_merged


def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / "balanced_dataset" / "labelstudio_output.json"
    
    # Add to balanced_dataset
    csv_v1 = script_dir / "balanced_dataset" / "balanced_dataset.csv"
    add_coder_labels_to_csv(str(csv_v1), str(json_path))
    
    # Add to balanced_dataset_v2
    csv_v2 = script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv"
    if csv_v2.exists():
        add_coder_labels_to_csv(str(csv_v2), str(json_path))


if __name__ == "__main__":
    main()

