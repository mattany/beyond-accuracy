#!/usr/bin/env python3
"""
Reorder columns in the dataset CSVs so related metrics are grouped together.
"""
from pathlib import Path
import pandas as pd


def reorder_columns(csv_path: str):
    """Reorder columns to group related metrics together."""
    print(f"Reordering columns in: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Define the column order
    # Start with identifiers
    base_cols = ["question_id", "question", "answer", "model"]
    
    # Define metrics and their related columns
    metrics = [
        {
            "name": "analogy",
            "binary": "analogy_explicit",
            "v1_score": "analogy_explicit_score",
            "v2_score": "analogy_v2_score",
            "coders": ["analogy_mattan_yeroushalmi", "analogy_maximbr", "analogy_nirgrn"],
        },
        {
            "name": "metaphor",
            "binary": "metaphor_explicit",
            "v1_score": "metaphor_explicit_score",
            "v2_score": "metaphor_v2_score",
            "coders": ["metaphor_mattan_yeroushalmi", "metaphor_maximbr", "metaphor_nirgrn"],
        },
        {
            "name": "humor",
            "binary": "humor_explicit",
            "v1_score": "humor_explicit_score",
            "v2_score": "humor_v2_score",
            "coders": ["humor_mattan_yeroushalmi", "humor_maximbr", "humor_nirgrn"],
        },
        {
            "name": "connection",
            "binary": "connection_to_everyday_life",
            "v1_score": "connection_to_everyday_life_score",
            "v2_score": "connection_to_everyday_life_v2_score",
            "coders": ["connection_mattan_yeroushalmi", "connection_maximbr", "connection_nirgrn"],
        },
    ]
    
    # Build ordered column list
    ordered_cols = base_cols.copy()
    
    for metric in metrics:
        # Add columns in order: binary, v1_score, v2_score, coders
        for col_key in ["binary", "v1_score", "v2_score"]:
            col = metric.get(col_key)
            if col and col in df.columns:
                ordered_cols.append(col)
        
        # Add coder columns
        for coder_col in metric.get("coders", []):
            if coder_col in df.columns:
                ordered_cols.append(coder_col)
    
    # Add metric_sum at the end if present
    if "metric_sum" in df.columns:
        ordered_cols.append("metric_sum")
    
    # Add any remaining columns we might have missed
    remaining = [c for c in df.columns if c not in ordered_cols]
    if remaining:
        print(f"  Note: Appending unordered columns: {remaining}")
        ordered_cols.extend(remaining)
    
    # Reorder and save
    df = df[ordered_cols]
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved with column order:")
    for i, col in enumerate(ordered_cols):
        print(f"    {i+1}. {col}")


def main():
    script_dir = Path(__file__).parent
    
    # Reorder balanced_dataset
    csv_v1 = script_dir / "balanced_dataset" / "balanced_dataset.csv"
    if csv_v1.exists():
        reorder_columns(str(csv_v1))
    
    # Reorder balanced_dataset_v2
    csv_v2 = script_dir / "balanced_dataset_v2" / "balanced_dataset_v2.csv"
    if csv_v2.exists():
        reorder_columns(str(csv_v2))


if __name__ == "__main__":
    main()

