#!/usr/bin/env python3
"""
Extract rows where annotators disagreed from the formatted CSV files.
"""
import pandas as pd
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    
    # Define input files and their annotator column pairs
    files_and_metrics = [
        (
            script_dir / "balanced_30_metaphor_v8_scaffolding_v2" / "balanced_30_metaphor_v8_scaffolding_formatted.csv",
            [
                ("mattany_metaphor_v8", "nirgrn_metaphor_v8"),
                ("mattany_scaffolding_v2", "nirgrn_scaffolding_v2"),
            ]
        ),
        (
            script_dir / "balanced_30_dataset_humor_v5_conn_v4" / "balanced_30_humor_v5_conn_v4_formatted.csv",
            [
                ("mattany_humor_v5", "nirgrn_humor_v5"),
                ("mattany_connection_v4", "nirgrn_connection_v4"),
            ]
        ),
    ]
    
    all_disagreements = []
    
    for csv_path, annotator_pairs in files_and_metrics:
        print(f"\nProcessing: {csv_path.name}")
        df = pd.read_csv(csv_path)
        
        for col1, col2 in annotator_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                print(f"  Warning: Columns {col1}, {col2} not found, skipping")
                continue
            
            # Find disagreements: one says 0, other says 1
            disagreement_mask = df[col1] != df[col2]
            disagreement_rows = df[disagreement_mask].copy()
            
            # Add metadata about which metric had the disagreement
            metric_name = col1.replace("mattany_", "").replace("nirgrn_", "")
            disagreement_rows["disagreement_metric"] = metric_name
            disagreement_rows["source_file"] = csv_path.name
            
            n_disagreements = len(disagreement_rows)
            print(f"  {metric_name}: {n_disagreements} disagreements")
            
            if n_disagreements > 0:
                all_disagreements.append(disagreement_rows)
    
    if all_disagreements:
        # Combine all disagreements
        combined_df = pd.concat(all_disagreements, ignore_index=True)
        
        # Remove duplicates (same row might disagree on multiple metrics)
        # Keep all columns but note which metrics had disagreement
        # Group by question to see all disagreement metrics for same row
        
        # Actually, let's keep duplicates to show which metric(s) each row disagrees on
        # But sort by question/Index for readability
        if "Index" in combined_df.columns:
            combined_df = combined_df.sort_values(["Index", "disagreement_metric"])
        elif "question" in combined_df.columns:
            combined_df = combined_df.sort_values(["question", "disagreement_metric"])
        
        # Move metadata columns to the front
        cols = combined_df.columns.tolist()
        meta_cols = ["disagreement_metric", "source_file"]
        other_cols = [c for c in cols if c not in meta_cols]
        combined_df = combined_df[meta_cols + other_cols]
        
        output_path = script_dir / "disagreements.csv"
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n=== Summary ===")
        print(f"Total disagreement rows: {len(combined_df)}")
        print(f"Saved to: {output_path}")
        
        # Show breakdown
        print(f"\nBreakdown by metric:")
        print(combined_df["disagreement_metric"].value_counts().to_string())
    else:
        print("\nNo disagreements found!")


if __name__ == "__main__":
    main()

