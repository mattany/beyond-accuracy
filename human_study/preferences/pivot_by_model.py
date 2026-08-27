#!/usr/bin/env python3
"""
Merge first_exp.csv and sec_exp.csv into a single CSV while keeping the original schema.
Filters out control questions and cleans the question column for sec_exp.csv.
"""

import pandas as pd
from pathlib import Path


def clean_question_text(text):
    """Remove appended answer text from questions in sec_exp.csv"""
    if pd.isna(text):
        return text
    text = str(text)
    # Split at '\nExplanation A' and keep only the question part
    if '\nExplanation A' in text:
        return text.split('\nExplanation A')[0].strip()
    return text


def clean_model_name(source):
    """Clean model name - handle duplicated names like 'model,model'"""
    if pd.isna(source) or source == '':
        return None
    source = str(source)
    if ',' in source:
        parts = [p.strip() for p in source.split(',')]
        unique_parts = list(dict.fromkeys(parts))  # Remove duplicates, preserve order
        return unique_parts[0] if unique_parts else None
    return source.strip()


def merge_experiments(data_dir: Path, output_path: Path = None) -> pd.DataFrame:
    """
    Merge first_exp.csv and sec_exp.csv while keeping the original schema.
    
    Args:
        data_dir: Path to the data directory containing the CSV files
        output_path: Optional path to save merged data
    
    Returns:
        Merged DataFrame with original schema
    """
    first_exp_path = data_dir / "first_exp.csv"
    sec_exp_path = data_dir / "sec_exp.csv"
    
    # Read both CSVs
    df_first = pd.read_csv(first_exp_path)
    df_second = pd.read_csv(sec_exp_path)
    
    print(f"Loaded first_exp.csv: {len(df_first)} rows")
    print(f"Loaded sec_exp.csv: {len(df_second)} rows")
    
    # Add experiment column to identify source
    df_first['experiment'] = 1
    df_second['experiment'] = 2
    
    # Clean question column for sec_exp (has answers appended by mistake)
    df_second['question'] = df_second['question'].apply(clean_question_text)
    print("Cleaned question column in sec_exp.csv")
    
    # Clean SOURCE columns - remove duplicated model names like "model,model"
    for df in [df_first, df_second]:
        df['explanation_a_SOURCE'] = df['explanation_a_SOURCE'].apply(clean_model_name)
        df['explanation_b_SOURCE'] = df['explanation_b_SOURCE'].apply(clean_model_name)
    print("Cleaned SOURCE columns (removed duplicate model names)")
    
    # Merge the dataframes
    df_merged = pd.concat([df_first, df_second], ignore_index=True)
    merged_total = len(df_merged)
    print(f"\nMerged total: {merged_total} rows")
    
    # Filter out rows where either SOURCE column is null
    incomplete_mask = (
        df_merged['explanation_a_SOURCE'].isna() | 
        df_merged['explanation_b_SOURCE'].isna()
    )
    incomplete_count = int(incomplete_mask.sum())
    df_merged = df_merged[~incomplete_mask]
    print(f"Removed {incomplete_count} rows with missing source")
    
    # Filter out rows where both models are the same
    same_model_mask = df_merged['explanation_a_SOURCE'] == df_merged['explanation_b_SOURCE']
    same_model_count = int(same_model_mask.sum())
    df_merged = df_merged[~same_model_mask]
    final_count = len(df_merged)
    print(f"Removed {same_model_count} rows with same model for A and B")
    print(f"Final row count: {final_count}")
    
    # Add cluster column - unique number for each model pair (regardless of order)
    def get_canonical_pair(row):
        """Create a canonical (sorted) tuple of the two models"""
        model_a = row['explanation_a_SOURCE'] if pd.notna(row['explanation_a_SOURCE']) else ''
        model_b = row['explanation_b_SOURCE'] if pd.notna(row['explanation_b_SOURCE']) else ''
        # Sort to make order-independent
        return tuple(sorted([model_a, model_b]))
    
    # Create canonical pairs for each row
    df_merged['_pair'] = df_merged.apply(get_canonical_pair, axis=1)
    
    # Get unique pairs and assign cluster numbers
    unique_pairs = df_merged['_pair'].unique()
    pair_to_cluster = {pair: i for i, pair in enumerate(unique_pairs)}
    
    # Map pairs to cluster numbers
    df_merged['cluster'] = df_merged['_pair'].map(pair_to_cluster)
    
    # Remove temporary column
    df_merged = df_merged.drop(columns=['_pair'])
    
    num_clusters = len(unique_pairs)
    print(f"Created {num_clusters} unique clusters")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = data_dir / "experiment_b_merged.csv"
    
    # Save merged data
    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Write stats.txt
    stats_path = data_dir / "stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Merged total: {merged_total} rows\n")
        f.write(f"Removed {incomplete_count} rows with missing source\n")
        f.write(f"Removed {same_model_count} rows with same model for A and B\n")
        f.write(f"Final row count: {final_count}\n")
        f.write(f"Created {num_clusters} unique clusters\n")
        f.write("\n")
        f.write("Cluster details:\n")
        f.write("-" * 60 + "\n")
        
        # Get cluster details
        cluster_to_pair = {v: k for k, v in pair_to_cluster.items()}
        for cluster_id in sorted(cluster_to_pair.keys()):
            pair = cluster_to_pair[cluster_id]
            row_count = len(df_merged[df_merged['cluster'] == cluster_id])
            f.write(f"Cluster {cluster_id}: {pair[0]} vs {pair[1]} ({row_count} rows)\n")
    
    print(f"Saved stats to: {stats_path}")
    
    return df_merged


def main():
    data_dir = Path(__file__).parent / "data"
    
    if not (data_dir / "first_exp.csv").exists():
        print(f"Error: first_exp.csv not found in {data_dir}")
        return
    
    if not (data_dir / "sec_exp.csv").exists():
        print(f"Error: sec_exp.csv not found in {data_dir}")
        return
    
    merge_experiments(data_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
