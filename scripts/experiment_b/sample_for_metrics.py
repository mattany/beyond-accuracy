#!/usr/bin/env python3
"""
Sample data from experiment_b for metric evaluation.
The sampled CSV grows on subsequent runs with higher sample counts.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLES_PER_CLUSTER = 100  # Change this to increase sample size
RANDOM_SEED = 42  # For reproducibility

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = Path(__file__).parent / "data"
MERGED_CSV = DATA_DIR / "experiment_b_merged.csv"
SAMPLED_CSV = DATA_DIR / "experiment_b_sampled.csv"


def sample_from_clusters(
    df: pd.DataFrame,
    samples_per_cluster: int,
    existing_indices: set = None,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Sample from each cluster, respecting existing samples.
    
    IMPORTANT: Existing samples are added FIRST (across all clusters),
    then new samples are added. This ensures stable row indices when
    the dataset grows, so checkpoints remain valid.
    
    Args:
        df: Full merged dataframe
        samples_per_cluster: Target samples per cluster
        existing_indices: Set of already-sampled indices to preserve
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sampled rows
    """
    existing_indices = existing_indices or set()
    np.random.seed(seed)
    
    existing_rows = []
    new_rows = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Get existing samples from this cluster
        existing_in_cluster = cluster_df[cluster_df.index.isin(existing_indices)]
        existing_count = len(existing_in_cluster)
        
        # Collect existing samples (will be added first)
        if existing_count > 0:
            existing_rows.append(existing_in_cluster)
        
        # Calculate how many more we need
        needed = samples_per_cluster - existing_count
        
        if needed > 0:
            # Get available rows (not already sampled)
            available = cluster_df[~cluster_df.index.isin(existing_indices)]
            
            # Sample what we need (or all available if not enough)
            n_to_sample = min(needed, len(available))
            if n_to_sample > 0:
                new_samples = available.sample(n=n_to_sample, random_state=seed + cluster_id)
                new_rows.append(new_samples)
                
        print(f"Cluster {cluster_id}: {existing_count} existing + {max(0, min(needed, len(cluster_df) - existing_count))} new = {min(samples_per_cluster, cluster_size)} total")
    
    # Combine: ALL existing first, then ALL new (preserves index stability)
    all_rows = existing_rows + new_rows
    if all_rows:
        result = pd.concat(all_rows)
        # Reset index to get sequential 0, 1, 2, ... but keep order
        return result.reset_index(drop=True)
    return pd.DataFrame()


def main():
    print(f"Sampling {SAMPLES_PER_CLUSTER} examples per cluster")
    print("=" * 60)
    
    # Load merged data
    if not MERGED_CSV.exists():
        print(f"Error: {MERGED_CSV} not found. Run pivot_by_model.py first.")
        return
    
    df = pd.read_csv(MERGED_CSV)
    df = df.reset_index(drop=True)  # Ensure clean index
    print(f"Loaded {len(df)} rows from merged dataset")
    print(f"Clusters: {sorted(df['cluster'].unique())}")
    print()
    
    # Check for existing sampled data
    existing_indices = set()
    if SAMPLED_CSV.exists():
        existing_df = pd.read_csv(SAMPLED_CSV)
        # Match by qid + explanation sources to find existing samples
        existing_keys = set(zip(
            existing_df['qid'].astype(str),
            existing_df['explanation_a_SOURCE'].astype(str),
            existing_df['explanation_b_SOURCE'].astype(str)
        ))
        
        # Find matching indices in the full dataset
        for idx, row in df.iterrows():
            key = (str(row['qid']), str(row['explanation_a_SOURCE']), str(row['explanation_b_SOURCE']))
            if key in existing_keys:
                existing_indices.add(idx)
        
        print(f"Found {len(existing_indices)} existing samples to preserve")
        print()
    
    # Sample from each cluster
    sampled_df = sample_from_clusters(df, SAMPLES_PER_CLUSTER, existing_indices)
    
    print()
    print("=" * 60)
    print(f"Total sampled: {len(sampled_df)} rows")
    print(f"Cluster distribution:")
    print(sampled_df['cluster'].value_counts().sort_index())
    
    # Save sampled data
    sampled_df.to_csv(SAMPLED_CSV, index=False)
    print(f"\nSaved to: {SAMPLED_CSV}")
    
    # Print summary for metric running
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run: python run_metrics_exp_b.py")
    print("2. Run: python metric_correlation.py")


if __name__ == "__main__":
    main()

