import os
import pandas as pd
import numpy as np
import sys

# Define lower-is-better metrics (must match aggregate.py)
LOWER_IS_BETTER_METRICS = ["ari", "dale_chall", "flesch_kincaid"]

# Define Baram-Tsabari Metrics
BARAM_TSABARI_METRICS = {
    "jargon", 
    "explanation_type", 
    "metaphor_explicit", 
    "content_units_explicit",
    "humor_explicit", 
    "analogy_explicit", 
    "connection_to_everyday_life"
}

def normalize_row(row):
    """
    Normalizes the Mean_Score based on the metric.
    Note: Standard Error (SE) scales linearly with the score. 
    If we flip the score (1-x) or scale it, we must scale SE too.
    """
    metric = row['Metric']
    score = row['Mean_Score']
    se = row['Bootstrap_SE']
    
    # 1. Handle "Lower is Better" (ARI, Dale-Chall, Flesch-Kincaid)
    # These are unbounded or specific scales. aggregate.py normalizes them to [0,1].
    # Since we don't have the global min/max here easily without reading all files again,
    # we might face a consistency issue if we don't use the exact same min/max as aggregate.py.
    #
    # However, bootstrap_analysis_results contains means. 
    # If we want to aggregate them, we MUST normalize them to a common scale.
    #
    # Strategy: We will perform Min-Max normalization per metric across all models IN THIS FILE.
    return score, se

def aggregate_bootstrap_results(directory):
    # Read from nested bootstrap folder
    bootstrap_dir = os.path.join(directory, "bootstrap")
    input_path = os.path.join(bootstrap_dir, "bootstrap_analysis_results.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Filter for Baram-Tsabari metrics only
    df = df[df['Metric'].isin(BARAM_TSABARI_METRICS)]
    
    if df.empty:
        print(f"Warning: No Baram-Tsabari metrics found in {input_path}")
        return

    # We need to normalize scores per Metric group before averaging
    normalized_rows = []
    
    grouped = df.groupby('Metric')
    
    for metric, group in grouped:
        # Get Min/Max for this metric across all models to normalize
        min_val = group['Mean_Score'].min()
        max_val = group['Mean_Score'].max()
        
        is_lower_better = metric in LOWER_IS_BETTER_METRICS
        
        for idx, row in group.iterrows():
            raw_score = row['Mean_Score']
            raw_se = row['Bootstrap_SE']
            
            # Normalize
            if min_val == max_val:
                norm_score = raw_score # Default to raw if no variance (e.g. all 1.0)
                # If all 1.0, it's effectively 1.0. If all 15.0 (ARI), it's problematic to sum.
                # Assuming 0-1 metrics stay 0-1. Unbounded metrics with no variance are tricky.
                # For safety in aggregation, if it's > 1 and constant, we might want to set to 0.5 or 1?
                # Let's assume if it's constant, it doesn't distinguish models.
                norm_se = raw_se # SE scale is unchanged if no scaling
                
            else:
                if is_lower_better:
                    # Formula: 1 - (x - min) / (max - min) = (max - x) / (max - min)
                    norm_score = (max_val - raw_score) / (max_val - min_val)
                else:
                    # Formula: (x - min) / (max - min)
                    # WAIT: aggregate.py line 28-30:
                    # if min >= 0 and max <= 1: normalized = numeric_df (already 0-1)
                    # else: normalized = (numeric_df - min) / (max - min)
                    
                    if min_val >= 0 and max_val <= 1:
                        norm_score = raw_score
                        # SE is on same scale
                        norm_se = raw_se
                    else:
                        norm_score = (raw_score - min_val) / (max_val - min_val)
                        # SE must be scaled by the same factor (1 / (max-min))
                        scale_factor = 1 / (max_val - min_val)
                        norm_se = raw_se * scale_factor
            
            # Handle Lower is Better Normalization for scaling SE
            if is_lower_better and min_val != max_val:
                 scale_factor = 1 / (max_val - min_val)
                 norm_se = raw_se * scale_factor

            normalized_rows.append({
                'Model': row['Model'],
                'Metric': metric,
                'Norm_Score': norm_score,
                'Norm_SE': norm_se
            })
            
    norm_df = pd.DataFrame(normalized_rows)
    
    # Aggregate by Model
    # 1. Average Score (Mean of Normalized Scores)
    # 2. Combined Error (Mean of SEs, or Quadrature)
    #    If we average metrics M1...Mn, the SE of the average is 1/n * sqrt(sum(SE_i^2)) assuming independence.
    
    agg_df = norm_df.groupby('Model').agg(
        Aggregated_Score=('Norm_Score', 'mean'),
        # Error propagation: sqrt(sum(sigma^2)) / N
        Aggregated_SE=('Norm_SE', lambda x: np.sqrt(np.sum(x**2)) / len(x))
    ).reset_index()
    
    # Sort
    agg_df = agg_df.sort_values('Aggregated_Score', ascending=False)
    
    # Calculate 95% CI for the aggregated score
    # CI = Score +/- 1.96 * SE
    agg_df['CI_Lower'] = agg_df['Aggregated_Score'] - (1.96 * agg_df['Aggregated_SE'])
    agg_df['CI_Upper'] = agg_df['Aggregated_Score'] + (1.96 * agg_df['Aggregated_SE'])
    
    # Save to nested bootstrap folder
    output_path = os.path.join(bootstrap_dir, "bootstrap_aggregated_model_scores.csv")
    agg_df.to_csv(output_path, index=False)
    
    print(f"Aggregated bootstrap scores saved to: {output_path}")
    print("\nTop Models by Aggregated Score:")
    print(agg_df.head().to_string(index=False))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/run_7"
        print(f"No directory specified, defaulting to: {target_dir}")
    
    aggregate_bootstrap_results(target_dir)

