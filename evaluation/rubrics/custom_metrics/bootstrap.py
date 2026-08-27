import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import v2 definitions from aggregate_v2.py to keep in sync
from aggregate_v2 import (
    METRIC_WEIGHTS, 
    METRIC_FALLBACKS, 
    LOWER_IS_BETTER,
    NORMALIZATION_RANGES,
)


def bootstrap_analysis_v2(
    directory: str,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    output_file: str = "bootstrap_v2_results.csv"
):
    """
    Bootstrap analysis for the v2 weighted scoring system.
    Computes CIs for the weighted total score per model.
    Uses definitions from aggregate_v2.py for consistency.
    """
    print(f"\n{'='*60}")
    print("Bootstrap Analysis for V2 Aggregation")
    print(f"{'='*60}")
    
    # Use metric weights from aggregate_v2.py
    metric_weights = {name: info["weight"] for name, info in METRIC_WEIGHTS.items()}
    print(f"V2 Metric weights: {metric_weights}")
    
    # Load all metric data
    metric_data = {}  # metric_name -> {model_name -> scores_array}
    
    for metric_name in metric_weights.keys():
        # Try primary name first, then fallbacks
        names_to_try = [metric_name] + METRIC_FALLBACKS.get(metric_name, [])
        
        for name in names_to_try:
            filepath = os.path.join(directory, f"{name}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                score_cols = [c for c in df.columns if c.endswith("__score")]
                
                if score_cols:
                    metric_data[metric_name] = {}
                    for col in score_cols:
                        model_name = col.replace("__score", "")
                        scores = df[col].dropna().values
                        if len(scores) > 0:
                            metric_data[metric_name][model_name] = scores
                    print(f"  Loaded {metric_name} (from {name}): {len(metric_data[metric_name])} models")
                    break
        else:
            print(f"  Warning: {metric_name} not found")
    
    if not metric_data:
        print("No metric data found!")
        return
    
    # Get all models that appear in any metric
    all_models = set()
    for metric_scores in metric_data.values():
        all_models.update(metric_scores.keys())
    
    print(f"\nFound {len(all_models)} models: {sorted(all_models)}")
    
    # For each model, compute bootstrap CIs for weighted total score
    results = []
    
    for model in tqdm(sorted(all_models), desc="Bootstrapping models"):
        # Collect all metric scores for this model
        model_metric_scores = {}
        for metric_name, metric_scores in metric_data.items():
            if model in metric_scores:
                model_metric_scores[metric_name] = metric_scores[model]
        
        if not model_metric_scores:
            continue
        
        # Get the minimum number of samples across metrics (they should align by question)
        n_samples = min(len(scores) for scores in model_metric_scores.values())
        
        if n_samples < 2:
            continue
        
        # Truncate all to same length
        for metric_name in model_metric_scores:
            model_metric_scores[metric_name] = model_metric_scores[metric_name][:n_samples]
        
        # Pre-compute normalization ranges from FULL data (before bootstrapping)
        # This prevents artificial variance from changing min/max in each bootstrap sample
        normalization_ranges = {}
        for metric_name, scores in model_metric_scores.items():
            if metric_name in NORMALIZATION_RANGES:
                normalization_ranges[metric_name] = NORMALIZATION_RANGES[metric_name]
            else:
                normalization_ranges[metric_name] = (scores.min(), scores.max())
        
        # Compute original weighted total score
        def compute_weighted_total(scores_dict, indices=None):
            """Compute weighted total from metric scores, optionally using bootstrap indices."""
            total_weighted = 0
            total_weight = 0
            
            for metric_name, scores in scores_dict.items():
                weight = metric_weights.get(metric_name, 0)
                if weight == 0:
                    continue
                
                if indices is not None:
                    sample_scores = scores[indices]
                else:
                    sample_scores = scores
                
                # Use pre-computed normalization range (fixed across all bootstrap iterations)
                min_val, max_val = normalization_ranges[metric_name]
                
                if min_val == max_val:
                    normalized = np.full_like(sample_scores, 0.5, dtype=float)
                else:
                    normalized = (sample_scores - min_val) / (max_val - min_val)
                    # Clip to [0, 1] range
                    normalized = np.clip(normalized, 0, 1)
                
                # Invert if lower is better
                if metric_name in LOWER_IS_BETTER:
                    normalized = 1 - normalized
                
                # Weighted contribution
                mean_score = normalized.mean()
                total_weighted += mean_score * weight
                total_weight += weight
            
            return total_weighted / total_weight if total_weight > 0 else 0
        
        original_total = compute_weighted_total(model_metric_scores)
        
        # Bootstrap
        bootstrap_totals = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, n_samples)
            bootstrap_total = compute_weighted_total(model_metric_scores, indices)
            bootstrap_totals.append(bootstrap_total)
        
        bootstrap_totals = np.array(bootstrap_totals)
        bootstrap_se = np.std(bootstrap_totals, ddof=1)
        
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        ci_lower, ci_upper = np.percentile(bootstrap_totals, [lower_percentile, upper_percentile])
        
        results.append({
            "Model": model,
            "Total_Score": original_total,
            "Bootstrap_SE": bootstrap_se,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "N_Samples": n_samples,
            "N_Metrics": len(model_metric_scores),
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Total_Score", ascending=False)
    
    output_dir = os.path.join(directory, "bootstrap")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"V2 Bootstrap results saved to: {output_path}")
    print(f"{'='*60}")
    print(results_df.to_string(index=False, float_format="%.4f"))
    
    return results_df


def bootstrap_analysis(
    directory: str, 
    n_bootstrap: int = 10000, 
    confidence_level: float = 0.95,
    output_file: str = "bootstrap_analysis_results.csv"
):
    """
    Analyzes all CSV files in the directory to calculate Mean, Std Dev, 
    and Bootstrapped Standard Error/CI for each model.
    """
    results = []
    
    # List all CSV files (excluding aggregations or non-metric files)
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    print(f"Found {len(files)} metric files. Starting bootstrap analysis...")
    
    for file_name in tqdm(files):
        # Skip aggregation files if they exist in the root
        if "aggregate" in file_name or "summary" in file_name:
            continue
            
        metric_name = file_name.replace(".csv", "")
        file_path = os.path.join(directory, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
            # Identify model score columns
            # Format 1: ModelName__score (Run 7+)
            score_cols_v1 = [c for c in df.columns if c.endswith("__score")]
            
            # Format 2: metric_name_score__ModelName (Older runs)
            score_cols_v2 = [c for c in df.columns if "_score__" in c]
            
            if score_cols_v1:
                score_cols = score_cols_v1
                version = 1
            elif score_cols_v2:
                score_cols = score_cols_v2
                version = 2
            else:
                continue

            for col in score_cols:
                if version == 1:
                    model_name = col.replace("__score", "")
                else:
                    # e.g. explanation_type_score__llama_2_sft -> llama_2_sft
                    model_name = col.split("_score__")[1]
                
                # Get the scores, dropping NaNs
                scores = df[col].dropna().values
                
                if len(scores) < 2:
                    continue
                
                # 1. Original Statistics
                original_mean = np.mean(scores)
                original_std = np.std(scores, ddof=1) # Sample std dev
                
                # 2. Bootstrapping
                # Generate indices for resampling
                # shape: (n_bootstrap, n_samples)
                resampled_indices = np.random.randint(0, len(scores), (n_bootstrap, len(scores)))
                resampled_scores = scores[resampled_indices]
                
                # Calculate means for each bootstrap sample
                bootstrap_means = np.mean(resampled_scores, axis=1)
                
                # 3. Bootstrap Statistics
                bootstrap_se = np.std(bootstrap_means, ddof=1) # Standard Error
                
                # Confidence Intervals
                lower_percentile = (1 - confidence_level) / 2 * 100
                upper_percentile = (1 + confidence_level) / 2 * 100
                ci_lower, ci_upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
                
                results.append({
                    "Metric": metric_name,
                    "Model": model_name,
                    "N_Samples": len(scores),
                    "Mean_Score": original_mean,
                    "Std_Dev_Data": original_std,
                    "Bootstrap_SE": bootstrap_se, # This is the "Error" of the judge's estimate
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper
                })
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Create DataFrame and Save
    results_df = pd.DataFrame(results)
    
    # Sort for better readability
    results_df = results_df.sort_values(["Metric", "Mean_Score"], ascending=[True, False])
    
    output_path = os.path.join(directory, output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis complete. Results saved to:\n{output_path}")
    
    # Display a snippet
    print("\nSnippet of Results:")
    print(results_df[["Metric", "Model", "Mean_Score", "Bootstrap_SE"]].head(10).to_string(index=False))

# --- Run configuration ---
from evaluation.rubrics.settings import result_directory

target_directory = result_directory(9)

if __name__ == "__main__":
    # Ensure numpy seed for reproducibility
    np.random.seed(42)
    
    import sys
    
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
        print(f"Analyzing directory provided by argument: {target_directory}")
    else:
        print(f"Analyzing default directory: {target_directory}")
    
    # Run original bootstrap analysis
    bootstrap_analysis(target_directory)
    
    # Run v2 bootstrap analysis
    bootstrap_analysis_v2(target_directory)