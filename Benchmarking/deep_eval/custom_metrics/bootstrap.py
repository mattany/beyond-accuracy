import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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
target_directory = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/run_7"

if __name__ == "__main__":
    # Ensure numpy seed for reproducibility
    np.random.seed(42)
    
    import sys
    
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
        print(f"Analyzing directory provided by argument: {target_directory}")
    else:
        print(f"Analyzing default directory: {target_directory}")
        
    bootstrap_analysis(target_directory)