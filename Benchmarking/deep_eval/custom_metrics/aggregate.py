import os
import sys
# Add parent directory to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from config import PROJECT_DIR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

run_number = 8


def normalize_df(df, is_lower_better):
    """Normalize values in the DataFrame to [0, 1] range if not already normalized."""
    if df.select_dtypes(include=[np.number]).map(lambda x: 0 <= x <= 1).all().all():
        return df  # Already normalized

    numeric_df = df.select_dtypes(include=[np.number])
    min_val = numeric_df.min().min()
    max_val = numeric_df.max().max()

    if min_val == max_val:
        return df  # Avoid division by zero; leave unchanged
    if is_lower_better:
        if min_val >= 0 and max_val < 1:
            normalized = 1 - numeric_df
        normalized = 1 - ((numeric_df - min_val) / (max_val - min_val))
    else:
        if min_val >= 0 and max_val < 1:
            normalized = numeric_df
        else:
            normalized = (numeric_df - min_val) / (max_val - min_val)
    df.loc[:, normalized.columns] = normalized
    return df


def aggregate_over_model_question(
    directory: str, ignore_files: list[str], lower_is_better_metrics: list[str]
):
    metric_files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f not in ignore_files
    ]

    normalized_dfs = []
    for file_name in metric_files:
        full_path = os.path.join(directory, file_name)
        df = pd.read_csv(full_path)

        # Filter to score columns only (exclude "reason" columns)
        score_columns = [
            col for col in df.columns if not col.lower().endswith("reason")
        ]
        df_scores = df[score_columns].copy()
        is_lower_better = file_name[:-4] in lower_is_better_metrics
        df_normalized = normalize_df(df_scores, is_lower_better)
        normalized_dfs.append(df_normalized)

    # Check that all dataframes have the same shape
    shape_set = set(df.shape for df in normalized_dfs)
    if len(shape_set) > 1:
        raise ValueError(f"Mismatch in shapes across CSVs: {shape_set}")

    # Average the scores
    aggregated_df = pd.concat(normalized_dfs).groupby(level=0).mean()
    output_path = os.path.join(directory, "aggregations/aggregate_scores.csv")
    aggregated_df.index.name = "Index"
    suffix = "__score"
    aggregated_df.columns = [
        col[: -len(suffix)] if col.endswith(suffix) else col for col in aggregated_df.columns
    ]

    aggregated_df.to_csv(output_path, index=True)
    print(f"Aggregate scores written to {output_path}")


def aggregate_over_model_metric(directory, lower_is_better_metrics):
    def process_csvs():
        output_file = os.path.join(directory, "aggregations/normalized_means.csv")
        # Delete normalized_means.csv if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Deleted existing {output_file}")

        rows = []
        all_normalized_dfs = []

        for filename in os.listdir(directory):
            if not filename.endswith(".csv") or filename == "normalized_means.csv":
                continue

            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                df = normalize_df(
                    df, is_lower_better=filename[:-4] in lower_is_better_metrics
                )
                all_normalized_dfs.append(df)
                means = df.mean(numeric_only=True, skipna=True)
                means["metric"] = os.path.splitext(filename)[0]
                rows.append(means)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

        if rows:
            means_df = pd.DataFrame(rows)
            cols = ["metric"] + [col for col in means_df.columns if col != "metric"]
            means_df = means_df[cols]
            
            # Define the desired order of metrics
            metric_order = [
                "metaphor_explicit",
                "humor_explicit",
                "content_units_explicit",
                "connection_to_everyday_life",
                "explanation_type",
                "analogy_explicit",
                "jargon",
                "alternatives_explicit",
                "internal_coherence_explicit",
                "perceived_truth_explicit",
                "articulation_explicit",
                "completeness_explicit",
                "flesch_reading_ease",
                "flesch_kincaid",
                "dale_chall",
                "ari"
            ]
            
            # Set the metric column as categorical with the specified order
            means_df['metric'] = pd.Categorical(means_df['metric'], categories=metric_order, ordered=True)
            
            # Sort by the metric column
            means_df = means_df.sort_values('metric')
            
            means_df.to_csv(output_file, index=False)
            print(f"Wrote normalized means to {output_file}")

            # Also compute global means across all CSVs (column-wise), ignoring nulls
            combined_df = pd.concat(all_normalized_dfs, ignore_index=True)
            global_means = combined_df.mean(numeric_only=True, skipna=True)
            print("\nGlobal mean for each column across all CSVs (ignoring nulls):")
            print(global_means.sort_values())
        else:
            print("No valid CSV files found.")

    process_csvs()  # Replace "." with your target directory if needed


def aggregate_final_model_scores(directory: str, ignore_files: list[str], lower_is_better_metrics: list[str]):
    """
    Create a CSV with a single final score for each model.
    The score is an average across all (question, metric) pairs.
    """
    metric_files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f not in ignore_files
    ]
    
    all_scores = []
    
    for file_name in metric_files:
        full_path = os.path.join(directory, file_name)
        df = pd.read_csv(full_path)
        
        # Filter to score columns only (exclude "reason" columns)
        score_columns = [
            col for col in df.columns if not col.lower().endswith("reason")
        ]
        df_scores = df[score_columns].copy()
        
        # Normalize the scores
        is_lower_better = file_name[:-4] in lower_is_better_metrics
        df_normalized = normalize_df(df_scores, is_lower_better)
        
        # Add metric name as identifier and collect all scores
        metric_name = os.path.splitext(file_name)[0]
        for idx, row in df_normalized.iterrows():
            for col in df_normalized.columns:
                if pd.notna(row[col]):  # Skip NaN values
                    all_scores.append({
                        'model': col,
                        'question': idx,
                        'metric': metric_name,
                        'score': row[col]
                    })
    
    if not all_scores:
        print("No valid scores found.")
        return
    
    # Convert to DataFrame and compute final scores per model
    scores_df = pd.DataFrame(all_scores)
    final_scores = scores_df.groupby('model')['score'].mean().reset_index()
    final_scores.columns = ['model', 'final_score']
    final_scores = final_scores.sort_values('final_score', ascending=False)
    
    # Save to CSV
    output_path = os.path.join(directory, "aggregations/final_model_scores.csv")
    final_scores.to_csv(output_path, index=False)
    print(f"Final model scores written to {output_path}")
    
    # Print summary
    print("\nFinal Model Scores (sorted by performance):")
    print(final_scores.to_string(index=False, float_format='%.4f'))


def aggregate_categorical_model_scores(directory: str, ignore_files: list[str], lower_is_better_metrics: list[str]):
    """
    Create a CSV with categorical scores for each model.
    Categories: Baram Tsabari, Zemla, and Readability metrics.
    """
    # Define metric categories based on run.py
    baram_tsabari_metrics = {
        "jargon", "explanation_type", "metaphor_explicit", "content_units_explicit",
        "humor_explicit", "analogy_explicit", "connection_to_everyday_life"
    }
    
    zemla_metrics = {
        "internal_coherence_explicit", "completeness_explicit", "alternatives_explicit",
        "articulation_explicit", "perceived_truth_explicit"
    }
    
    readability_metrics = {
        "flesch_kincaid", "flesch_reading_ease", "dale_chall", "ari"
    }
    
    metric_files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f not in ignore_files
    ]
    
    all_scores = []
    
    for file_name in metric_files:
        full_path = os.path.join(directory, file_name)
        df = pd.read_csv(full_path)
        
        # Filter to score columns only (exclude "reason" columns)
        score_columns = [
            col for col in df.columns if not col.lower().endswith("reason")
        ]
        df_scores = df[score_columns].copy()
        
        # Normalize the scores
        is_lower_better = file_name[:-4] in lower_is_better_metrics
        df_normalized = normalize_df(df_scores, is_lower_better)
        
        # Determine metric category
        metric_name = os.path.splitext(file_name)[0]
        if metric_name in baram_tsabari_metrics:
            category = "baram_tsabari"
        elif metric_name in zemla_metrics:
            category = "zemla"
        elif metric_name in readability_metrics:
            category = "readability"
        else:
            continue  # Skip metrics that don't belong to any category
        
        # Collect all scores for this category
        for idx, row in df_normalized.iterrows():
            for col in df_normalized.columns:
                if pd.notna(row[col]):  # Skip NaN values
                    all_scores.append({
                        'model': col,
                        'question': idx,
                        'metric': metric_name,
                        'category': category,
                        'score': row[col]
                    })
    
    if not all_scores:
        print("No valid scores found.")
        return
    
    # Convert to DataFrame and compute categorical scores per model
    scores_df = pd.DataFrame(all_scores)
    categorical_scores = scores_df.groupby(['model', 'category'])['score'].mean().reset_index()
    
    # Pivot to get one row per model with separate columns for each category
    result_df = categorical_scores.pivot(index='model', columns='category', values='score').reset_index()
    
    # Ensure all categories are present (fill with NaN if missing)
    for category in ['baram_tsabari', 'zemla', 'readability']:
        if category not in result_df.columns:
            result_df[category] = np.nan
    
    # Reorder columns
    result_df = result_df[['model', 'baram_tsabari', 'zemla', 'readability']]
    
    # Rename columns for clarity
    result_df.columns = ['model', 'baram_tsabari_score', 'zemla_score', 'readability_score']
    
    # Sort by overall performance (average of all categories)
    result_df['overall_average'] = result_df[['baram_tsabari_score', 'zemla_score', 'readability_score']].mean(axis=1, skipna=True)
    result_df = result_df.sort_values('overall_average', ascending=False)
    result_df = result_df.drop('overall_average', axis=1)  # Remove the temporary column
    
    # Save to CSV
    output_path = os.path.join(directory, "aggregations/categorical_model_scores.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Categorical model scores written to {output_path}")
    
    # Print summary
    print("\nCategorical Model Scores (sorted by overall performance):")
    print(result_df.to_string(index=False, float_format='%.4f'))


def calculate_category_errors(directory: str, lower_is_better_metrics: list[str]):
    """
    Get pre-computed confidence intervals for the Baram-Tsabari category from the 
    bootstrap aggregated results file.
    Returns DataFrame: [model, ci_lower, ci_upper]
    """
    bootstrap_path = os.path.join(directory, "bootstrap", "bootstrap_aggregated_model_scores.csv")
    if not os.path.exists(bootstrap_path):
        print(f"Warning: Bootstrap aggregated results not found at {bootstrap_path}. Error bars will be missing.")
        return None
        
    df = pd.read_csv(bootstrap_path)
    
    # The file contains: Model, Aggregated_Score, Aggregated_SE, CI_Lower, CI_Upper
    # We return the data needed for plotting error bars
    return df[['Model', 'Aggregated_Score', 'CI_Lower', 'CI_Upper']]


def _plot_single_category(df, score_column, title, output_path, color='steelblue', error_df=None):
    """
    Helper function to create a single bar plot for a category.
    """
    # Filter out NaN values and sort by score
    plot_data = df.dropna(subset=[score_column]).sort_values(score_column, ascending=False)
    
    if plot_data.empty:
        print(f"No data available for {title}")
        return
    
    # Prepare error bars if available (only for Baram Tsabari)
    yerr = None
    if error_df is not None and "baram_tsabari" in score_column:
        # error_df contains: Model, Aggregated_Score, CI_Lower, CI_Upper
        # yerr should be the distance from bar top to CI bounds
        errors = []
        for model in plot_data['model']:
            # Match model name (remove __score suffix if present)
            model_clean = model.replace('__score', '')
            match = error_df[error_df['Model'] == model_clean]
            if not match.empty:
                agg_score = match.iloc[0]['Aggregated_Score']
                ci_upper = match.iloc[0]['CI_Upper']
                # Error bar = CI_Upper - Score (half-width of CI)
                errors.append(ci_upper - agg_score)
            else:
                errors.append(0)
        yerr = errors
    
    plt.figure(figsize=(12, 8))
    # Increased capsize to 10 and added error_kw for visibility
    bars = plt.bar(range(len(plot_data)), plot_data[score_column], yerr=yerr, capsize=10, 
                  error_kw={'elinewidth': 2, 'capthick': 2}, color=color, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'{title} - Model Performance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(plot_data)), plot_data['model'], rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Adjust label position if error bar is present
        y_pos = height + (yerr[i] if yerr else 0) + 0.01
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_categorical_scores(directory: str):
    """
    Plot 3 separate graphs for each category (Baram Tsabari, Zemla, Readability).
    """
    # Read the categorical scores
    input_path = os.path.join(directory, "aggregations/categorical_model_scores.csv")
    
    if not os.path.exists(input_path):
        print(f"Categorical scores file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # Calculate errors
    error_df = calculate_category_errors(directory, lower_is_better_metrics=["ari", "dale_chall", "flesch_kincaid"])
    
    # Create plots directory
    plots_dir = os.path.join(directory, "aggregations/plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define categories and their colors
    categories = [
        ('baram_tsabari_score', 'Baram Tsabari Metrics', 'steelblue'),
        ('zemla_score', 'Zemla Metrics', 'forestgreen'),
        ('readability_score', 'Readability Metrics', 'coral')
    ]
    
    # Plot each category
    for score_col, title, color in categories:
        output_path = os.path.join(plots_dir, f"{score_col.replace('_score', '')}_scores.png")
        
        # Pass error_df for all categories; _plot_single_category will only use it for baram_tsabari
        _plot_single_category(df, score_col, title, output_path, color, error_df)


def plot_total_scores(directory: str):
    """
    Plot overall model performance (average across all categories).
    """
    # Read the categorical scores
    input_path = os.path.join(directory, "aggregations/categorical_model_scores.csv")
    
    if not os.path.exists(input_path):
        print(f"Categorical scores file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # Calculate overall average score
    score_columns = ['baram_tsabari_score', 'zemla_score', 'readability_score']
    df['overall_average'] = df[score_columns].mean(axis=1, skipna=True)
    
    # Create plots directory
    plots_dir = os.path.join(directory, "aggregations/plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot overall scores
    output_path = os.path.join(plots_dir, "overall_scores.png")
    _plot_single_category(df, 'overall_average', 'Overall Performance', output_path, 'purple')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_number = sys.argv[1]
    
    os.makedirs(
        f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/aggregations",
        exist_ok=True,
    )
    csv_path = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}"
    lower_is_better_metrics = ["ari", "dale_chall", "flesch_kincaid"]
    aggregate_over_model_metric(
        directory=csv_path,
        lower_is_better_metrics={"ari", "dale_chall", "flesch_kincaid"},
    )
    aggregate_over_model_question(
        directory=csv_path,
        ignore_files=[],
        lower_is_better_metrics=lower_is_better_metrics,
    )
    
    # Generate final model scores
    aggregate_final_model_scores(
        directory=csv_path,
        ignore_files=[],
        lower_is_better_metrics=lower_is_better_metrics,
    )
    
    # Generate categorical model scores
    aggregate_categorical_model_scores(
        directory=csv_path,
        ignore_files=[],
        lower_is_better_metrics=lower_is_better_metrics,
    )
    #
    # Generate plots
    print("\nGenerating plots...")
    plot_categorical_scores(directory=csv_path)
    plot_total_scores(directory=csv_path)
    print("Plotting complete!")
