import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_consistency(output_dir):
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Load data
    stats_path = os.path.join(output_dir, "consistency_stats.csv")
    summary_path = os.path.join(output_dir, "metric_consistency_summary.csv")
    intermediate_path = os.path.join(output_dir, "intermediate_results.csv")
    
    if not os.path.exists(stats_path) or not os.path.exists(summary_path):
        print(f"Error: Data files not found in {output_dir}")
        return

    stats_df = pd.read_csv(stats_path)
    summary_df = pd.read_csv(summary_path)
    intermediate_df = pd.read_csv(intermediate_path) if os.path.exists(intermediate_path) else None

    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Bar Plot: Average Inconsistency (Std Dev) per Metric
    plt.figure(figsize=(12, 6))
    summary_df_sorted = summary_df.sort_values('avg_std_dev (inconsistency)', ascending=True)
    sns.barplot(
        data=summary_df_sorted,
        x='avg_std_dev (inconsistency)',
        y='metric',
        palette='viridis'
    )
    plt.title('Average Inconsistency (Standard Deviation) by Metric', fontsize=15)
    plt.xlabel('Average Standard Deviation (Lower is Better)', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "avg_inconsistency_per_metric.png"), dpi=300)
    plt.close()
    
    # 2. Box Plot: Distribution of Std Devs across Questions for each Metric
    # This shows if a metric is consistently stable or if it has outliers (some questions are very unstable)
    plt.figure(figsize=(12, 8))
    # Order by median inconsistency
    order = stats_df.groupby('metric')['std_dev'].median().sort_values().index
    sns.boxplot(
        data=stats_df,
        x='std_dev',
        y='metric',
        order=order,
        palette='coolwarm'
    )
    plt.title('Distribution of Instability across Questions', fontsize=15)
    plt.xlabel('Standard Deviation per Question (N=15 Repetitions)', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "inconsistency_distribution.png"), dpi=300)
    plt.close()

    # 3. Scatter Plot: Mean Score vs. Standard Deviation
    # Does the model get more inconsistent when it gives higher/lower scores?
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stats_df,
        x='mean_score',
        y='std_dev',
        hue='metric',
        alpha=0.6,
        palette='deep'
    )
    plt.title('Consistency vs. Score Value', fontsize=15)
    plt.xlabel('Mean Score (0-1)', fontsize=12)
    plt.ylabel('Standard Deviation (Instability)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "score_vs_inconsistency.png"), dpi=300)
    plt.close()
    
    # 4. (Optional) Heatmap of Raw Scores if intermediate data exists
    # Visualizing one random question's repetitions to see the "flicker"
    if intermediate_df is not None:
        # Pick the metric with highest average inconsistency to visualize
        worst_metric = summary_df_sorted.iloc[-1]['metric']
        
        # Filter for that metric
        subset = intermediate_df[intermediate_df['metric'] == worst_metric].copy()
        
        # We need to pivot: Rows=Questions, Cols=Repetitions, Values=Scores
        # Create a label for each question (e.g., "Q1 (Model X)")
        subset['label'] = "Q" + subset['question_idx'].astype(str) + "\n" + subset['model']
        
        pivot_df = subset.pivot(index='label', columns='repetition', values='score')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Score'})
        plt.title(f'Raw Score Variance: {worst_metric} (Most Unstable Metric)', fontsize=15)
        plt.xlabel('Repetition #', fontsize=12)
        plt.ylabel('Question / Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"heatmap_raw_scores_{worst_metric}.png"), dpi=300)
        plt.close()

    print(f"Graphs saved to {graphs_dir}")

if __name__ == "__main__":
    # Define directory
    # Assuming relative path from where script is run, or absolute path
    # Using the path defined in your project structure
    target_dir = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/consistency_check/15_samples_10_tries/"
    visualize_consistency(target_dir)

