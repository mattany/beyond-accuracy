"""
Visualize metric consistency/stability results.

Usage:
    cd Benchmarking/deep_eval && poetry run python custom_metrics/visualize_consistency.py
    poetry run python custom_metrics/visualize_consistency.py --dir /path/to/results
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROJECT_DIR


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
    plt.figure(figsize=(12, 8))
    order = stats_df.groupby('metric')['std_dev'].median().sort_values().index
    sns.boxplot(
        data=stats_df,
        x='std_dev',
        y='metric',
        order=order,
        palette='coolwarm'
    )
    plt.title('Distribution of Instability across Questions', fontsize=15)
    plt.xlabel('Standard Deviation per Question', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "inconsistency_distribution.png"), dpi=300)
    plt.close()

    # 3. Scatter Plot: Mean Score vs. Standard Deviation
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
    
    # 4. Binary Agreement Bar Plot (if available)
    if 'avg_binary_agreement' in summary_df.columns:
        plt.figure(figsize=(12, 6))
        summary_df_sorted_binary = summary_df.sort_values('avg_binary_agreement', ascending=False)
        sns.barplot(
            data=summary_df_sorted_binary,
            x='avg_binary_agreement',
            y='metric',
            palette='RdYlGn'
        )
        plt.title('Binary Decision Agreement (>0.5 threshold)', fontsize=15)
        plt.xlabel('Average Agreement Rate (Higher is Better)', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "binary_agreement.png"), dpi=300)
        plt.close()
    
    # 5. Heatmap of Raw Scores if intermediate data exists
    if intermediate_df is not None:
        # Pick the metric with highest average inconsistency to visualize
        worst_metric = summary_df_sorted.iloc[-1]['metric']
        
        # Filter for that metric
        subset = intermediate_df[intermediate_df['metric'] == worst_metric].copy()
        
        if len(subset) > 0:
            # Create a label for each question
            subset['label'] = "Q" + subset['question_idx'].astype(str)
            if 'model' in subset.columns and subset['model'].notna().any():
                subset['label'] = subset['label'] + "\n" + subset['model'].fillna('')
            
            pivot_df = subset.pivot(index='label', columns='repetition', values='score')
            
            plt.figure(figsize=(12, max(6, len(pivot_df) * 0.5)))
            sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Score'})
            plt.title(f'Raw Score Variance: {worst_metric} (Most Unstable Metric)', fontsize=15)
            plt.xlabel('Repetition #', fontsize=12)
            plt.ylabel('Question', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"heatmap_raw_scores_{worst_metric}.png"), dpi=300)
            plt.close()
    
    # 6. Per-question CI plot for single metric runs
    if len(summary_df) == 1:
        metric_name = summary_df.iloc[0]['metric']
        
        plt.figure(figsize=(14, max(6, len(stats_df) * 0.4)))
        
        # Sort by mean score
        stats_sorted = stats_df.sort_values('mean_score', ascending=True).reset_index(drop=True)
        
        # Plot error bars
        plt.errorbar(
            stats_sorted['mean_score'],
            range(len(stats_sorted)),
            xerr=[stats_sorted['mean_score'] - stats_sorted['ci_lower'],
                  stats_sorted['ci_upper'] - stats_sorted['mean_score']],
            fmt='o',
            capsize=3,
            capthick=1,
            color='steelblue'
        )
        
        # Add vertical line at 0.5 threshold
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Binary threshold (0.5)')
        
        plt.yticks(range(len(stats_sorted)), [f"Q{idx}" for idx in stats_sorted['question_idx']])
        plt.xlabel('Score (Mean ± 95% CI)', fontsize=12)
        plt.ylabel('Question Index', fontsize=12)
        plt.title(f'{metric_name}: Score Distribution with 95% Confidence Intervals', fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"confidence_intervals_{metric_name}.png"), dpi=300)
        plt.close()

    print(f"Graphs saved to {graphs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize metric consistency results")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to consistency check results directory")
    args = parser.parse_args()
    
    if args.dir:
        target_dir = args.dir
    else:
        # Default: find most recent results directory
        base_dir = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/consistency_check/"
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                # Sort by modification time, most recent first
                subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
                target_dir = os.path.join(base_dir, subdirs[0])
                print(f"Using most recent results: {target_dir}")
            else:
                print("No results found. Run consistency_check.py first.")
                return
        else:
            print(f"Results directory not found: {base_dir}")
            return
    
    visualize_consistency(target_dir)


if __name__ == "__main__":
    main()
