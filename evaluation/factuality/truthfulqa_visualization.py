#!/usr/bin/env python3
"""
TruthfulQA Results Visualization

Creates a publication-ready bar chart comparing base LLaMA 3.1 
to fine-tuned models on TruthfulQA MC2.

MC2 Score Explanation:
- Each question has multiple answer choices (some correct, some incorrect)
- Model assigns probability to each choice via log-likelihood
- MC2 = sum of probabilities on correct answers
- Range: 0 to 1 (higher = more probability mass on correct answers)

Usage:
    python truthfulqa_visualization.py --results-dir ./truthfulqa_results
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats


# Model display names (short labels for the chart)
MODEL_LABELS = {
    'Meta-Llama-3.1-8B-Instruct': 'Base LLaMA 3.1',
    'SciComma-3.1-8B': 'Synthetic SFT',
    'SciComma-3.1-8B-DPO': 'Synthetic DPO',
    'organic_sft': 'Human SFT',
    'organic_dpo': 'Human DPO',
}

# Display order (base first, then trained models)
DISPLAY_ORDER = [
    'Meta-Llama-3.1-8B-Instruct',
    'SciComma-3.1-8B',
    'SciComma-3.1-8B-DPO',
    'organic_sft',
    'organic_dpo',
]


def load_results(results_dir: str) -> dict:
    """Load MC2 scores from checkpoint CSV files."""
    model_scores = {}
    results_path = Path(results_dir)
    
    for filepath in results_path.glob('checkpoint_*.csv'):
        model_name = filepath.stem.replace('checkpoint_', '')
        df = pd.read_csv(filepath)
        
        if 'mc2_score' in df.columns:
            score = df['mc2_score'].mean()
            n_questions = len(df)
            model_scores[model_name] = {
                'score': score,
                'n': n_questions,
                'std': df['mc2_score'].std(),
                'scores': df['mc2_score'].values,  # Keep per-question scores
                'question_idx': df['question_idx'].values,
            }
            print(f"  {model_name}: {score:.4f} (n={n_questions})")
    
    return model_scores


def compute_statistical_significance(model_scores: dict, base_model: str = 'Meta-Llama-3.1-8B-Instruct'):
    """
    Compute statistical significance of each model vs. the base model.
    
    Uses paired tests since all models answer the same questions.
    Returns dict with p-values and effect sizes.
    """
    if base_model not in model_scores:
        print(f"Warning: Base model '{base_model}' not found")
        return {}
    
    base_scores = model_scores[base_model]['scores']
    base_idx = model_scores[base_model]['question_idx']
    
    significance = {}
    
    for model_name, data in model_scores.items():
        if model_name == base_model:
            continue
        
        model_idx = data['question_idx']
        model_sc = data['scores']
        
        # Align by question index (ensure we're comparing same questions)
        common_idx = np.intersect1d(base_idx, model_idx)
        
        base_mask = np.isin(base_idx, common_idx)
        model_mask = np.isin(model_idx, common_idx)
        
        base_aligned = base_scores[base_mask]
        model_aligned = model_sc[model_mask]
        
        if len(base_aligned) != len(model_aligned):
            print(f"Warning: Mismatched questions for {model_name}")
            continue
        
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(model_aligned, base_aligned)
        
        # Wilcoxon signed-rank test (non-parametric)
        # Handle case where differences might all be zero
        differences = model_aligned - base_aligned
        if np.all(differences == 0):
            w_stat, w_pval = 0, 1.0
        else:
            w_stat, w_pval = stats.wilcoxon(differences)
        
        # Effect size: Cohen's d for paired samples
        diff_mean = np.mean(differences)
        diff_std = np.std(differences, ddof=1)
        cohens_d = diff_mean / diff_std if diff_std > 0 else 0
        
        significance[model_name] = {
            't_stat': t_stat,
            't_pval': t_pval,
            'wilcoxon_stat': w_stat,
            'wilcoxon_pval': w_pval,
            'cohens_d': cohens_d,
            'mean_diff': diff_mean,
            'n_questions': len(common_idx),
        }
    
    return significance


def print_significance_table(significance: dict):
    """Print statistical significance results."""
    if not significance:
        return
    
    print("\n" + "=" * 80)
    print("Statistical Significance vs. Base Model (paired tests, same 817 questions)")
    print("=" * 80)
    print(f"{'Model':<20} {'Δ Mean':>10} {'Cohens d':>10} {'t-test p':>12} {'Wilcoxon p':>12} {'Sig?':>6}")
    print("-" * 80)
    
    for model_name in DISPLAY_ORDER:
        if model_name in significance:
            s = significance[model_name]
            label = MODEL_LABELS.get(model_name, model_name)
            
            # Significance markers
            if s['wilcoxon_pval'] < 0.001:
                sig = "***"
            elif s['wilcoxon_pval'] < 0.01:
                sig = "**"
            elif s['wilcoxon_pval'] < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            print(f"{label:<20} {s['mean_diff']:>+10.4f} {s['cohens_d']:>10.3f} "
                  f"{s['t_pval']:>12.2e} {s['wilcoxon_pval']:>12.2e} {sig:>6}")
    
    print("-" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    print("=" * 80)


def export_results_csv(model_scores: dict, significance: dict, output_path: str):
    """Export full results to CSV for reproducibility."""
    rows = []
    
    for model_name in DISPLAY_ORDER:
        if model_name not in model_scores:
            continue
            
        data = model_scores[model_name]
        label = MODEL_LABELS.get(model_name, model_name)
        
        row = {
            'model_name': model_name,
            'display_name': label,
            'mc2_score': round(data['score'], 3),
            'n_questions': data['n'],
            'std': round(data['std'], 4),
        }
        
        # Add significance data if available
        if model_name in significance:
            s = significance[model_name]
            row['delta_base'] = round(s['mean_diff'], 3)
            row['wilcoxon_p'] = round(s['wilcoxon_pval'], 3)
            row['cohens_d'] = round(s['cohens_d'], 3)
            row['t_test_p'] = round(s['t_pval'], 3)
        else:
            # Base model
            row['delta_base'] = None
            row['wilcoxon_p'] = None
            row['cohens_d'] = None
            row['t_test_p'] = None
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Exported results CSV: {output_path}")


def create_comparison_chart(model_scores: dict, output_path: str):
    """Create publication-ready bar chart."""
    
    # Filter and order models
    available = [m for m in DISPLAY_ORDER if m in model_scores]
    scores = [model_scores[m]['score'] for m in available]
    labels = [MODEL_LABELS.get(m, m) for m in available]
    
    if not available:
        print("No model results found!")
        return
    
    # Setup figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors: base model distinct, others grouped by training type
    colors = ['#4A90A4']  # Base model - muted blue
    colors += ['#7CB342', '#558B2F']  # Synthetic - greens
    colors += ['#FB8C00', '#E65100']  # Human - oranges
    colors = colors[:len(available)]
    
    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_ylabel('MC2 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0, max(scores) * 1.15)
    ax.set_title('TruthfulQA Performance: Base vs. Fine-tuned Models',
                 fontsize=14, fontweight='bold', pad=15)
    
    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    
    # Add horizontal reference line at base model score
    if 'Meta-Llama-3.1-8B-Instruct' in model_scores:
        base_score = model_scores['Meta-Llama-3.1-8B-Instruct']['score']
        ax.axhline(y=base_score, color='#4A90A4', linestyle='--', 
                   alpha=0.5, linewidth=1, label='Base model')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()


def print_summary_table(model_scores: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 60)
    print("TruthfulQA MC2 Results Summary")
    print("=" * 60)
    
    # Get base score for comparison
    base_score = model_scores.get('Meta-Llama-3.1-8B-Instruct', {}).get('score', None)
    
    print(f"{'Model':<25} {'MC2 Score':>10} {'vs Base':>10} {'N':>6}")
    print("-" * 60)
    
    for model_name in DISPLAY_ORDER:
        if model_name in model_scores:
            data = model_scores[model_name]
            label = MODEL_LABELS.get(model_name, model_name)
            score = data['score']
            n = data['n']
            
            if base_score and model_name != 'Meta-Llama-3.1-8B-Instruct':
                diff = score - base_score
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "-"
            
            print(f"{label:<25} {score:>10.4f} {diff_str:>10} {n:>6}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TruthfulQA benchmark results'
    )
    # Default to truthfulqa_results folder next to this script
    script_dir = Path(__file__).parent
    default_results = script_dir / 'truthfulqa_results'
    
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default=str(default_results),
        help=(
            'Directory containing checkpoint_*.csv files '
            '(default: evaluation/factuality/truthfulqa_results/)'
        )
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the chart (default: results_dir/truthfulqa_comparison.png)'
    )
    
    args = parser.parse_args()
    
    # Verify directory exists
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        print("Copy your checkpoint_*.csv files to this directory first.")
        return
    
    print(f"Loading results from: {args.results_dir}")
    model_scores = load_results(args.results_dir)
    
    if not model_scores:
        print("No checkpoint files found!")
        return
    
    # Print summary table
    print_summary_table(model_scores)
    
    # Compute and print statistical significance
    significance = compute_statistical_significance(model_scores)
    print_significance_table(significance)
    
    # Export results CSV (for reproducibility)
    csv_path = os.path.join(args.results_dir, 'truthfulqa_results_summary.csv')
    export_results_csv(model_scores, significance, csv_path)
    
    # Generate chart
    output_path = args.output or os.path.join(args.results_dir, 'truthfulqa_comparison.png')
    create_comparison_chart(model_scores, output_path)


if __name__ == '__main__':
    main()

