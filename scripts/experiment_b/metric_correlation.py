#!/usr/bin/env python3
"""
Analyze correlation between automated metrics and human preferences.
Methods: Point-biserial correlation + Logistic Regression

This script reads the metric scores from run 10 and correlates them
with human preferences from experiment_b.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_NUMBER = 10
DATA_DIR = Path(__file__).parent / "data"
METRICS_DIR = Path(__file__).parent.parent.parent / "Benchmarking" / "deep_eval" / "data" / f"run_{RUN_NUMBER}"

# Metrics to analyze
LLM_METRICS = ['jargon', 'metaphor_v8', 'humor_v5', 'analogy_v2', 'scaffolding_v2']
READABILITY_METRICS = ['flesch_reading_ease', 'flesch_kincaid', 'ari', 'dale_chall']


def load_metric_scores() -> dict:
    """
    Load metric scores from run 10.
    Returns dict: {metric_name: {'a': scores_array, 'b': scores_array}}
    """
    scores = {}
    
    for metric in LLM_METRICS + READABILITY_METRICS:
        metric_path = METRICS_DIR / f"{metric}.csv"
        if metric_path.exists():
            df = pd.read_csv(metric_path)
            
            # Score columns are named: explanation_a__score, explanation_b__score
            score_col_a = 'explanation_a__score'
            score_col_b = 'explanation_b__score'
            
            if score_col_a in df.columns and score_col_b in df.columns:
                scores[metric] = {
                    'a': df[score_col_a].values,
                    'b': df[score_col_b].values
                }
                print(f"  Loaded {metric}: {len(df)} rows")
            else:
                available = [c for c in df.columns if '__score' in c]
                print(f"  Warning: Expected columns not found in {metric}.csv")
                print(f"    Available: {available}")
        else:
            print(f"  Warning: {metric_path} not found")
    
    return scores


def load_evaluation_data() -> pd.DataFrame:
    """Load the evaluation dataset with human preferences."""
    eval_path = DATA_DIR / "experiment_b_eval_dataset.csv"
    if not eval_path.exists():
        print(f"Error: {eval_path} not found. Run run_metrics_exp_b.py first.")
        return None
    
    return pd.read_csv(eval_path)


def prepare_comparison_data(eval_df: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """
    Prepare data for correlation analysis.
    Computes metric differences (A - B) for each comparison.
    """
    comparisons = []
    
    for idx, row in eval_df.iterrows():
        comparison = {
            'comparison_id': row['comparison_id'],
            'human_choice': row['human_choice'],
            'human_pref': 1 if row['human_choice'] == 'Explanation A' else 0,
            'cluster': row['cluster'],
        }
        
        # Calculate metric differences (A - B) for each metric
        for metric, metric_scores in scores.items():
            if idx < len(metric_scores['a']) and idx < len(metric_scores['b']):
                score_a = metric_scores['a'][idx]
                score_b = metric_scores['b'][idx]
                if pd.notna(score_a) and pd.notna(score_b):
                    comparison[f'{metric}_diff'] = score_a - score_b
                    comparison[f'{metric}_a'] = score_a
                    comparison[f'{metric}_b'] = score_b
                else:
                    comparison[f'{metric}_diff'] = np.nan
            else:
                comparison[f'{metric}_diff'] = np.nan
        
        comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)


def analyze_correlations(df: pd.DataFrame):
    """Run correlation analysis: point-biserial + logistic regression."""
    
    # Define feature columns
    metric_cols = [f'{m}_diff' for m in LLM_METRICS]
    
    # Add combined readability
    readability_cols = [f'{m}_diff' for m in READABILITY_METRICS]
    if all(col in df.columns for col in readability_cols):
        df['readability_diff'] = df[readability_cols].mean(axis=1)
        metric_cols = ['readability_diff'] + metric_cols
    
    # Drop rows with any NaN in metrics
    available_cols = [c for c in metric_cols if c in df.columns]
    if not available_cols:
        print("Error: No metric columns found. Run metrics first.")
        return None
    
    analysis_df = df.dropna(subset=available_cols + ['human_pref'])
    
    print(f"\nN = {len(analysis_df)} valid comparisons")
    print(f"Human chose A: {analysis_df['human_pref'].sum()} ({analysis_df['human_pref'].mean()*100:.1f}%)")
    
    # ==========================================================================
    # 1. Point-Biserial Correlations
    # ==========================================================================
    print("\n" + "=" * 65)
    print("POINT-BISERIAL CORRELATIONS (Individual Metric Validity)")
    print("=" * 65)
    print(f"{'Metric':<20} {'r':>8} {'p-value':>12} {'Agreement':>10}")
    print("-" * 65)
    
    for col in available_cols:
        metric_name = col.replace('_diff', '')
        r, p = stats.pointbiserialr(analysis_df['human_pref'], analysis_df[col])
        
        # Agreement: when metric favors A (diff > 0), does human also choose A?
        metric_pred = (analysis_df[col] > 0).astype(int)
        agreement = (metric_pred == analysis_df['human_pref']).mean() * 100
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{metric_name:<20} {r:>7.3f}{sig} {p:>12.4f} {agreement:>9.1f}%")
    
    # ==========================================================================
    # 2. Logistic Regression
    # ==========================================================================
    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION (Multivariate Analysis)")
    print("=" * 65)
    
    X = analysis_df[available_cols].values
    y = analysis_df['human_pref'].values
    
    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    
    # Cross-validation
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')
    print(f"5-fold CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    print(f"\n{'Metric':<20} {'β (std)':>10} {'Odds Ratio':>12} {'Interpretation':<18}")
    print("-" * 65)
    
    for col, coef in sorted(zip(available_cols, lr.coef_[0]), key=lambda x: -abs(x[1])):
        metric_name = col.replace('_diff', '')
        odds = np.exp(coef)
        if coef > 0.1:
            interp = "↑ favors A"
        elif coef < -0.1:
            interp = "↓ favors B"
        else:
            interp = "— weak"
        print(f"{metric_name:<20} {coef:>10.3f} {odds:>12.2f} {interp:<18}")
    
    print("\n" + "=" * 65)
    print("Interpretation:")
    print("  β > 0: When metric favors A, humans also favor A")
    print("  Odds Ratio: For 1 SD increase in metric diff, odds of choosing A")
    print("=" * 65)
    
    return analysis_df


def main():
    print("=" * 65)
    print("METRIC-HUMAN PREFERENCE CORRELATION ANALYSIS")
    print("=" * 65)
    
    # Check if metrics have been run
    if not METRICS_DIR.exists():
        print(f"\nError: {METRICS_DIR} not found.")
        print("Run run_metrics_exp_b.py first to generate metric scores.")
        return
    
    print(f"\nLoading metrics from run {RUN_NUMBER}...")
    scores = load_metric_scores()
    
    if not scores:
        print("\nNo metric scores found. Run run_metrics_exp_b.py first.")
        return
    
    print(f"\nLoading evaluation data...")
    eval_df = load_evaluation_data()
    
    if eval_df is None:
        return
    
    print(f"Loaded {len(eval_df)} comparisons")
    
    print(f"\nPreparing comparison data...")
    comparison_df = prepare_comparison_data(eval_df, scores)
    print(f"Created {len(comparison_df)} comparisons with metric differences")
    
    # Run analysis
    analyze_correlations(comparison_df)
    
    # Save results
    results_path = DATA_DIR / "correlation_results.csv"
    comparison_df.to_csv(results_path, index=False)
    print(f"\nSaved comparison data to: {results_path}")


if __name__ == "__main__":
    main()
