#!/usr/bin/env python3
"""
Analyze correlation between automated metrics and human preferences.
Methods: Point-biserial correlation + Logistic Regression

This script reads the metric scores from run 10 and correlates them
with human preferences from experiment_b.

Agreement calculation: Binarize scores first (threshold 0.5), then:
- If both A and B have the feature → TIE (excluded from agreement)
- If only A has the feature → Predict A
- If only B has the feature → Predict B
- If neither has the feature → TIE (excluded from agreement)
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
METRICS_DIR = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "results"
    / "preference_metrics"
)

# Metrics to analyze (binary threshold)
BINARY_THRESHOLD = 0.5

# Binary metrics (presence/absence - use binarized agreement)
BINARY_METRICS = ['metaphor_v8', 'humor_v5', 'analogy_v2', 'scaffolding_v2']

# Continuous metrics (use sign of difference for agreement)
CONTINUOUS_METRICS = ['jargon']

# Note: Jargon scores don't differentiate in this dataset (all ~0.86, max diff 0.04)
# So we treat it as continuous without threshold

# All LLM metrics
LLM_METRICS = CONTINUOUS_METRICS + BINARY_METRICS

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
    Computes metric differences (A - B) and binarized scores for each comparison.
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
                    # Also store binarized versions
                    comparison[f'{metric}_bin_a'] = 1 if score_a > BINARY_THRESHOLD else 0
                    comparison[f'{metric}_bin_b'] = 1 if score_b > BINARY_THRESHOLD else 0
                else:
                    comparison[f'{metric}_diff'] = np.nan
                    comparison[f'{metric}_a'] = np.nan
                    comparison[f'{metric}_b'] = np.nan
                    comparison[f'{metric}_bin_a'] = np.nan
                    comparison[f'{metric}_bin_b'] = np.nan
            else:
                comparison[f'{metric}_diff'] = np.nan
        
        comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)


def compute_binarized_agreement(df: pd.DataFrame, metric: str) -> dict:
    """
    Compute agreement using binarized scores.
    
    Returns dict with:
    - agreement: % of non-tie cases where metric prediction matches human
    - n_valid: number of non-tie cases
    - n_ties: number of tie cases (both have or neither has feature)
    - n_only_a: cases where only A has feature
    - n_only_b: cases where only B has feature
    """
    bin_a_col = f'{metric}_bin_a'
    bin_b_col = f'{metric}_bin_b'
    
    if bin_a_col not in df.columns or bin_b_col not in df.columns:
        return None
    
    valid_df = df.dropna(subset=[bin_a_col, bin_b_col, 'human_pref'])
    
    bin_a = valid_df[bin_a_col].values
    bin_b = valid_df[bin_b_col].values
    human_pref = valid_df['human_pref'].values
    
    # Categorize cases
    both_have = (bin_a == 1) & (bin_b == 1)  # TIE
    neither_has = (bin_a == 0) & (bin_b == 0)  # TIE
    only_a_has = (bin_a == 1) & (bin_b == 0)  # Predict A
    only_b_has = (bin_a == 0) & (bin_b == 1)  # Predict B
    
    n_both = both_have.sum()
    n_neither = neither_has.sum()
    n_only_a = only_a_has.sum()
    n_only_b = only_b_has.sum()
    n_ties = n_both + n_neither
    n_valid = n_only_a + n_only_b
    
    if n_valid == 0:
        return {
            'agreement': np.nan,
            'n_valid': 0,
            'n_ties': n_ties,
            'n_both': n_both,
            'n_neither': n_neither,
            'n_only_a': n_only_a,
            'n_only_b': n_only_b,
        }
    
    # Compute agreement on non-tie cases
    # metric_pred = 1 if only_a_has, 0 if only_b_has
    metric_pred = np.where(only_a_has, 1, np.where(only_b_has, 0, np.nan))
    non_tie_mask = ~np.isnan(metric_pred)
    
    agreement = (metric_pred[non_tie_mask] == human_pref[non_tie_mask]).mean() * 100
    
    return {
        'agreement': agreement,
        'n_valid': n_valid,
        'n_ties': n_ties,
        'n_both': n_both,
        'n_neither': n_neither,
        'n_only_a': n_only_a,
        'n_only_b': n_only_b,
    }


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
    # 1. Point-Biserial Correlations with BINARIZED Agreement
    # ==========================================================================
    print("\n" + "=" * 85)
    print("POINT-BISERIAL CORRELATIONS (with Binarized Agreement)")
    print("=" * 85)
    print(f"{'Metric':<15} {'r':>7} {'p':>10} {'Agree%':>8} {'N_valid':>8} {'N_ties':>8} {'(Both/Neither)':<16}")
    print("-" * 85)
    
    for col in available_cols:
        metric_name = col.replace('_diff', '')
        
        # Check if this is a binary or continuous metric
        is_binary = metric_name in BINARY_METRICS
        
        if is_binary:
            # For binary metrics: use binarized scores and exclude ties
            bin_a_col = f'{metric_name}_bin_a'
            bin_b_col = f'{metric_name}_bin_b'
            
            if bin_a_col in analysis_df.columns and bin_b_col in analysis_df.columns:
                bin_a = analysis_df[bin_a_col].values
                bin_b = analysis_df[bin_b_col].values
                
                # Identify non-tie cases (only A has OR only B has)
                only_a_has = (bin_a == 1) & (bin_b == 0)
                only_b_has = (bin_a == 0) & (bin_b == 1)
                non_tie_mask = only_a_has | only_b_has
                
                n_valid = non_tie_mask.sum()
                n_both = ((bin_a == 1) & (bin_b == 1)).sum()
                n_neither = ((bin_a == 0) & (bin_b == 0)).sum()
                n_ties = n_both + n_neither
                
                if n_valid >= 5:
                    # Create binary predictor: 1 = only A has, 0 = only B has
                    metric_pred = only_a_has[non_tie_mask].astype(int)
                    human_pref_valid = analysis_df['human_pref'].values[non_tie_mask]
                    
                    # Point-biserial on binarized, non-tie cases only
                    r, p = stats.pointbiserialr(human_pref_valid, metric_pred)
                    
                    # Agreement
                    agreement = (metric_pred == human_pref_valid).mean() * 100
                    agree_str = f"{agreement:>7.1f}%"
                else:
                    r, p = np.nan, np.nan
                    agree_str = "    N/A"
                
                tie_detail = f"({n_both}/{n_neither})"
            else:
                r, p = np.nan, np.nan
                agree_str = "    N/A"
                n_valid = 0
                n_ties = len(analysis_df)
                tie_detail = ""
        else:
            # For continuous metrics (readability, jargon): use sign of difference
            r, p = stats.pointbiserialr(analysis_df['human_pref'], analysis_df[col])
            metric_pred = (analysis_df[col] > 0).astype(int)
            agree_pct = (metric_pred == analysis_df['human_pref']).mean() * 100
            agree_str = f"{agree_pct:>7.1f}%"
            n_valid = len(analysis_df)
            n_ties = 0
            tie_detail = "(continuous)"
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{metric_name:<15} {r:>6.3f}{sig} {p:>10.4f} {agree_str} {n_valid:>8} {n_ties:>8} {tie_detail:<16}")
    
    # ==========================================================================
    # 2. Logistic Regression
    # ==========================================================================
    print("\n" + "=" * 85)
    print("LOGISTIC REGRESSION (Multivariate Analysis)")
    print("=" * 85)
    
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
    
    print("\n" + "=" * 85)
    print("Notes:")
    print("  - Binary metrics (analogy, metaphor, humor, scaffolding) use BINARIZED scores")
    print("  - Ties (both have OR neither has feature) are EXCLUDED from r and agreement")
    print("  - N_valid = cases where metric differentiates (only A or only B has feature)")
    print("  - Continuous metrics (readability, jargon) use raw diff scores")
    print("=" * 85)
    
    return analysis_df


def main():
    print("=" * 85)
    print("METRIC-HUMAN PREFERENCE CORRELATION ANALYSIS")
    print("=" * 85)
    
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
