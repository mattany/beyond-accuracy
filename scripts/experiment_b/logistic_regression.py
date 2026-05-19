#!/usr/bin/env python3
"""
Logistic Regression Analysis: Metric Differences → Human Preference

Two modes:
- binarized: Uses binarized metric scores (threshold 0.5) as predictors
- continuous: Uses raw continuous metric scores as predictors

Two model types:
- difference: A_is_preferred = c1*(metric_a - metric_b) + c2*(metric_a - metric_b) + ...
- interaction: A_is_preferred = c1*metric_a + c2*metric_a*(metric_a - metric_b) + ...

Output: CSV with regression table (coefficients, odds ratios, p-values, etc.)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_NUMBER = 10
DATA_DIR = Path(__file__).parent / "data"
METRICS_DIR = Path(__file__).parent.parent.parent / "Benchmarking" / "deep_eval" / "data" / f"run_{RUN_NUMBER}"

# Clusters to exclude when --exclude-prompted-dpo is set
# Cluster 1: SFT_p vs SynthDPO_p
# Cluster 3: GPT_p vs SynthDPO_p
PROMPTED_DPO_CLUSTERS = [1, 3]

# Cluster 7: Human vs GPT_cot
HUMAN_ANSWER_CLUSTERS = [7]

# Binary threshold
BINARY_THRESHOLD = 0.5

# LLM metrics that get binarized in binarized mode
BINARY_LLM_METRICS = ['metaphor_v8', 'analogy_v2', 'scaffolding_v2']

# Continuous-only metrics (never binarized, even in binarized mode)
CONTINUOUS_ONLY_METRICS = ['jargon']

# Readability metrics (will be aggregated into single 'readability' predictor)
READABILITY_METRICS = ['flesch_reading_ease', 'flesch_kincaid', 'ari', 'dale_chall']

# All metrics to load
ALL_METRICS = BINARY_LLM_METRICS + CONTINUOUS_ONLY_METRICS + READABILITY_METRICS

# Normalization ranges from aggregate_v2.py
# Values are clipped to [0, 1] after normalization
NORMALIZATION_RANGES = {
    "jargon": (0.65, 1.0),  # 0.65 → 0, 1.0 → 1
    "flesch_kincaid": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "ari": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "dale_chall": (7, 12),  # 7th grade → 1, college level → 0
    "flesch_reading_ease": (0.3, 0.7),  # 0.3 (difficult) → 0, 0.7+ (easy) → 1
}

# Metrics where lower values are better (will be inverted during normalization)
LOWER_IS_BETTER = ["ari", "dale_chall", "flesch_kincaid"]

# Model types
MODEL_TYPE_DIFFERENCE = 'difference'  # A_is_preferred = c1*(a-b) + c2*(a-b) + ...
MODEL_TYPE_INTERACTION = 'interaction'  # A_is_preferred = c1*a + c2*a*(a-b) + c3*a + c4*a*(a-b) + ...

# Cluster information for per-cluster analysis
CLUSTER_INFO = {
    0: {"name": "GPT_cot vs SFT", "model_a": "gpt-3.5-turbo-0125_cot", "model_b": "SciComma-3.1-8B_y"},
    1: {"name": "SFT_p vs DPO_p", "model_a": "SciComma-3.1-8B_prompt", "model_b": "scicomma-3.1-dpo_prompt"},
    2: {"name": "DPO vs GPT", "model_a": "scicomma-3.1-dpo", "model_b": "gpt-3.5-turbo-0125"},
    3: {"name": "GPT_cot vs DPO_p", "model_a": "gpt-3.5-turbo-0125_cot", "model_b": "scicomma-3.1-dpo_prompt"},
    4: {"name": "SFT_p vs GPT_cot", "model_a": "SciComma-3.1-8B_prompt", "model_b": "gpt-3.5-turbo-0125_cot"},
    5: {"name": "SFT_p vs OrgSFT_p", "model_a": "SciComma-3.1-8B_prompt", "model_b": "organic_SFT_prompted"},
    6: {"name": "SFT vs Vanilla_p", "model_a": "SciComma-3.1-8B_y", "model_b": "vanilla_prompted"},
    7: {"name": "Human vs GPT_cot", "model_a": "human_answers", "model_b": "gpt-3.5-turbo-0125_cot"},
}


def normalize_score(value: float, metric: str) -> float:
    """
    Normalize a score using the same ranges as aggregate_v2.py.
    
    Args:
        value: Raw metric value
        metric: Metric name
    
    Returns:
        Normalized score in [0, 1] range
    """
    if metric in NORMALIZATION_RANGES:
        min_val, max_val = NORMALIZATION_RANGES[metric]
    else:
        min_val, max_val = 0, 1
    
    if min_val == max_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    
    # Invert for metrics where lower is better
    if metric in LOWER_IS_BETTER:
        normalized = 1 - normalized
    
    return normalized


def load_metric_scores() -> dict:
    """
    Load metric scores from run directory.
    Returns dict: {metric_name: {'a': scores_array, 'b': scores_array}}
    """
    scores = {}
    
    for metric in ALL_METRICS:
        metric_path = METRICS_DIR / f"{metric}.csv"
        if metric_path.exists():
            df = pd.read_csv(metric_path)
            
            score_col_a = 'explanation_a__score'
            score_col_b = 'explanation_b__score'
            
            if score_col_a in df.columns and score_col_b in df.columns:
                scores[metric] = {
                    'a': df[score_col_a].values,
                    'b': df[score_col_b].values
                }
                print(f"  Loaded {metric}: {len(df)} rows")
            else:
                print(f"  Warning: Expected columns not found in {metric}.csv")
        else:
            print(f"  Warning: {metric_path} not found")
    
    return scores


def load_evaluation_data(exclude_prompted_dpo: bool = False, exclude_human: bool = False) -> pd.DataFrame:
    """
    Load the evaluation dataset with human preferences.
    
    Args:
        exclude_prompted_dpo: If True, exclude clusters that compare prompted models with SynthDPO_p
                              (Cluster 1: SFT_p vs SynthDPO_p, Cluster 3: GPT_p vs SynthDPO_p)
        exclude_human: If True, exclude clusters containing human-written answers
                       (Cluster 7: Human vs GPT_cot)
    """
    eval_path = DATA_DIR / "experiment_b_eval_dataset.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"{eval_path} not found. Run run_metrics_exp_b.py first.")
    
    df = pd.read_csv(eval_path)
    
    if exclude_prompted_dpo:
        original_len = len(df)
        df = df[~df['cluster'].isin(PROMPTED_DPO_CLUSTERS)]
        excluded_len = original_len - len(df)
        print(f"  Excluded {excluded_len} comparisons from prompted-DPO clusters (clusters {PROMPTED_DPO_CLUSTERS})")
    
    if exclude_human:
        original_len = len(df)
        df = df[~df['cluster'].isin(HUMAN_ANSWER_CLUSTERS)]
        excluded_len = original_len - len(df)
        print(f"  Excluded {excluded_len} comparisons from human-answer clusters (clusters {HUMAN_ANSWER_CLUSTERS})")
    
    return df


def prepare_data(eval_df: pd.DataFrame, scores: dict, mode: str, no_readability: bool = False,
                 model_type: str = MODEL_TYPE_DIFFERENCE) -> pd.DataFrame:
    """
    Prepare data for logistic regression.
    
    - Binary LLM metrics are binarized in binarized mode
    - Jargon is always continuous (normalized)
    - Readability metrics are normalized and aggregated into a single predictor
    
    Args:
        eval_df: Evaluation dataset with human preferences
        scores: Dict of metric scores
        mode: 'binarized' or 'continuous'
        no_readability: If True, skip readability metrics entirely
        model_type: 'difference' (default) or 'interaction'
            - difference: uses (metric_a - metric_b) as predictors
            - interaction: uses metric_a + metric_a*(metric_a - metric_b) as predictors
    
    Returns:
        DataFrame with feature columns and target variable
    """
    data = []
    
    for idx, row in eval_df.iterrows():
        record = {
            'comparison_id': row['comparison_id'],
            'human_pref': 1 if row['human_choice'] == 'Explanation A' else 0,
        }
        
        # Process binary LLM metrics (binarized in binarized mode)
        for metric in BINARY_LLM_METRICS:
            if metric in scores:
                metric_scores = scores[metric]
                if idx < len(metric_scores['a']) and idx < len(metric_scores['b']):
                    score_a = metric_scores['a'][idx]
                    score_b = metric_scores['b'][idx]
                    
                    if pd.notna(score_a) and pd.notna(score_b):
                        if mode == 'binarized':
                            val_a = 1 if score_a > BINARY_THRESHOLD else 0
                            val_b = 1 if score_b > BINARY_THRESHOLD else 0
                        else:
                            val_a = score_a
                            val_b = score_b
                        
                        diff = val_a - val_b
                        
                        if model_type == MODEL_TYPE_INTERACTION:
                            # Interaction model: metric_a + metric_a * (metric_a - metric_b)
                            record[f'{metric}_a'] = val_a
                            record[f'{metric}_a_x_diff'] = val_a * diff
                        else:
                            # Difference model: just (metric_a - metric_b)
                            record[f'{metric}_diff'] = diff
                    else:
                        if model_type == MODEL_TYPE_INTERACTION:
                            record[f'{metric}_a'] = np.nan
                            record[f'{metric}_a_x_diff'] = np.nan
                        else:
                            record[f'{metric}_diff'] = np.nan
                else:
                    if model_type == MODEL_TYPE_INTERACTION:
                        record[f'{metric}_a'] = np.nan
                        record[f'{metric}_a_x_diff'] = np.nan
                    else:
                        record[f'{metric}_diff'] = np.nan
        
        # Process continuous-only metrics (never binarized, always normalized)
        for metric in CONTINUOUS_ONLY_METRICS:
            if metric in scores:
                metric_scores = scores[metric]
                if idx < len(metric_scores['a']) and idx < len(metric_scores['b']):
                    score_a = metric_scores['a'][idx]
                    score_b = metric_scores['b'][idx]
                    
                    if pd.notna(score_a) and pd.notna(score_b):
                        # Always use normalized continuous values
                        norm_a = normalize_score(score_a, metric)
                        norm_b = normalize_score(score_b, metric)
                        diff = norm_a - norm_b
                        
                        if model_type == MODEL_TYPE_INTERACTION:
                            record[f'{metric}_a'] = norm_a
                            record[f'{metric}_a_x_diff'] = norm_a * diff
                        else:
                            record[f'{metric}_diff'] = diff
                    else:
                        if model_type == MODEL_TYPE_INTERACTION:
                            record[f'{metric}_a'] = np.nan
                            record[f'{metric}_a_x_diff'] = np.nan
                        else:
                            record[f'{metric}_diff'] = np.nan
                else:
                    if model_type == MODEL_TYPE_INTERACTION:
                        record[f'{metric}_a'] = np.nan
                        record[f'{metric}_a_x_diff'] = np.nan
                    else:
                        record[f'{metric}_diff'] = np.nan
        
        # Aggregate readability metrics into single predictor (normalized)
        if not no_readability:
            readability_vals_a = []
            readability_diffs = []
            for metric in READABILITY_METRICS:
                if metric in scores:
                    metric_scores = scores[metric]
                    if idx < len(metric_scores['a']) and idx < len(metric_scores['b']):
                        score_a = metric_scores['a'][idx]
                        score_b = metric_scores['b'][idx]
                        
                        if pd.notna(score_a) and pd.notna(score_b):
                            # Normalize scores using aggregate_v2.py ranges
                            norm_a = normalize_score(score_a, metric)
                            norm_b = normalize_score(score_b, metric)
                            
                            if mode == 'binarized':
                                # Binarize the normalized scores
                                val_a = 1 if norm_a > BINARY_THRESHOLD else 0
                                val_b = 1 if norm_b > BINARY_THRESHOLD else 0
                            else:
                                val_a = norm_a
                                val_b = norm_b
                            
                            readability_vals_a.append(val_a)
                            readability_diffs.append(val_a - val_b)
            
            if readability_diffs:
                avg_a = np.mean(readability_vals_a)
                avg_diff = np.mean(readability_diffs)
                
                if model_type == MODEL_TYPE_INTERACTION:
                    record['readability_a'] = avg_a
                    record['readability_a_x_diff'] = avg_a * avg_diff
                else:
                    record['readability_diff'] = avg_diff
            else:
                if model_type == MODEL_TYPE_INTERACTION:
                    record['readability_a'] = np.nan
                    record['readability_a_x_diff'] = np.nan
                else:
                    record['readability_diff'] = np.nan
        
        # Answer length as a control variable (log-transformed word count)
        text_a = str(row.get('explanation_a', ''))
        text_b = str(row.get('explanation_b', ''))
        log_len_a = np.log1p(len(text_a.split()))
        log_len_b = np.log1p(len(text_b.split()))
        len_diff = log_len_a - log_len_b

        if model_type == MODEL_TYPE_INTERACTION:
            record['answer_length_a'] = log_len_a
            record['answer_length_a_x_diff'] = log_len_a * len_diff
        else:
            record['answer_length_diff'] = len_diff
        
        data.append(record)
    
    return pd.DataFrame(data)


def run_logistic_regression(df: pd.DataFrame, mode: str, model_type: str = MODEL_TYPE_DIFFERENCE) -> tuple[pd.DataFrame, int]:
    """
    Run logistic regression and return results as a DataFrame.
    
    Uses L2 regularization (ridge) to handle perfect/quasi-separation issues,
    which are common in binarized mode.
    
    Args:
        df: DataFrame with feature columns and target variable
        mode: 'binarized' or 'continuous'
        model_type: 'difference' or 'interaction'
    
    Returns:
        Tuple of (DataFrame with regression results, number of samples used)
    """
    # Get feature columns based on model type
    if model_type == MODEL_TYPE_INTERACTION:
        # For interaction model: columns ending with _a or _a_x_diff
        feature_cols = [col for col in df.columns if col.endswith('_a') or col.endswith('_a_x_diff')]
    else:
        # For difference model: columns ending with _diff
        feature_cols = [col for col in df.columns if col.endswith('_diff')]
    
    # Drop rows with NaN
    analysis_df = df.dropna(subset=feature_cols + ['human_pref'])
    
    # Remove zero-variance predictors (common in binarized mode)
    zero_var_cols = [col for col in feature_cols if analysis_df[col].var() == 0]
    if zero_var_cols:
        print(f"  Removing zero-variance predictors: {[c.replace('_diff', '') for c in zero_var_cols]}")
        feature_cols = [col for col in feature_cols if col not in zero_var_cols]
    
    print(f"\nN = {len(analysis_df)} valid comparisons")
    print(f"Human chose A: {analysis_df['human_pref'].sum()} ({analysis_df['human_pref'].mean()*100:.1f}%)")
    
    # Prepare X and y
    X = analysis_df[feature_cols]
    y = analysis_df['human_pref']
    
    # Add constant for intercept
    X_const = sm.add_constant(X)
    
    # Use Logit with regularization to handle potential separation issues
    # We'll use penalized maximum likelihood (Firth-like)
    model = sm.Logit(y, X_const)
    
    # Try standard fit first
    result = None
    has_se = False
    
    try:
        result = model.fit(disp=0, method='bfgs', maxiter=1000)
        # Check for convergence issues (extremely large coefficients indicate separation)
        if np.abs(result.params).max() > 10:
            print("  Note: Large coefficients detected, possible separation issues")
        has_se = hasattr(result, 'bse') and result.bse is not None and not result.bse.isna().any()
    except Exception as e:
        print(f"  Warning: Standard fit failed: {e}")
    
    # If standard fit failed or has issues, use regularized fit
    if result is None or not has_se:
        print("  Using regularized logistic regression (L2 penalty)")
        # Use fit_regularized with L1_wt=0 for pure L2 (ridge)
        # alpha controls regularization strength - smaller = less regularization
        try:
            result = model.fit_regularized(disp=0, alpha=0.1, L1_wt=0)
            has_se = False  # Regularized fit doesn't have standard SEs
        except Exception as e:
            print(f"  Regularized fit also failed: {e}")
            return pd.DataFrame(), 0
    
    # Build results table
    results_data = []
    
    # Get parameter names (excluding const)
    param_names = [col for col in X_const.columns if col != 'const']
    
    for param in ['const'] + param_names:
        coef = result.params[param]
        
        if has_se:
            se = result.bse[param]
            z_stat = result.tvalues[param]
            p_value = result.pvalues[param]
            
            # Confidence intervals
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            se = np.nan
            z_stat = np.nan
            p_value = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        
        # Odds ratio
        odds_ratio = np.exp(coef)
        odds_ci_lower = np.exp(ci_lower) if not np.isnan(ci_lower) else np.nan
        odds_ci_upper = np.exp(ci_upper) if not np.isnan(ci_upper) else np.nan
        
        # Clean up metric name
        metric_name = param.replace('_diff', '') if param != 'const' else 'intercept'
        
        # Significance stars
        if not np.isnan(p_value):
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''
        else:
            sig = ''
        
        results_data.append({
            'metric': metric_name,
            'coefficient': coef,
            'std_error': se,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significance': sig,
            'odds_ratio': odds_ratio,
            'odds_ratio_ci_lower': odds_ci_lower,
            'odds_ratio_ci_upper': odds_ci_upper,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Add n_samples as metadata column
    results_df['n_samples'] = len(analysis_df)
    
    # Add model summary stats
    if has_se:
        print(f"\nModel Summary:")
        print(f"  Pseudo R² (McFadden): {result.prsquared:.4f}")
        print(f"  Log-Likelihood: {result.llf:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
    else:
        print("\n  Note: Standard errors not available for regularized fit.")
        print("  Coefficients represent penalized MLE estimates.")
    
    return results_df, len(analysis_df)


def print_results_table(results_df: pd.DataFrame, mode: str):
    """Print formatted results table to console."""
    print(f"\n{'=' * 90}")
    print(f"LOGISTIC REGRESSION RESULTS ({mode.upper()} MODE)")
    print(f"{'=' * 90}")
    print(f"{'Metric':<20} {'β':>10} {'SE':>8} {'z':>8} {'p':>10} {'':>4} {'OR':>8} {'95% CI':>18}")
    print(f"{'-' * 90}")
    
    for _, row in results_df.iterrows():
        metric = row['metric']
        coef = row['coefficient']
        se = row['std_error']
        z = row['z_statistic']
        p = row['p_value']
        sig = row['significance']
        odds = row['odds_ratio']
        ci_low = row['odds_ratio_ci_lower']
        ci_high = row['odds_ratio_ci_upper']
        
        if np.isnan(se):
            se_str = "N/A"
            z_str = "N/A"
            p_str = "N/A"
            ci_str = "N/A"
        else:
            se_str = f"{se:.4f}"
            z_str = f"{z:.3f}"
            p_str = f"{p:.4f}"
            ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
        
        print(f"{metric:<20} {coef:>10.4f} {se_str:>8} {z_str:>8} {p_str:>10} {sig:>4} {odds:>8.3f} {ci_str:>18}")
    
    print(f"{'=' * 90}")
    print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05")


def create_table_figure(results_df: pd.DataFrame, mode: str, output_path: Path):
    """
    Create a matplotlib table figure from the regression results.
    
    Args:
        results_df: DataFrame with regression results
        mode: 'binarized' or 'continuous'
        output_path: Path to save the figure
    """
    # Filter out intercept for cleaner table
    df = results_df[results_df['metric'] != 'intercept'].copy()
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        metric = row['metric'].replace('_v8', '').replace('_v2', '').replace('_', ' ').title()
        coef = f"{row['coefficient']:.3f}"
        se = f"{row['std_error']:.3f}" if not np.isnan(row['std_error']) else "N/A"
        p_val = row['p_value']
        if np.isnan(p_val):
            p_str = "N/A"
        elif p_val < 0.001:
            p_str = "<.001***"
        elif p_val < 0.01:
            p_str = f"{p_val:.3f}**"
        elif p_val < 0.05:
            p_str = f"{p_val:.3f}*"
        else:
            p_str = f"{p_val:.3f}"
        
        odds = f"{row['odds_ratio']:.2f}"
        ci_low = row['odds_ratio_ci_lower']
        ci_high = row['odds_ratio_ci_upper']
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A"
        
        table_data.append([metric, coef, se, p_str, odds, ci_str])
    
    # Column headers
    columns = ['Metric', 'β', 'SE', 'p-value', 'OR', '95% CI']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#E6E6E6'] * len(columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        # Highlight significant rows
        if row > 0:
            p_val = df.iloc[row - 1]['p_value']
            if not np.isnan(p_val) and p_val < 0.05:
                cell.set_facecolor('#E8F4E8')  # Light green for significant
    
    # Title
    mode_label = "Binarized" if mode == "binarized" else "Continuous"
    ax.set_title(f"Logistic Regression: {mode_label} Mode\n(Predicting Human Preference for Explanation A)", 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved table figure: {output_path}")


def create_combined_table_figure(binarized_df: pd.DataFrame, continuous_df: pd.DataFrame, output_path: Path, n_samples):
    """
    Create a single matplotlib figure with both binarized and continuous tables.
    
    Args:
        binarized_df: DataFrame with binarized regression results
        continuous_df: DataFrame with continuous regression results
        output_path: Path to save the figure
        n_samples: Number of samples used in the regression
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    for ax, (results_df, mode) in zip(axes, [(binarized_df, 'binarized'), (continuous_df, 'continuous')]):
        ax.axis('off')
        ax.axis('tight')
        
        # Filter out intercept
        df = results_df[results_df['metric'] != 'intercept'].copy()
        
        # Prepare table data
        table_data = []
        for _, row in df.iterrows():
            metric = row['metric'].replace('_v8', '').replace('_v2', '').replace('_', ' ').title()
            coef = f"{row['coefficient']:.3f}"
            se = f"{row['std_error']:.3f}" if not np.isnan(row['std_error']) else "N/A"
            p_val = row['p_value']
            if np.isnan(p_val):
                p_str = "N/A"
            elif p_val < 0.001:
                p_str = "<.001***"
            elif p_val < 0.01:
                p_str = f"{p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"{p_val:.3f}*"
            else:
                p_str = f"{p_val:.3f}"
            
            odds = f"{row['odds_ratio']:.2f}"
            ci_low = row['odds_ratio_ci_lower']
            ci_high = row['odds_ratio_ci_upper']
            ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A"
            
            table_data.append([metric, coef, se, p_str, odds, ci_str])
        
        # Column headers
        columns = ['Metric', 'β', 'SE', 'p-value', 'OR', '95% CI']
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colColours=['#E6E6E6'] * len(columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        
        # Bold header and highlight significant rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
            if row > 0:
                p_val = df.iloc[row - 1]['p_value']
                if not np.isnan(p_val) and p_val < 0.05:
                    cell.set_facecolor('#E8F4E8')
        
        # Title for each table
        mode_label = "Binarized" if mode == "binarized" else "Continuous"
        ax.set_title(f"{mode_label} Mode", fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle(f"Logistic Regression: Predicting Human Preference for Explanation A\n(N = {n_samples})", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved combined table figure: {output_path}")


def create_per_cluster_combined_figure(cluster_results: dict, output_path: Path):
    """
    Create a single figure with all 8 cluster regression tables in a 2x4 grid.
    
    Args:
        cluster_results: Dict mapping cluster_id -> (results_df, n_samples, human_a_pct)
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for cluster_id in range(8):
        ax = axes[cluster_id]
        ax.axis('off')
        ax.axis('tight')
        
        if cluster_id not in cluster_results:
            ax.text(0.5, 0.5, f"Cluster {cluster_id}\nNo data", ha='center', va='center', fontsize=12)
            continue
        
        results_df, n_samples, human_a_pct = cluster_results[cluster_id]
        cluster_name = CLUSTER_INFO[cluster_id]["name"]
        
        # Filter out intercept
        df = results_df[results_df['metric'] != 'intercept'].copy()
        
        if len(df) == 0:
            ax.text(0.5, 0.5, f"Cluster {cluster_id}: {cluster_name}\nNo valid results", 
                   ha='center', va='center', fontsize=10)
            continue
        
        # Prepare table data (simplified: metric, β, p, OR)
        table_data = []
        cell_colors = []
        for _, row in df.iterrows():
            metric = row['metric'].replace('_v8', '').replace('_v2', '').replace('_', ' ').title()
            coef = row['coefficient']
            p_val = row['p_value']
            odds = row['odds_ratio']
            
            # Format coefficient
            coef_str = f"{coef:.2f}"
            
            # Format p-value
            if np.isnan(p_val):
                p_str = "N/A"
                sig = False
            elif p_val < 0.001:
                p_str = "<.001***"
                sig = True
            elif p_val < 0.01:
                p_str = f".{int(p_val*1000):03d}**"
                sig = True
            elif p_val < 0.05:
                p_str = f".{int(p_val*1000):03d}*"
                sig = True
            else:
                p_str = f".{int(p_val*1000):03d}"
                sig = False
            
            # Format OR
            odds_str = f"{odds:.2f}"
            
            table_data.append([metric, coef_str, p_str, odds_str])
            
            # Color based on coefficient direction and significance
            if sig:
                if coef > 0:
                    row_color = ['#D5F5E3', '#D5F5E3', '#D5F5E3', '#D5F5E3']  # Light green (positive)
                else:
                    row_color = ['#FADBD8', '#FADBD8', '#FADBD8', '#FADBD8']  # Light red (negative)
            else:
                row_color = ['white', 'white', 'white', 'white']
            cell_colors.append(row_color)
        
        # Column headers
        columns = ['Metric', 'β', 'p', 'OR']
        header_colors = ['#2C3E50'] * len(columns)
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors,
            colColours=header_colors
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        
        # Style header cells
        for col_idx in range(len(columns)):
            table[(0, col_idx)].set_text_props(color='white', fontweight='bold')
        
        # Title with cluster info
        title = f"Cluster {cluster_id}: {cluster_name}\n(N={n_samples}, A chosen: {human_a_pct:.0f}%)"
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
    
    # Main title
    fig.suptitle("Per-Cluster Logistic Regression (Continuous, Difference Model)\n"
                 "Predicting Human Preference for Explanation A from Metric Differences",
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D5F5E3', edgecolor='black', label='Significant positive (p<0.05)'),
        Patch(facecolor='#FADBD8', edgecolor='black', label='Significant negative (p<0.05)'),
        Patch(facecolor='white', edgecolor='black', label='Not significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved per-cluster combined figure: {output_path}")


def run_per_cluster_analysis(scores: dict, mode: str = 'continuous', 
                              model_type: str = MODEL_TYPE_DIFFERENCE,
                              no_readability: bool = False) -> dict:
    """
    Run logistic regression for each cluster separately.
    
    Args:
        scores: Dict of metric scores
        mode: 'binarized' or 'continuous'
        model_type: 'difference' or 'interaction'
        no_readability: If True, exclude readability metrics
    
    Returns:
        Dict mapping cluster_id -> (results_df, n_samples, human_a_pct)
    """
    # Load full evaluation data
    eval_path = DATA_DIR / "experiment_b_eval_dataset.csv"
    full_eval_df = pd.read_csv(eval_path)
    
    cluster_results = {}
    
    for cluster_id in range(8):
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id}: {CLUSTER_INFO[cluster_id]['name']}")
        print(f"{'='*60}")
        
        # Filter to this cluster (keep original index for score lookup)
        cluster_eval_df = full_eval_df[full_eval_df['cluster'] == cluster_id].copy()
        
        if len(cluster_eval_df) == 0:
            print(f"  No data for cluster {cluster_id}")
            continue
        
        # Calculate human preference for A in this cluster
        human_a_count = (cluster_eval_df['human_choice'] == 'Explanation A').sum()
        human_a_pct = human_a_count / len(cluster_eval_df) * 100
        print(f"  Human chose A: {human_a_count}/{len(cluster_eval_df)} ({human_a_pct:.1f}%)")
        
        # Get the original row indices for this cluster to filter metric scores
        # The metric scores are indexed by row position in the original 800-row dataset
        # cluster_eval_df still has its original index from full_eval_df
        cluster_indices = cluster_eval_df.index.values
        
        # Filter metric scores to this cluster
        cluster_scores = {}
        for metric, metric_data in scores.items():
            cluster_scores[metric] = {
                'a': metric_data['a'][cluster_indices],
                'b': metric_data['b'][cluster_indices]
            }
        
        # Reset index for prepare_data (which iterates over idx, row)
        cluster_eval_df_reset = cluster_eval_df.reset_index(drop=True)
        
        # Prepare data
        data_df = prepare_data(cluster_eval_df_reset, cluster_scores, mode, 
                               no_readability=no_readability, model_type=model_type)
        
        # Run regression
        try:
            results_df, n_samples = run_logistic_regression(data_df, mode, model_type=model_type)
            if len(results_df) > 0:
                cluster_results[cluster_id] = (results_df, n_samples, human_a_pct)
                print_results_table(results_df, mode)
            else:
                print(f"  Regression failed for cluster {cluster_id}")
        except Exception as e:
            print(f"  Error running regression for cluster {cluster_id}: {e}")
    
    return cluster_results


def create_all_vs_nodpo_figure(all_df: pd.DataFrame, nodpo_df: pd.DataFrame, output_path: Path, 
                                n_all: int, n_nodpo: int, model_type: str = 'difference',
                                label_b: str = 'Excluding Prompted-DPO'):
    """
    Create a single matplotlib figure with both All data and a filtered subset side by side.
    
    Args:
        all_df: DataFrame with regression results for all data
        nodpo_df: DataFrame with regression results for the filtered subset
        output_path: Path to save the figure
        n_all: Number of samples in all data
        n_nodpo: Number of samples in filtered subset
        model_type: 'difference' or 'interaction'
        label_b: Label for the second (filtered) table
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for ax, (results_df, label, n_samples) in zip(axes, [
        (all_df, 'All Data', n_all), 
        (nodpo_df, label_b, n_nodpo)
    ]):
        ax.axis('off')
        ax.axis('tight')
        
        # Filter out intercept
        df = results_df[results_df['metric'] != 'intercept'].copy()
        
        # Prepare table data
        table_data = []
        for _, row in df.iterrows():
            metric = row['metric'].replace('_v8', '').replace('_v2', '').replace('_', ' ').title()
            # Shorten interaction term names
            metric = metric.replace('A X Diff', 'A×Δ').replace('Diff', 'Δ')
            coef = f"{row['coefficient']:.3f}"
            se = f"{row['std_error']:.3f}" if not np.isnan(row['std_error']) else "N/A"
            p_val = row['p_value']
            if np.isnan(p_val):
                p_str = "N/A"
            elif p_val < 0.001:
                p_str = "<.001***"
            elif p_val < 0.01:
                p_str = f"{p_val:.3f}**"
            elif p_val < 0.05:
                p_str = f"{p_val:.3f}*"
            else:
                p_str = f"{p_val:.3f}"
            
            odds = f"{row['odds_ratio']:.2f}"
            ci_low = row['odds_ratio_ci_lower']
            ci_high = row['odds_ratio_ci_upper']
            ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A"
            
            table_data.append([metric, coef, se, p_str, odds, ci_str])
        
        # Column headers
        columns = ['Metric', 'β', 'SE', 'p-value', 'OR', '95% CI']
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colColours=['#E6E6E6'] * len(columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Bold header and highlight significant rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
            if row > 0:
                p_val = df.iloc[row - 1]['p_value']
                if not np.isnan(p_val) and p_val < 0.05:
                    cell.set_facecolor('#E8F4E8')
        
        # Title for each table
        ax.set_title(f"{label} (N={n_samples})", fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    model_label = "Interaction Model" if model_type == MODEL_TYPE_INTERACTION else "Difference Model"
    fig.suptitle(f"Logistic Regression: {model_label} (Continuous)\nPredicting Human Preference for Explanation A", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved combined All vs No-DPO figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Analysis: Metric Differences → Human Preference"
    )
    parser.add_argument(
        '--mode',
        choices=['binarized', 'continuous', 'both'],
        required=True,
        help="Mode: 'binarized', 'continuous', or 'both' (runs both and creates combined figure)"
    )
    parser.add_argument(
        '--model-type',
        choices=['difference', 'interaction'],
        default='difference',
        help=("Model type: 'difference' (default) uses (a-b) terms; "
              "'interaction' uses a + a*(a-b) terms for each metric")
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV path (default: data/logistic_regression_{mode}.csv)"
    )
    parser.add_argument(
        '--exclude-prompted-dpo',
        action='store_true',
        help="Exclude clusters with prompted DPO comparisons (SFT_p vs SynthDPO_p, GPT_p vs SynthDPO_p)"
    )
    parser.add_argument(
        '--no-readability',
        action='store_true',
        help="Exclude readability metrics from regression (avoids filtering rows with short explanations)"
    )
    parser.add_argument(
        '--exclude-human',
        action='store_true',
        help="Exclude clusters with human-written answers (Human vs GPT_cot)"
    )
    parser.add_argument(
        '--compare-dpo',
        action='store_true',
        help="Run continuous mode for both All data and No-DPO, output combined figure"
    )
    parser.add_argument(
        '--compare-human',
        action='store_true',
        help="Run continuous mode for both All data and No-Human, output combined figure"
    )
    parser.add_argument(
        '--per-cluster',
        action='store_true',
        help="Run regression for each cluster separately and create combined visualization"
    )
    
    args = parser.parse_args()
    
    # Map CLI argument to model type constant
    model_type = MODEL_TYPE_INTERACTION if args.model_type == 'interaction' else MODEL_TYPE_DIFFERENCE
    
    # Check if metrics directory exists
    if not METRICS_DIR.exists():
        print(f"\nError: {METRICS_DIR} not found.")
        print("Run run_metrics_exp_b.py first to generate metric scores.")
        return
    
    if args.per_cluster:
        # Run per-cluster analysis
        print("=" * 90)
        print(f"PER-CLUSTER LOGISTIC REGRESSION ANALYSIS ({args.model_type.upper()} MODEL)")
        if args.no_readability:
            print("(Excluding readability metrics)")
        print("=" * 90)
        
        print(f"\nLoading metrics from run {RUN_NUMBER}...")
        scores = load_metric_scores()
        
        if not scores:
            print("\nNo metric scores found.")
            return
        
        # Create output directory
        per_cluster_dir = DATA_DIR / "per_cluster"
        per_cluster_dir.mkdir(parents=True, exist_ok=True)
        
        # Run per-cluster analysis
        mode = 'continuous'  # Default to continuous for per-cluster
        cluster_results = run_per_cluster_analysis(
            scores, mode=mode, model_type=model_type, no_readability=args.no_readability
        )
        
        # Save individual CSVs
        for cluster_id, (results_df, n_samples, human_a_pct) in cluster_results.items():
            csv_path = per_cluster_dir / f"cluster_{cluster_id}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
        
        # Create combined figure
        combined_path = per_cluster_dir / "per_cluster_combined.png"
        create_per_cluster_combined_figure(cluster_results, combined_path)
        
        # Print summary analysis
        print("\n" + "=" * 90)
        print("SUMMARY: COEFFICIENTS BY CLUSTER")
        print("=" * 90)
        print(f"{'Cluster':<25} {'Metaphor β':>12} {'Readability β':>15} {'N':>6} {'A%':>6}")
        print("-" * 70)
        
        for cluster_id in range(8):
            if cluster_id in cluster_results:
                results_df, n_samples, human_a_pct = cluster_results[cluster_id]
                cluster_name = CLUSTER_INFO[cluster_id]["name"]
                
                # Get metaphor coefficient
                metaphor_row = results_df[results_df['metric'] == 'metaphor_v8']
                if len(metaphor_row) > 0:
                    metaphor_coef = metaphor_row.iloc[0]['coefficient']
                    metaphor_p = metaphor_row.iloc[0]['p_value']
                    metaphor_sig = '*' if metaphor_p < 0.05 else ''
                    metaphor_str = f"{metaphor_coef:+.2f}{metaphor_sig}"
                else:
                    metaphor_str = "N/A"
                
                # Get readability coefficient
                read_row = results_df[results_df['metric'] == 'readability']
                if len(read_row) > 0:
                    read_coef = read_row.iloc[0]['coefficient']
                    read_p = read_row.iloc[0]['p_value']
                    read_sig = '*' if read_p < 0.05 else ''
                    read_str = f"{read_coef:+.2f}{read_sig}"
                else:
                    read_str = "N/A"
                
                print(f"{cluster_id}: {cluster_name:<20} {metaphor_str:>12} {read_str:>15} {n_samples:>6} {human_a_pct:>5.0f}%")
        
        print("=" * 90)
        print("* = p < 0.05")
        return
    
    if args.compare_dpo:
        # Run continuous mode for both All and No-DPO, create combined figure
        print("=" * 90)
        print(f"LOGISTIC REGRESSION: ALL vs NO-DPO COMPARISON ({args.model_type.upper()} MODEL)")
        if args.no_readability:
            print("(Excluding readability metrics)")
        print("=" * 90)
        
        results = {}
        n_samples_dict = {}
        
        for exclude_dpo, label in [(False, 'all'), (True, 'nodpo')]:
            print(f"\n{'='*40}")
            print(f"Running: {'Excluding Prompted-DPO' if exclude_dpo else 'All Data'}")
            print(f"{'='*40}")
            
            print(f"\nLoading metrics from run {RUN_NUMBER}...")
            scores = load_metric_scores()
            
            if not scores:
                print("\nNo metric scores found.")
                return
            
            print(f"\nLoading evaluation data...")
            eval_df = load_evaluation_data(exclude_prompted_dpo=exclude_dpo)
            print(f"Loaded {len(eval_df)} comparisons")
            
            print(f"\nPreparing data (continuous mode, {args.model_type} model)...")
            data_df = prepare_data(eval_df, scores, 'continuous', no_readability=args.no_readability, model_type=model_type)
            
            results_df, n_samples = run_logistic_regression(data_df, 'continuous', model_type=model_type)
            print_results_table(results_df, 'continuous')
            
            results[label] = results_df
            n_samples_dict[label] = n_samples
        
        # Create combined figure
        suffix = ""
        if model_type == MODEL_TYPE_INTERACTION:
            suffix += "_interaction"
        if args.no_readability:
            suffix += "_no_readability"
        combined_path = DATA_DIR / f"logistic_regression_all_vs_nodpo{suffix}.png"
        create_all_vs_nodpo_figure(
            results['all'], results['nodpo'], combined_path,
            n_samples_dict['all'], n_samples_dict['nodpo'],
            model_type=model_type
        )
        return
    
    if args.compare_human:
        print("=" * 90)
        print(f"LOGISTIC REGRESSION: ALL vs NO-HUMAN COMPARISON ({args.model_type.upper()} MODEL)")
        if args.no_readability:
            print("(Excluding readability metrics)")
        print("=" * 90)
        
        results = {}
        n_samples_dict = {}
        
        for exclude_human, label in [(False, 'all'), (True, 'nohuman')]:
            print(f"\n{'='*40}")
            print(f"Running: {'Excluding Human-Answer Clusters' if exclude_human else 'All Data'}")
            print(f"{'='*40}")
            
            print(f"\nLoading metrics from run {RUN_NUMBER}...")
            scores = load_metric_scores()
            
            if not scores:
                print("\nNo metric scores found.")
                return
            
            print(f"\nLoading evaluation data...")
            eval_df = load_evaluation_data(exclude_human=exclude_human)
            print(f"Loaded {len(eval_df)} comparisons")
            
            print(f"\nPreparing data (continuous mode, {args.model_type} model)...")
            data_df = prepare_data(eval_df, scores, 'continuous', no_readability=args.no_readability, model_type=model_type)
            
            results_df, n_samples = run_logistic_regression(data_df, 'continuous', model_type=model_type)
            print_results_table(results_df, 'continuous')
            
            results[label] = results_df
            n_samples_dict[label] = n_samples
        
        suffix = ""
        if model_type == MODEL_TYPE_INTERACTION:
            suffix += "_interaction"
        if args.no_readability:
            suffix += "_no_readability"
        combined_path = DATA_DIR / f"logistic_regression_all_vs_nohuman{suffix}.png"
        create_all_vs_nodpo_figure(
            results['all'], results['nohuman'], combined_path,
            n_samples_dict['all'], n_samples_dict['nohuman'],
            model_type=model_type,
            label_b='Excluding Human Answers'
        )
        return
    
    if args.mode == 'both':
        # Run both modes and create combined figure
        results = {}
        
        for mode in ['binarized', 'continuous']:
            print("=" * 90)
            print(f"LOGISTIC REGRESSION ANALYSIS - {mode.upper()} MODE ({args.model_type.upper()} MODEL)")
            if args.exclude_prompted_dpo:
                print("(Excluding prompted-DPO clusters)")
            if args.no_readability:
                print("(Excluding readability metrics)")
            print("=" * 90)
            
            print(f"\nLoading metrics from run {RUN_NUMBER}...")
            scores = load_metric_scores()
            
            if not scores:
                print("\nNo metric scores found.")
                return
            
            print(f"\nLoading evaluation data...")
            eval_df = load_evaluation_data(exclude_prompted_dpo=args.exclude_prompted_dpo, exclude_human=args.exclude_human)
            print(f"Loaded {len(eval_df)} comparisons")
            
            print(f"\nPreparing data ({mode} mode, {args.model_type} model)...")
            data_df = prepare_data(eval_df, scores, mode, no_readability=args.no_readability, model_type=model_type)
            
            results_df, n_samples = run_logistic_regression(data_df, mode, model_type=model_type)
            print_results_table(results_df, mode)
            
            # Save individual CSV
            suffix = ""
            if model_type == MODEL_TYPE_INTERACTION:
                suffix += "_interaction"
            if args.exclude_prompted_dpo:
                suffix += "_no_prompted_dpo"
            if args.no_readability:
                suffix += "_no_readability"
            output_path = DATA_DIR / f"logistic_regression_{mode}{suffix}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            results[mode] = results_df
        
        # Create combined figure
        suffix = ""
        if model_type == MODEL_TYPE_INTERACTION:
            suffix += "_interaction"
        if args.exclude_prompted_dpo:
            suffix += "_no_prompted_dpo"
        if args.no_readability:
            suffix += "_no_readability"
        combined_path = DATA_DIR / f"logistic_regression_combined{suffix}.png"
        create_combined_table_figure(results['binarized'], results['continuous'], combined_path, n_samples=n_samples)
    else:
        print("=" * 90)
        print(f"LOGISTIC REGRESSION ANALYSIS - {args.mode.upper()} MODE ({args.model_type.upper()} MODEL)")
        if args.exclude_prompted_dpo:
            print("(Excluding prompted-DPO clusters)")
        if args.no_readability:
            print("(Excluding readability metrics)")
        print("=" * 90)
        
        print(f"\nLoading metrics from run {RUN_NUMBER}...")
        scores = load_metric_scores()
        
        if not scores:
            print("\nNo metric scores found.")
            return
        
        print(f"\nLoading evaluation data...")
        eval_df = load_evaluation_data(exclude_prompted_dpo=args.exclude_prompted_dpo, exclude_human=args.exclude_human)
        print(f"Loaded {len(eval_df)} comparisons")
        
        print(f"\nPreparing data ({args.mode} mode, {args.model_type} model)...")
        data_df = prepare_data(eval_df, scores, args.mode, no_readability=args.no_readability, model_type=model_type)
        
        results_df, n_samples = run_logistic_regression(data_df, args.mode, model_type=model_type)
        print_results_table(results_df, args.mode)
        
        # Build output path with suffixes
        suffix = ""
        if model_type == MODEL_TYPE_INTERACTION:
            suffix += "_interaction"
        if args.exclude_prompted_dpo:
            suffix += "_no_prompted_dpo"
        if args.no_readability:
            suffix += "_no_readability"
        
        output_path = args.output or (DATA_DIR / f"logistic_regression_{args.mode}{suffix}.csv")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        fig_path = output_path.with_suffix('.png')
        create_table_figure(results_df, args.mode, fig_path)


if __name__ == "__main__":
    main()

