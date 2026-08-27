#!/usr/bin/env python3
"""
Comprehensive visualization for per-cluster correlation analysis.
Creates combined analysis figure similar to judge_alignment/balanced_dataset/combined_analysis.png

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "combined_cluster_analysis.png"

# Binary threshold for agreement calculation
BINARY_THRESHOLD = 0.5

# Binary metrics (presence/absence - use binarized agreement)
BINARY_METRICS = ['metaphor_v8', 'humor_v5', 'analogy_v2', 'scaffolding_v2']

# Continuous metrics (use sign of difference for agreement)
CONTINUOUS_METRICS = ['jargon']

# Note: Jargon is treated as continuous (no binary threshold)

# Cluster mapping with SHORT names matching the paper
CLUSTER_INFO = {
    0: ("SFT vs GPT_p", "SFT", "GPT_p"),
    1: ("SFT_p vs Synth DPO_p", "SFT_p", "Synth DPO_p"),
    2: ("GPT vs Synth DPO", "GPT", "Synth DPO"),
    3: ("GPT_p vs Synth DPO_p", "GPT_p", "Synth DPO_p"),
    4: ("SFT_p vs GPT_p", "SFT_p", "GPT_p"),
    5: ("SFT_p vs Organic SFT_p", "SFT_p", "Organic SFT_p"),
    6: ("SFT vs Vanilla_p", "SFT", "Vanilla_p"),
    7: ("GPT_p vs Human", "GPT_p", "Human"),
}

# Short names for table
CLUSTER_SHORT = {
    0: "SFT vs GPT_p",
    1: "SFT_p vs SynthDPO_p",
    2: "GPT vs SynthDPO",
    3: "GPT_p vs SynthDPO_p",
    4: "SFT_p vs GPT_p",
    5: "SFT_p vs OrgSFT_p",
    6: "SFT vs Vanilla_p",
    7: "GPT_p vs Human",
}

SYNTH_DPO_CLUSTERS = {1, 2, 3}
NON_DPO_CLUSTERS = {0, 4, 5, 6, 7}

# Colors
DPO_COLOR = '#E74C3C'  # Red for DPO clusters
NON_DPO_COLOR = '#3498DB'  # Blue for non-DPO clusters
POSITIVE_COLOR = '#2ECC71'  # Green for positive correlations
NEGATIVE_COLOR = '#E74C3C'  # Red for negative correlations
NEUTRAL_COLOR = '#95A5A6'  # Gray for non-significant


def load_data():
    """Load correlation results and merge with model info."""
    df = pd.read_csv(DATA_DIR / "correlation_results.csv")
    
    # Load eval dataset to get model_a and model_b columns
    eval_df = pd.read_csv(DATA_DIR / "experiment_b_eval_dataset.csv")
    
    # Merge model columns
    df = df.merge(
        eval_df[['comparison_id', 'model_a', 'model_b']], 
        on='comparison_id', 
        how='left'
    )
    
    return df


def compute_binarized_agreement(df: pd.DataFrame, metric: str) -> dict:
    """
    Compute agreement using binarized scores.
    
    Returns dict with detailed breakdown:
    - agreement: % of non-tie cases where metric prediction matches human
    - n_valid: number of non-tie cases
    - n_ties: number of tie cases (both have or neither has feature)
    - n_only_a: cases where only A has feature
    - n_only_b: cases where only B has feature
    - agree_only_a: agreement % when only A has feature (should predict A)
    - agree_only_b: agreement % when only B has feature (should predict B)
    """
    bin_a_col = f'{metric}_bin_a'
    bin_b_col = f'{metric}_bin_b'
    
    # If binarized columns don't exist, create them from raw scores
    if bin_a_col not in df.columns:
        a_col = f'{metric}_a'
        b_col = f'{metric}_b'
        if a_col not in df.columns or b_col not in df.columns:
            return None
        df = df.copy()
        df[bin_a_col] = (df[a_col] > BINARY_THRESHOLD).astype(float)
        df[bin_b_col] = (df[b_col] > BINARY_THRESHOLD).astype(float)
    
    valid_df = df.dropna(subset=[bin_a_col, bin_b_col, 'human_pref'])
    
    if len(valid_df) == 0:
        return None
    
    bin_a = valid_df[bin_a_col].values
    bin_b = valid_df[bin_b_col].values
    human_pref = valid_df['human_pref'].values
    
    # Categorize cases
    both_have = (bin_a == 1) & (bin_b == 1)
    neither_has = (bin_a == 0) & (bin_b == 0)
    only_a_has = (bin_a == 1) & (bin_b == 0)
    only_b_has = (bin_a == 0) & (bin_b == 1)
    
    n_both = int(both_have.sum())
    n_neither = int(neither_has.sum())
    n_only_a = int(only_a_has.sum())
    n_only_b = int(only_b_has.sum())
    n_ties = n_both + n_neither
    n_valid = n_only_a + n_only_b
    
    # Agreement when only A has feature (predict A=1, check if human_pref=1)
    if n_only_a > 0:
        agree_only_a = (human_pref[only_a_has] == 1).mean() * 100
    else:
        agree_only_a = np.nan
    
    # Agreement when only B has feature (predict B=0, check if human_pref=0)
    if n_only_b > 0:
        agree_only_b = (human_pref[only_b_has] == 0).mean() * 100
    else:
        agree_only_b = np.nan
    
    if n_valid == 0:
        return {
            'agreement': np.nan,
            'n_valid': 0,
            'n_ties': n_ties,
            'n_both': n_both,
            'n_neither': n_neither,
            'n_only_a': n_only_a,
            'n_only_b': n_only_b,
            'agree_only_a': agree_only_a,
            'agree_only_b': agree_only_b,
        }
    
    # Compute overall agreement on non-tie cases
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
        'agree_only_a': agree_only_a,
        'agree_only_b': agree_only_b,
    }


def compute_continuous_agreement(df: pd.DataFrame, metric: str) -> dict:
    """
    Compute agreement for continuous metrics (jargon, readability).
    Uses sign of difference: if diff > 0, predict A; if diff < 0, predict B.
    """
    diff_col = f'{metric}_diff'
    if diff_col not in df.columns:
        return None
    
    valid_df = df.dropna(subset=[diff_col, 'human_pref'])
    if len(valid_df) == 0:
        return None
    
    diff_values = valid_df[diff_col].values
    human_pref = valid_df['human_pref'].values
    
    # Continuous metrics - no threshold, use sign of difference
    n_valid = len(valid_df)
    n_ties = 0
    metric_pred = (diff_values > 0).astype(int)
    agreement = (metric_pred == human_pref).mean() * 100
    
    return {
        'agreement': agreement,
        'n_valid': n_valid,
        'n_ties': n_ties,
        'is_continuous': True,
    }


def compute_cluster_stats(df):
    """Compute statistics for each cluster."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    results = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id].dropna(subset=['human_pref'])
        
        row = {
            'cluster': cluster_id,
            'cluster_name': CLUSTER_SHORT.get(cluster_id, f'Cluster {cluster_id}'),
            'is_dpo': cluster_id in SYNTH_DPO_CLUSTERS,
            'n': len(cluster_df),
            'human_a_pct': cluster_df['human_pref'].mean() * 100,
        }
        
        for metric in metrics:
            diff_col = f'{metric}_diff'
            metric_df = cluster_df.dropna(subset=[diff_col])
            
            if len(metric_df) >= 5:
                # Check if binary or continuous metric
                if metric in CONTINUOUS_METRICS:
                    # Continuous metric (jargon) - use continuous diff for correlation
                    r, p = stats.pointbiserialr(metric_df['human_pref'], metric_df[diff_col])
                    row[f'{metric}_r'] = r
                    row[f'{metric}_p'] = p
                    
                    cont_stats = compute_continuous_agreement(metric_df, metric)
                    if cont_stats:
                        row[f'{metric}_agree'] = cont_stats['agreement']
                        row[f'{metric}_n_valid'] = cont_stats['n_valid']
                        row[f'{metric}_n_ties'] = 0
                        row[f'{metric}_n_only_a'] = 0
                        row[f'{metric}_n_only_b'] = 0
                        row[f'{metric}_agree_only_a'] = np.nan
                        row[f'{metric}_agree_only_b'] = np.nan
                        row[f'{metric}_is_continuous'] = True
                    else:
                        row[f'{metric}_agree'] = np.nan
                        row[f'{metric}_n_valid'] = 0
                        row[f'{metric}_n_ties'] = 0
                        row[f'{metric}_n_only_a'] = 0
                        row[f'{metric}_n_only_b'] = 0
                        row[f'{metric}_agree_only_a'] = np.nan
                        row[f'{metric}_agree_only_b'] = np.nan
                        row[f'{metric}_is_continuous'] = True
                else:
                    # Binary metric - use binarized scores, exclude ties
                    bin_a_col = f'{metric}_bin_a'
                    bin_b_col = f'{metric}_bin_b'
                    
                    if bin_a_col in metric_df.columns and bin_b_col in metric_df.columns:
                        bin_a = metric_df[bin_a_col].values
                        bin_b = metric_df[bin_b_col].values
                        
                        # Non-tie cases only
                        only_a_has = (bin_a == 1) & (bin_b == 0)
                        only_b_has = (bin_a == 0) & (bin_b == 1)
                        non_tie_mask = only_a_has | only_b_has
                        n_valid = non_tie_mask.sum()
                        
                        if n_valid >= 5:
                            # Binarized predictor: 1 = only A has, 0 = only B has
                            metric_pred = only_a_has[non_tie_mask].astype(int)
                            human_pref_valid = metric_df['human_pref'].values[non_tie_mask]
                            
                            r, p = stats.pointbiserialr(human_pref_valid, metric_pred)
                            row[f'{metric}_r'] = r
                            row[f'{metric}_p'] = p
                        else:
                            row[f'{metric}_r'] = np.nan
                            row[f'{metric}_p'] = np.nan
                        
                        # Binarized agreement stats
                        bin_stats = compute_binarized_agreement(metric_df, metric)
                        if bin_stats:
                            row[f'{metric}_agree'] = bin_stats['agreement']
                            row[f'{metric}_n_valid'] = bin_stats['n_valid']
                            row[f'{metric}_n_ties'] = bin_stats['n_ties']
                            row[f'{metric}_n_only_a'] = bin_stats['n_only_a']
                            row[f'{metric}_n_only_b'] = bin_stats['n_only_b']
                            row[f'{metric}_agree_only_a'] = bin_stats['agree_only_a']
                            row[f'{metric}_agree_only_b'] = bin_stats['agree_only_b']
                            row[f'{metric}_is_continuous'] = False
                        else:
                            row[f'{metric}_agree'] = np.nan
                            row[f'{metric}_n_valid'] = 0
                            row[f'{metric}_n_ties'] = len(metric_df)
                            row[f'{metric}_n_only_a'] = 0
                            row[f'{metric}_n_only_b'] = 0
                            row[f'{metric}_agree_only_a'] = np.nan
                            row[f'{metric}_agree_only_b'] = np.nan
                            row[f'{metric}_is_continuous'] = False
                    else:
                        row[f'{metric}_r'] = np.nan
                        row[f'{metric}_p'] = np.nan
                        row[f'{metric}_agree'] = np.nan
                        row[f'{metric}_n_valid'] = 0
                        row[f'{metric}_n_ties'] = len(metric_df)
                        row[f'{metric}_n_only_a'] = 0
                        row[f'{metric}_n_only_b'] = 0
                        row[f'{metric}_agree_only_a'] = np.nan
                        row[f'{metric}_agree_only_b'] = np.nan
                        row[f'{metric}_is_continuous'] = False
            else:
                row[f'{metric}_r'] = np.nan
                row[f'{metric}_p'] = np.nan
                row[f'{metric}_agree'] = np.nan
                row[f'{metric}_n_valid'] = 0
                row[f'{metric}_n_ties'] = 0
                row[f'{metric}_n_only_a'] = 0
                row[f'{metric}_n_only_b'] = 0
                row[f'{metric}_agree_only_a'] = np.nan
                row[f'{metric}_agree_only_b'] = np.nan
                row[f'{metric}_is_continuous'] = metric in CONTINUOUS_METRICS
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_aggregate_stats(df, clusters=None):
    """Compute aggregate statistics for a set of clusters."""
    if clusters is not None:
        df = df[df['cluster'].isin(clusters)]
    
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    results = {}
    
    for metric in metrics:
        diff_col = f'{metric}_diff'
        metric_df = df.dropna(subset=['human_pref', diff_col])
        
        if len(metric_df) >= 5:
            # Check if binary or continuous metric
            if metric in CONTINUOUS_METRICS:
                # Continuous metric (jargon) - use continuous diff
                r, p = stats.pointbiserialr(metric_df['human_pref'], metric_df[diff_col])
                cont_stats = compute_continuous_agreement(metric_df, metric)
                if cont_stats:
                    agree = cont_stats['agreement']
                    n_valid = cont_stats['n_valid']
                else:
                    agree = np.nan
                    n_valid = 0
            else:
                # Binary metric - use binarized scores, exclude ties
                bin_a_col = f'{metric}_bin_a'
                bin_b_col = f'{metric}_bin_b'
                
                if bin_a_col in metric_df.columns and bin_b_col in metric_df.columns:
                    bin_a = metric_df[bin_a_col].values
                    bin_b = metric_df[bin_b_col].values
                    
                    only_a_has = (bin_a == 1) & (bin_b == 0)
                    only_b_has = (bin_a == 0) & (bin_b == 1)
                    non_tie_mask = only_a_has | only_b_has
                    n_valid = non_tie_mask.sum()
                    
                    if n_valid >= 5:
                        metric_pred = only_a_has[non_tie_mask].astype(int)
                        human_pref_valid = metric_df['human_pref'].values[non_tie_mask]
                        r, p = stats.pointbiserialr(human_pref_valid, metric_pred)
                    else:
                        r, p = np.nan, np.nan
                    
                    bin_stats = compute_binarized_agreement(metric_df, metric)
                    agree = bin_stats['agreement'] if bin_stats else np.nan
                else:
                    r, p = np.nan, np.nan
                    agree = np.nan
                    n_valid = 0
            
            results[metric] = {'r': r, 'p': p, 'agree': agree, 'n': len(metric_df), 'n_valid': n_valid}
    
    return results


def plot_correlation_bars(ax, cluster_stats, metric, title):
    """Plot per-cluster correlation bars for a metric."""
    clusters = cluster_stats['cluster'].values
    r_values = cluster_stats[f'{metric}_r'].values
    p_values = cluster_stats[f'{metric}_p'].values
    
    # Use single color for all bars
    colors = ['#3498DB' if pd.notna(r) else NEUTRAL_COLOR for r in r_values]
    
    bars = ax.bar(range(len(clusters)), r_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add significance markers
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if pd.notna(p):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = ''
            
            if marker:
                height = bar.get_height()
                y_pos = height + 0.02 if height >= 0 else height - 0.08
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, marker, 
                       ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters], fontsize=8)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Point-Biserial r')
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3, linewidth=0.8)


def plot_agreement_bars(ax, cluster_stats, metric, title):
    """Plot per-cluster binarized agreement (total, regardless of which model has feature)."""
    clusters = cluster_stats['cluster'].values
    agree_values = cluster_stats[f'{metric}_agree'].values
    n_valid = cluster_stats[f'{metric}_n_valid'].values
    
    # Use single color for all bars
    colors = ['#3498DB' if pd.notna(a) and n > 0 else NEUTRAL_COLOR 
              for a, n in zip(agree_values, n_valid)]
    
    bars = ax.bar(range(len(clusters)), agree_values, color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters], fontsize=8)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Agreement %')
    ax.set_title(title)
    ax.set_ylim(0, 100)
    
    # Add value labels with N
    for bar, agree, n in zip(bars, agree_values, n_valid):
        if pd.notna(agree) and n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{agree:.0f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=6)
        elif n == 0:
            ax.text(bar.get_x() + bar.get_width()/2, 5, 'n=0', 
                   ha='center', va='bottom', fontsize=6, color='gray')


def plot_aggregate_comparison(ax, df):
    """Plot aggregate correlation for all clusters."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    metric_labels = ['Analogy', 'Metaphor', 'Scaffolding', 'Humor', 'Jargon']
    
    all_stats = compute_aggregate_stats(df)
    
    x = np.arange(len(metrics))
    all_r = [all_stats.get(m, {}).get('r', np.nan) for m in metrics]
    
    ax.bar(x, all_r, color='#3498DB', edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Point-Biserial r')
    ax.set_title('B) Aggregate Correlation')
    ax.set_ylim(-0.5, 0.3)


def plot_agreement_comparison(ax, df):
    """Plot binarized agreement for all clusters."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    metric_labels = ['Analogy', 'Metaphor', 'Scaffolding', 'Humor', 'Jargon']
    
    all_stats = compute_aggregate_stats(df)
    
    x = np.arange(len(metrics))
    all_agree = [all_stats.get(m, {}).get('agree', np.nan) for m in metrics]
    
    ax.bar(x, all_agree, color='#3498DB', edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Agreement %')
    ax.set_title('C) Aggregate Agreement')
    ax.set_ylim(0, 100)


def plot_cluster_heatmap(ax, cluster_stats):
    """Plot heatmap of correlations by cluster and metric."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    metric_labels = ['Analogy', 'Metaphor', 'Scaffold.', 'Humor', 'Jargon']
    
    data = []
    for _, row in cluster_stats.iterrows():
        data.append([row.get(f'{m}_r', np.nan) for m in metrics])
    
    data = np.array(data)
    n_clusters = len(cluster_stats)
    n_metrics = len(metrics)
    
    # Use pcolormesh for precise grid alignment (edges at integer boundaries)
    # Data is plotted with edges at 0, 1, 2, ... so cell centers are at 0.5, 1.5, etc.
    im = ax.pcolormesh(data, cmap='RdBu_r', vmin=-0.7, vmax=0.7, edgecolors='white', linewidth=1)
    
    cluster_labels = []
    for _, row in cluster_stats.iterrows():
        label = f"{int(row['cluster'])}"
        if row['is_dpo']:
            label += "*"
        cluster_labels.append(label)
    
    # Set ticks at cell centers (0.5, 1.5, 2.5, ...)
    ax.set_xticks(np.arange(n_metrics) + 0.5)
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_yticks(np.arange(n_clusters) + 0.5)
    ax.set_yticklabels(cluster_labels, fontsize=8)
    
    ax.set_xlim(0, n_metrics)
    ax.set_ylim(0, n_clusters)
    ax.invert_yaxis()  # Put cluster 0 at top
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Cluster (* = Synth DPO)')
    ax.set_title('D) Correlation Heatmap')
    
    # Add text annotations at cell centers
    for i in range(n_clusters):
        for j in range(n_metrics):
            val = data[i, j]
            if pd.notna(val):
                color = 'white' if abs(val) > 0.35 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('r', fontsize=8)


def plot_human_preference_by_cluster(ax, df):
    """Plot human preference for canonical 'first model' in each cluster."""
    # Define canonical first model for each cluster (matches CLUSTER_SHORT naming)
    CANONICAL_FIRST_MODEL = {
        0: 'SciComma-3.1-8B_y',        # SFT vs GPT_p
        1: 'SciComma-3.1-8B_prompt',   # SFT_p vs SynthDPO_p
        2: 'gpt-3.5-turbo-0125',       # GPT vs SynthDPO
        3: 'gpt-3.5-turbo-0125_cot',   # GPT_p vs SynthDPO_p
        4: 'SciComma-3.1-8B_prompt',   # SFT_p vs GPT_p
        5: 'SciComma-3.1-8B_prompt',   # SFT_p vs OrgSFT_p
        6: 'SciComma-3.1-8B_y',        # SFT vs Vanilla_p
        7: 'gpt-3.5-turbo-0125_cot',   # GPT_p vs Human
    }
    
    # For each row, determine if human chose the canonical first model
    def chose_first_model(row):
        canonical_first = CANONICAL_FIRST_MODEL.get(row['cluster'])
        if canonical_first is None:
            return np.nan
        # Check if human chose the model that matches canonical first
        if row['human_pref'] == 1:  # Chose explanation A
            return 1 if row['model_a'] == canonical_first else 0
        else:  # Chose explanation B
            return 1 if row['model_b'] == canonical_first else 0
    
    df = df.copy()
    df['chose_first_model'] = df.apply(chose_first_model, axis=1)
    
    cluster_stats = df.groupby('cluster').agg({
        'chose_first_model': ['mean', 'count']
    }).reset_index()
    cluster_stats.columns = ['cluster', 'first_model_pct', 'n']
    cluster_stats['first_model_pct'] *= 100
    cluster_stats['is_dpo'] = cluster_stats['cluster'].isin(SYNTH_DPO_CLUSTERS)
    
    colors = [DPO_COLOR if dpo else NON_DPO_COLOR for dpo in cluster_stats['is_dpo']]
    
    bars = ax.bar(range(len(cluster_stats)), cluster_stats['first_model_pct'], color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(cluster_stats)))
    ax.set_xticklabels([str(c) for c in cluster_stats['cluster']], fontsize=8)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Chose 1st Model (%)')
    ax.set_title('E) Human Chose First Model in Pair')
    ax.set_ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.0f}%', 
               ha='center', va='bottom', fontsize=7)


def plot_aggregate_stats_summary(ax, df):
    """Plot aggregate statistics (All, DPO, Non-DPO) for key metrics."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2']
    metric_labels = ['Analogy', 'Metaphor', 'Scaffolding']
    
    # Compute aggregate stats
    groups = {
        'All': None,
        'Synth DPO': SYNTH_DPO_CLUSTERS,
        'Non-DPO': [c for c in range(8) if c not in SYNTH_DPO_CLUSTERS]
    }
    
    results = {metric: {} for metric in metrics}
    
    for metric in metrics:
        for group_name, clusters in groups.items():
            if clusters is not None:
                group_df = df[df['cluster'].isin(clusters)]
            else:
                group_df = df
            
            diff_col = f'{metric}_diff'
            valid_df = group_df.dropna(subset=['human_pref', diff_col])
            
            if len(valid_df) >= 5:
                # Use binarized scores for binary metrics (all these are binary)
                bin_a_col = f'{metric}_bin_a'
                bin_b_col = f'{metric}_bin_b'
                
                if bin_a_col in valid_df.columns and bin_b_col in valid_df.columns:
                    bin_a = valid_df[bin_a_col].values
                    bin_b = valid_df[bin_b_col].values
                    
                    only_a_has = (bin_a == 1) & (bin_b == 0)
                    only_b_has = (bin_a == 0) & (bin_b == 1)
                    non_tie_mask = only_a_has | only_b_has
                    n_valid = non_tie_mask.sum()
                    
                    if n_valid >= 5:
                        metric_pred = only_a_has[non_tie_mask].astype(int)
                        human_pref_valid = valid_df['human_pref'].values[non_tie_mask]
                        r, p = stats.pointbiserialr(human_pref_valid, metric_pred)
                        results[metric][group_name] = {'r': r, 'p': p}
                    else:
                        results[metric][group_name] = {'r': np.nan, 'p': np.nan}
                else:
                    results[metric][group_name] = {'r': np.nan, 'p': np.nan}
            else:
                results[metric][group_name] = {'r': np.nan, 'p': np.nan}
    
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = {'All': '#3498DB', 'Synth DPO': DPO_COLOR, 'Non-DPO': NON_DPO_COLOR}
    
    for i, (group_name, color) in enumerate(colors.items()):
        r_values = [results[m][group_name]['r'] for m in metrics]
        p_values = [results[m][group_name]['p'] for m in metrics]
        
        bars = ax.bar(x + (i - 1) * width, r_values, width, label=group_name,
                      color=color, edgecolor='black', linewidth=0.5)
        
        # Add significance markers
        for j, (bar, p) in enumerate(zip(bars, p_values)):
            if pd.notna(p):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if sig:
                    ax.annotate(sig, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel('Correlation (r)')
    ax.set_title('L) Aggregate Correlations by Group')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-0.45, 0.2)


def plot_cluster_info_table(ax, cluster_stats):
    """Plot cluster info as a table with short model names."""
    ax.axis('off')
    
    headers = ['ID', 'Models', 'N']
    data = []
    for _, row in cluster_stats.iterrows():
        data.append([
            str(int(row['cluster'])),
            row['cluster_name'],
            str(int(row['n'])),
        ])
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='left',
                     colWidths=[0.08, 0.35, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('A) Cluster Information', fontsize=10, fontweight='bold', y=0.98)


def create_combined_figure(df, cluster_stats):
    """Create the combined analysis figure."""
    fig = plt.figure(figsize=(18, 16))
    
    # Use GridSpec with 5 columns for the correlation/agreement rows
    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.25, 
                          height_ratios=[1.2, 1, 1, 1],
                          left=0.04, right=0.98, top=0.94, bottom=0.04)
    
    # Row 1: Cluster info table + Aggregate comparisons + Human preference
    ax1 = fig.add_subplot(gs[0, 0])
    plot_cluster_info_table(ax1, cluster_stats)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_aggregate_comparison(ax2, df)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_agreement_comparison(ax3, df)
    
    ax4 = fig.add_subplot(gs[0, 3:5])
    plot_human_preference_by_cluster(ax4, df)
    
    # Row 2: Correlation heatmap (centered, narrower width)
    ax5 = fig.add_subplot(gs[1, 1:4])
    plot_cluster_heatmap(ax5, cluster_stats)
    
    # Row 3: Per-cluster correlations (5 metrics)
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    metric_names = ['Analogy', 'Metaphor', 'Scaffolding', 'Humor', 'Jargon']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = fig.add_subplot(gs[2, i])
        plot_correlation_bars(ax, cluster_stats, metric, f'{name} Correlation')
    
    # Row 4: Per-cluster agreement (5 metrics)
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = fig.add_subplot(gs[3, i])
        plot_agreement_bars(ax, cluster_stats, metric, f'{name} Agreement')
    
    fig.suptitle('Experiment B: Per-Cluster Metric-Human Preference Analysis', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    return fig


def save_statistics_csv(df, cluster_stats):
    """Save detailed statistics to CSV files."""
    metrics = ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']
    
    # =========================================================================
    # 1. Per-cluster statistics CSV
    # =========================================================================
    cluster_rows = []
    for _, row in cluster_stats.iterrows():
        cluster_row = {
            'cluster': int(row['cluster']),
            'cluster_name': row['cluster_name'],
            'is_synth_dpo': row['is_dpo'],
            'n_comparisons': int(row['n']),
            'human_chose_A_pct': row['human_a_pct'],
        }
        
        for metric in metrics:
            cluster_row[f'{metric}_r'] = row.get(f'{metric}_r', np.nan)
            cluster_row[f'{metric}_p'] = row.get(f'{metric}_p', np.nan)
            cluster_row[f'{metric}_agreement_pct'] = row.get(f'{metric}_agree', np.nan)
            cluster_row[f'{metric}_n_valid'] = row.get(f'{metric}_n_valid', 0)
            cluster_row[f'{metric}_n_ties'] = row.get(f'{metric}_n_ties', 0)
            cluster_row[f'{metric}_n_only_a'] = row.get(f'{metric}_n_only_a', 0)
            cluster_row[f'{metric}_n_only_b'] = row.get(f'{metric}_n_only_b', 0)
            cluster_row[f'{metric}_agree_only_a'] = row.get(f'{metric}_agree_only_a', np.nan)
            cluster_row[f'{metric}_agree_only_b'] = row.get(f'{metric}_agree_only_b', np.nan)
        
        cluster_rows.append(cluster_row)
    
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_csv_path = DATA_DIR / "cluster_statistics.csv"
    cluster_df.to_csv(cluster_csv_path, index=False)
    print(f">>> Saved: {cluster_csv_path}")
    
    # =========================================================================
    # 2. Aggregate statistics CSV (All / DPO / Non-DPO)
    # =========================================================================
    all_stats = compute_aggregate_stats(df)
    dpo_stats = compute_aggregate_stats(df, SYNTH_DPO_CLUSTERS)
    non_dpo_stats = compute_aggregate_stats(df, NON_DPO_CLUSTERS)
    
    agg_rows = []
    for metric in metrics:
        for group_name, group_stats in [('all', all_stats), ('synth_dpo_only', dpo_stats), ('non_dpo', non_dpo_stats)]:
            s = group_stats.get(metric, {})
            agg_rows.append({
                'metric': metric,
                'group': group_name,
                'r': s.get('r', np.nan),
                'p': s.get('p', np.nan),
                'agreement_pct': s.get('agree', np.nan),
                'n_total': s.get('n', 0),
                'n_valid': s.get('n_valid', 0),
            })
    
    agg_df = pd.DataFrame(agg_rows)
    agg_csv_path = DATA_DIR / "aggregate_statistics.csv"
    agg_df.to_csv(agg_csv_path, index=False)
    print(f">>> Saved: {agg_csv_path}")
    
    # =========================================================================
    # 3. Summary statistics CSV (one row per metric, wide format)
    # =========================================================================
    summary_rows = []
    for metric in metrics:
        all_s = all_stats.get(metric, {})
        dpo_s = dpo_stats.get(metric, {})
        non_dpo_s = non_dpo_stats.get(metric, {})
        
        # Compute significance stars
        p_all = all_s.get('p', np.nan)
        sig = '***' if p_all < 0.001 else '**' if p_all < 0.01 else '*' if p_all < 0.05 else ''
        
        summary_rows.append({
            'metric': metric,
            'all_r': all_s.get('r', np.nan),
            'all_p': p_all,
            'all_sig': sig,
            'all_agreement_pct': all_s.get('agree', np.nan),
            'all_n_valid': all_s.get('n_valid', 0),
            'all_n_total': all_s.get('n', 0),
            'dpo_r': dpo_s.get('r', np.nan),
            'dpo_p': dpo_s.get('p', np.nan),
            'dpo_agreement_pct': dpo_s.get('agree', np.nan),
            'dpo_n_valid': dpo_s.get('n_valid', 0),
            'non_dpo_r': non_dpo_s.get('r', np.nan),
            'non_dpo_p': non_dpo_s.get('p', np.nan),
            'non_dpo_agreement_pct': non_dpo_s.get('agree', np.nan),
            'non_dpo_n_valid': non_dpo_s.get('n_valid', 0),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = DATA_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f">>> Saved: {summary_csv_path}")
    
    return cluster_df, agg_df, summary_df


def main():
    print("Loading data...")
    df = load_data()
    
    print("Computing cluster statistics...")
    cluster_stats = compute_cluster_stats(df)
    
    print("Creating combined figure...")
    fig = create_combined_figure(df, cluster_stats)
    
    print(f"Saving to {OUTPUT_PATH}...")
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n>>> Saved: {OUTPUT_PATH}")
    
    # Save CSV files
    print("\nSaving statistics to CSV...")
    cluster_df, agg_df, summary_df = save_statistics_csv(df, cluster_stats)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Binarized Agreement)")
    print("=" * 80)
    
    all_stats = compute_aggregate_stats(df)
    dpo_stats = compute_aggregate_stats(df, SYNTH_DPO_CLUSTERS)
    non_dpo_stats = compute_aggregate_stats(df, NON_DPO_CLUSTERS)
    
    print(f"\n{'Metric':<15} {'All r':>8} {'DPO r':>8} {'Non-DPO r':>10} | {'All Agree':>10} {'DPO Agree':>10} {'Non-DPO':>10}")
    print("-" * 85)
    for metric in ['analogy_v2', 'metaphor_v8', 'scaffolding_v2', 'humor_v5', 'jargon']:
        all_r = all_stats.get(metric, {}).get('r', np.nan)
        dpo_r = dpo_stats.get(metric, {}).get('r', np.nan)
        non_dpo_r = non_dpo_stats.get(metric, {}).get('r', np.nan)
        
        all_agree = all_stats.get(metric, {}).get('agree', np.nan)
        dpo_agree = dpo_stats.get(metric, {}).get('agree', np.nan)
        non_dpo_agree = non_dpo_stats.get(metric, {}).get('agree', np.nan)
        
        all_agree_str = f"{all_agree:.1f}%" if pd.notna(all_agree) else "N/A"
        dpo_agree_str = f"{dpo_agree:.1f}%" if pd.notna(dpo_agree) else "N/A"
        non_dpo_agree_str = f"{non_dpo_agree:.1f}%" if pd.notna(non_dpo_agree) else "N/A"
        
        print(f"{metric:<15} {all_r:>8.3f} {dpo_r:>8.3f} {non_dpo_r:>10.3f} | {all_agree_str:>10} {dpo_agree_str:>10} {non_dpo_agree_str:>10}")


if __name__ == "__main__":
    main()
