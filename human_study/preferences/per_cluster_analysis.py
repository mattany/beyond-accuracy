#!/usr/bin/env python3
"""Per-cluster correlation analysis."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Cluster mapping
CLUSTER_INFO = {
    0: "SciComma-3.1-8B_y vs gpt-3.5-turbo-0125_cot",
    1: "SciComma-3.1-8B_prompt vs scicomma-3.1-dpo_prompt (SYNTH DPO)",
    2: "gpt-3.5-turbo-0125 vs scicomma-3.1-dpo (SYNTH DPO)",
    3: "gpt-3.5-turbo-0125_cot vs scicomma-3.1-dpo_prompt (SYNTH DPO)",
    4: "SciComma-3.1-8B_prompt vs gpt-3.5-turbo-0125_cot",
    5: "SciComma-3.1-8B_prompt vs organic_SFT_prompted",
    6: "SciComma-3.1-8B_y vs vanilla_prompted",
    7: "gpt-3.5-turbo-0125_cot vs human_answers",
}

SYNTH_DPO_CLUSTERS = {1, 2, 3}  # Clusters with synthetic DPO


def main():
    df = pd.read_csv(DATA_DIR / "correlation_results.csv")
    
    # Key metrics
    metrics = ['analogy_v2_diff', 'metaphor_v8_diff', 'scaffolding_v2_diff', 'humor_v5_diff']
    
    print("=" * 80)
    print("PER-CLUSTER CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Overall stats per cluster
    print("\n### CLUSTER SUMMARY ###\n")
    print(f"{'Cluster':<6} {'N':<5} {'Human A%':>10} {'Models':<50}")
    print("-" * 80)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id].dropna(subset=['human_pref'])
        n = len(cluster_df)
        human_a_pct = cluster_df['human_pref'].mean() * 100
        marker = " **" if cluster_id in SYNTH_DPO_CLUSTERS else ""
        print(f"{cluster_id:<6} {n:<5} {human_a_pct:>9.1f}% {CLUSTER_INFO.get(cluster_id, 'Unknown'):<50}{marker}")
    
    # Per-cluster correlations for key metrics
    print("\n\n### ANALOGY CORRELATION BY CLUSTER ###\n")
    print(f"{'Cluster':<6} {'N':<5} {'r':>8} {'p':>10} {'Agree%':>8} {'Interpretation':<30}")
    print("-" * 80)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id].dropna(subset=['human_pref', 'analogy_v2_diff'])
        if len(cluster_df) < 5:
            continue
        
        r, p = stats.pointbiserialr(cluster_df['human_pref'], cluster_df['analogy_v2_diff'])
        pred = (cluster_df['analogy_v2_diff'] > 0).astype(int)
        agree = (pred == cluster_df['human_pref']).mean() * 100
        
        marker = "SYNTH DPO" if cluster_id in SYNTH_DPO_CLUSTERS else ""
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{cluster_id:<6} {len(cluster_df):<5} {r:>7.3f}{sig} {p:>10.4f} {agree:>7.1f}% {marker:<30}")
    
    # Same for metaphor
    print("\n\n### METAPHOR CORRELATION BY CLUSTER ###\n")
    print(f"{'Cluster':<6} {'N':<5} {'r':>8} {'p':>10} {'Agree%':>8} {'Interpretation':<30}")
    print("-" * 80)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id].dropna(subset=['human_pref', 'metaphor_v8_diff'])
        if len(cluster_df) < 5:
            continue
        
        r, p = stats.pointbiserialr(cluster_df['human_pref'], cluster_df['metaphor_v8_diff'])
        pred = (cluster_df['metaphor_v8_diff'] > 0).astype(int)
        agree = (pred == cluster_df['human_pref']).mean() * 100
        
        marker = "SYNTH DPO" if cluster_id in SYNTH_DPO_CLUSTERS else ""
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{cluster_id:<6} {len(cluster_df):<5} {r:>7.3f}{sig} {p:>10.4f} {agree:>7.1f}% {marker:<30}")
    
    # Compare DPO vs non-DPO clusters
    print("\n\n### AGGREGATED: SYNTH DPO vs NON-DPO CLUSTERS ###\n")
    
    dpo_df = df[df['cluster'].isin(SYNTH_DPO_CLUSTERS)].dropna(subset=['human_pref', 'analogy_v2_diff', 'metaphor_v8_diff'])
    non_dpo_df = df[~df['cluster'].isin(SYNTH_DPO_CLUSTERS)].dropna(subset=['human_pref', 'analogy_v2_diff', 'metaphor_v8_diff'])
    
    print(f"{'Group':<20} {'N':<6} {'Analogy r':>10} {'Metaphor r':>12} {'Human A%':>10}")
    print("-" * 60)
    
    if len(dpo_df) >= 5:
        r_ana, _ = stats.pointbiserialr(dpo_df['human_pref'], dpo_df['analogy_v2_diff'])
        r_met, _ = stats.pointbiserialr(dpo_df['human_pref'], dpo_df['metaphor_v8_diff'])
        human_a = dpo_df['human_pref'].mean() * 100
        print(f"{'Synth DPO clusters':<20} {len(dpo_df):<6} {r_ana:>10.3f} {r_met:>12.3f} {human_a:>9.1f}%")
    
    if len(non_dpo_df) >= 5:
        r_ana, _ = stats.pointbiserialr(non_dpo_df['human_pref'], non_dpo_df['analogy_v2_diff'])
        r_met, _ = stats.pointbiserialr(non_dpo_df['human_pref'], non_dpo_df['metaphor_v8_diff'])
        human_a = non_dpo_df['human_pref'].mean() * 100
        print(f"{'Non-DPO clusters':<20} {len(non_dpo_df):<6} {r_ana:>10.3f} {r_met:>12.3f} {human_a:>9.1f}%")
    
    # Mean metric scores comparison
    print("\n\n### MEAN METRIC SCORES: DPO vs Non-DPO ###\n")
    print("(Higher = model A scores higher on that metric)")
    
    for metric in ['analogy_v2_diff', 'metaphor_v8_diff', 'scaffolding_v2_diff']:
        dpo_mean = dpo_df[metric].mean() if metric in dpo_df else np.nan
        non_dpo_mean = non_dpo_df[metric].mean() if metric in non_dpo_df else np.nan
        print(f"  {metric.replace('_diff', ''):<20} DPO: {dpo_mean:>7.3f}   Non-DPO: {non_dpo_mean:>7.3f}")


if __name__ == "__main__":
    main()

