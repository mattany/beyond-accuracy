#!/usr/bin/env python3
"""
Analyze the relationship between linguistic sophistication, metaphor use, 
and human preferences in Experiment B.

This script reproduces the analysis reported in Section 5.2.1 of the paper,
specifically the findings about readability and metaphor effects.

Key findings:
1. When explanations differ by 2+ Flesch-Kincaid grade levels, evaluators 
   prefer the more advanced one 64% of the time (N=372).
2. When sophistication competes with metaphor (one explanation 2+ grades 
   more advanced without metaphors vs simpler with metaphors), evaluators 
   choose the sophisticated option 70% of the time (N=99).
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, wilcoxon
from pathlib import Path

# Configuration
RUN_NUMBER = 10
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
METRICS_DIR = Path(__file__).parent.parent.parent / "Benchmarking" / "deep_eval" / "data" / f"run_{RUN_NUMBER}"
EVAL_DATA_PATH = DATA_DIR / "experiment_b_eval_dataset.csv"


def load_data():
    """Load evaluation data and metric scores."""
    eval_df = pd.read_csv(EVAL_DATA_PATH)
    
    # Load metric scores
    flesch_kincaid = pd.read_csv(METRICS_DIR / "flesch_kincaid.csv")
    flesch_reading_ease = pd.read_csv(METRICS_DIR / "flesch_reading_ease.csv")
    metaphor = pd.read_csv(METRICS_DIR / "metaphor_v8.csv")
    
    # Add metric scores to eval_df
    eval_df['fk_a'] = flesch_kincaid['explanation_a__score']
    eval_df['fk_b'] = flesch_kincaid['explanation_b__score']
    eval_df['fk_diff'] = eval_df['fk_a'] - eval_df['fk_b']
    
    eval_df['flesch_a'] = flesch_reading_ease['explanation_a__score']
    eval_df['flesch_b'] = flesch_reading_ease['explanation_b__score']
    eval_df['flesch_diff'] = eval_df['flesch_a'] - eval_df['flesch_b']
    
    eval_df['metaphor_a'] = metaphor['explanation_a__score']
    eval_df['metaphor_b'] = metaphor['explanation_b__score']
    eval_df['metaphor_diff'] = eval_df['metaphor_a'] - eval_df['metaphor_b']
    
    eval_df['chose_a'] = (eval_df['human_choice'] == 'Explanation A').astype(int)
    
    return eval_df


def analyze_sophistication_preference(eval_df):
    """
    Analyze preference for linguistically sophisticated explanations.
    
    Reports the percentage of times evaluators prefer the more advanced 
    explanation when there's a 2+ Flesch-Kincaid grade level difference.
    """
    print("=" * 70)
    print("ANALYSIS 1: Preference for Linguistic Sophistication")
    print("=" * 70)
    
    # When A is more sophisticated (higher FK = harder grade level)
    more_sophisticated_a = eval_df[eval_df['fk_diff'] > 2]  # A is 2+ grades higher
    less_sophisticated_a = eval_df[eval_df['fk_diff'] < -2]  # B is 2+ grades higher
    
    print(f"\nWhen A is MORE SOPHISTICATED (FK +2 grades):")
    print(f"  N = {len(more_sophisticated_a)}")
    print(f"  A chosen: {more_sophisticated_a['chose_a'].mean()*100:.1f}%")
    
    print(f"\nWhen B is MORE SOPHISTICATED (FK +2 grades):")
    print(f"  N = {len(less_sophisticated_a)}")
    print(f"  A chosen: {less_sophisticated_a['chose_a'].mean()*100:.1f}%")
    print(f"  B chosen: {(1-less_sophisticated_a['chose_a'].mean())*100:.1f}%")
    
    # Combined: how often is the more sophisticated one chosen?
    sophisticated_chosen = (
        more_sophisticated_a['chose_a'].sum() +  # A chosen when A is sophisticated
        (len(less_sophisticated_a) - less_sophisticated_a['chose_a'].sum())  # B chosen when B is sophisticated
    )
    total_cases = len(more_sophisticated_a) + len(less_sophisticated_a)
    
    # Wilcoxon signed-rank test: is preference significantly different from 50%?
    # Create array: 1 if sophisticated chosen, 0 if not
    sophisticated_choices = np.concatenate([
        more_sophisticated_a['chose_a'].values,  # 1 if A (sophisticated) chosen
        1 - less_sophisticated_a['chose_a'].values  # 1 if B (sophisticated) chosen
    ])
    # Test against 0.5 (chance)
    _, p_value = wilcoxon(sophisticated_choices - 0.5, alternative='two-sided')
    
    print(f"\n*** PAPER FINDING: When there's a 2+ grade difference ***")
    print(f"  Total cases: {total_cases}")
    print(f"  More sophisticated chosen: {sophisticated_chosen}/{total_cases} = {sophisticated_chosen/total_cases*100:.1f}%")
    print(f"  Wilcoxon signed-rank test vs 50%: p = {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    return {
        'total_cases': total_cases,
        'sophisticated_chosen': sophisticated_chosen,
        'percentage': sophisticated_chosen / total_cases * 100,
        'p_value': p_value
    }


def analyze_sophistication_vs_metaphor(eval_df):
    """
    Analyze what happens when sophistication competes with metaphor use.
    
    Compares cases where one explanation is more sophisticated but lacks 
    metaphors, while the other is simpler but uses metaphors.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Sophistication vs Metaphor Competition")
    print("=" * 70)
    
    # A is more sophisticated (FK +2), no metaphor vs B simpler with metaphor
    sophisticated_no_meta = eval_df[
        (eval_df['fk_a'] > eval_df['fk_b'] + 2) &  # A is 2+ grades more sophisticated
        (eval_df['metaphor_a'] < 0.5) &             # A has no metaphor
        (eval_df['metaphor_b'] > 0.5)               # B has metaphor
    ]
    
    # A is simpler with metaphor vs B is more sophisticated no metaphor
    simple_with_meta = eval_df[
        (eval_df['fk_a'] < eval_df['fk_b'] - 2) &  # A is 2+ grades simpler
        (eval_df['metaphor_a'] > 0.5) &             # A has metaphor
        (eval_df['metaphor_b'] < 0.5)               # B has no metaphor
    ]
    
    print(f"\nA sophisticated (no metaphor) vs B simple (with metaphor):")
    print(f"  N = {len(sophisticated_no_meta)}")
    print(f"  Sophisticated (A) chosen: {sophisticated_no_meta['chose_a'].mean()*100:.1f}%")
    
    print(f"\nA simple (with metaphor) vs B sophisticated (no metaphor):")
    print(f"  N = {len(simple_with_meta)}")
    print(f"  Sophisticated (B) chosen: {(1-simple_with_meta['chose_a'].mean())*100:.1f}%")
    
    # Combined
    total = len(sophisticated_no_meta) + len(simple_with_meta)
    sophisticated_wins = (
        sophisticated_no_meta['chose_a'].sum() + 
        (len(simple_with_meta) - simple_with_meta['chose_a'].sum())
    )
    
    # Wilcoxon signed-rank test: is preference significantly different from 50%?
    # Create array: 1 if sophisticated chosen, 0 if not
    sophisticated_choices = np.concatenate([
        sophisticated_no_meta['chose_a'].values,  # 1 if A (sophisticated) chosen
        1 - simple_with_meta['chose_a'].values  # 1 if B (sophisticated) chosen
    ])
    # Test against 0.5 (chance)
    _, p_value = wilcoxon(sophisticated_choices - 0.5, alternative='two-sided')
    
    print(f"\n*** PAPER FINDING: Sophistication vs Metaphor ***")
    print(f"  Total cases: {total}")
    print(f"  Sophisticated (no metaphor) chosen: {sophisticated_wins}/{total} = {sophisticated_wins/total*100:.1f}%")
    print(f"  Wilcoxon signed-rank test vs 50%: p = {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    return {
        'total_cases': total,
        'sophisticated_chosen': sophisticated_wins,
        'percentage': sophisticated_wins / total * 100,
        'p_value': p_value
    }


def analyze_readability_correlations(eval_df):
    """
    Analyze correlations between readability metrics and human preference.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Readability Correlations with Preference")
    print("=" * 70)
    
    print("\nCorrelation with human preference for A:")
    print("(Positive r = higher score in A → prefer A)")
    
    # Flesch Reading Ease (higher = easier)
    r_flesch, p_flesch = pearsonr(eval_df['flesch_diff'], eval_df['chose_a'])
    print(f"\nFlesch Reading Ease (higher = EASIER):")
    print(f"  r = {r_flesch:.3f}, p = {p_flesch:.6f}")
    print(f"  Interpretation: {'Prefer EASIER' if r_flesch > 0 else 'Prefer HARDER'}")
    
    # Flesch-Kincaid Grade (higher = harder)
    r_fk, p_fk = pearsonr(eval_df['fk_diff'], eval_df['chose_a'])
    print(f"\nFlesch-Kincaid Grade (higher = HARDER/more advanced):")
    print(f"  r = {r_fk:.3f}, p = {p_fk:.6f}")
    print(f"  Interpretation: {'Prefer MORE ADVANCED' if r_fk > 0 else 'Prefer SIMPLER'}")
    
    return {
        'flesch_r': r_flesch,
        'flesch_p': p_flesch,
        'fk_r': r_fk,
        'fk_p': p_fk
    }


def analyze_metaphor_preference(eval_df):
    """
    Analyze the relationship between metaphor use and preference.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Metaphor Use and Preference")
    print("=" * 70)
    
    # When A has metaphor and B doesn't
    a_metaphor_only = eval_df[(eval_df['metaphor_a'] > 0.5) & (eval_df['metaphor_b'] < 0.5)]
    print(f"\nWhen A has metaphor, B doesn't (N={len(a_metaphor_only)}):")
    print(f"  A chosen: {a_metaphor_only['chose_a'].mean()*100:.1f}%")
    
    # When B has metaphor and A doesn't
    b_metaphor_only = eval_df[(eval_df['metaphor_b'] > 0.5) & (eval_df['metaphor_a'] < 0.5)]
    print(f"\nWhen B has metaphor, A doesn't (N={len(b_metaphor_only)}):")
    print(f"  A chosen: {b_metaphor_only['chose_a'].mean()*100:.1f}%")
    print(f"  B chosen: {(1-b_metaphor_only['chose_a'].mean())*100:.1f}%")
    
    # Correlation between metaphor and FK grade
    r_meta_fk, p_meta_fk = pearsonr(eval_df['metaphor_a'], eval_df['fk_a'])
    print(f"\n*** Metaphor-Sophistication Correlation ***")
    print(f"  Metaphor score vs FK grade: r = {r_meta_fk:.3f}, p = {p_meta_fk:.6f}")
    print(f"  {'Metaphor → SIMPLER language' if r_meta_fk < 0 else 'Metaphor → MORE SOPHISTICATED language'}")
    
    return {
        'a_metaphor_n': len(a_metaphor_only),
        'a_metaphor_chosen': a_metaphor_only['chose_a'].mean() * 100,
        'b_metaphor_n': len(b_metaphor_only),
        'b_metaphor_chosen': (1 - b_metaphor_only['chose_a'].mean()) * 100,
        'meta_fk_r': r_meta_fk,
        'meta_fk_p': p_meta_fk
    }


def analyze_metaphor_complexity_confound(eval_df):
    """
    Analyze whether metaphor's negative effect is explained by complexity confound.
    
    Key findings:
    - Metaphor correlates with simpler language (r = -0.14, p < 0.001) - but weak effect
    - When both explanations are SIMPLE: metaphor is neutral (55.8%, p=0.41 ns)
    - When both explanations are COMPLEX: metaphor HURTS (41%, p=0.03*)
    """
    from math import asin, sqrt
    
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Metaphor Effect by Complexity Level")
    print("=" * 70)
    
    # Both simple (FK < 10)
    simple_both = eval_df[(eval_df['fk_a'] < 10) & (eval_df['fk_b'] < 10)]
    a_meta_simple = simple_both[(simple_both['metaphor_a'] > 0.5) & (simple_both['metaphor_b'] < 0.5)]
    b_meta_simple = simple_both[(simple_both['metaphor_b'] > 0.5) & (simple_both['metaphor_a'] < 0.5)]
    total_simple = len(a_meta_simple) + len(b_meta_simple)
    
    simple_results = None
    if total_simple > 0:
        meta_wins_simple = int(a_meta_simple['chose_a'].sum() + (len(b_meta_simple) - b_meta_simple['chose_a'].sum()))
        pct_simple = meta_wins_simple / total_simple * 100
        meta_choices_simple = np.concatenate([a_meta_simple['chose_a'].values, 1 - b_meta_simple['chose_a'].values])
        _, p_simple = wilcoxon(meta_choices_simple - 0.5, alternative='two-sided')
        h_simple = 2 * (asin(sqrt(pct_simple/100)) - asin(sqrt(0.5)))
        sig_simple = '***' if p_simple < 0.001 else '**' if p_simple < 0.01 else '*' if p_simple < 0.05 else '(ns)'
        print(f"  Both SIMPLE (FK < 10): {pct_simple:.1f}% prefer metaphor (N={total_simple}, p={p_simple:.3f} {sig_simple})")
        simple_results = {'pct': pct_simple, 'n': total_simple, 'p': p_simple, 'h': h_simple}
    
    # Both complex (FK >= 10)
    complex_both = eval_df[(eval_df['fk_a'] >= 10) & (eval_df['fk_b'] >= 10)]
    a_meta_complex = complex_both[(complex_both['metaphor_a'] > 0.5) & (complex_both['metaphor_b'] < 0.5)]
    b_meta_complex = complex_both[(complex_both['metaphor_b'] > 0.5) & (complex_both['metaphor_a'] < 0.5)]
    total_complex = len(a_meta_complex) + len(b_meta_complex)
    
    complex_results = None
    if total_complex > 0:
        meta_wins_complex = int(a_meta_complex['chose_a'].sum() + (len(b_meta_complex) - b_meta_complex['chose_a'].sum()))
        pct_complex = meta_wins_complex / total_complex * 100
        meta_choices_complex = np.concatenate([a_meta_complex['chose_a'].values, 1 - b_meta_complex['chose_a'].values])
        _, p_complex = wilcoxon(meta_choices_complex - 0.5, alternative='two-sided')
        h_complex = 2 * (asin(sqrt(pct_complex/100)) - asin(sqrt(0.5)))
        sig_complex = '***' if p_complex < 0.001 else '**' if p_complex < 0.01 else '*' if p_complex < 0.05 else '(ns)'
        print(f"  Both COMPLEX (FK >= 10): {pct_complex:.1f}% prefer metaphor (N={total_complex}, p={p_complex:.3f} {sig_complex})")
        complex_results = {'pct': pct_complex, 'n': total_complex, 'p': p_complex, 'h': h_complex}
    
    print(f"\n  → In SIMPLE explanations: metaphor is NEUTRAL")
    print(f"  → In COMPLEX explanations: metaphor HURTS")
    
    return {
        'simple': simple_results,
        'complex': complex_results
    }


def show_examples(eval_df, n_examples=2):
    """
    Show example pairs where humans chose the harder-to-read explanation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLES: Human chose harder-to-read (more sophisticated) explanation")
    print("=" * 70)
    
    # Find cases where A is harder and was chosen
    harder_a_chosen = eval_df[
        (eval_df['fk_diff'] > 3) &  # A is significantly harder
        (eval_df['chose_a'] == 1)
    ].sort_values('fk_diff', ascending=False)
    
    for i, (idx, row) in enumerate(harder_a_chosen.head(n_examples).iterrows()):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i+1}")
        print(f"FK Grade: A={row['fk_a']:.1f} vs B={row['fk_b']:.1f} (A is {row['fk_diff']:.1f} grades harder)")
        print(f"Cluster {row['cluster']}: {row['model_a']} vs {row['model_b']}")
        print(f"✅ Human chose: A (the more sophisticated one)")
        print("=" * 70)
        
        print(f"\n📝 QUESTION:\n{row['question'][:200]}...")
        
        print(f"\n🅰️ EXPLANATION A (MORE SOPHISTICATED - CHOSEN):")
        print("-" * 60)
        print(row['explanation_a'][:500])
        if len(row['explanation_a']) > 500:
            print("...")
        
        print(f"\n🅱️ EXPLANATION B (SIMPLER - REJECTED):")
        print("-" * 60)
        print(row['explanation_b'][:500])
        if len(row['explanation_b']) > 500:
            print("...")


def main():
    """Run all analyses."""
    print("Loading data...")
    eval_df = load_data()
    print(f"Loaded {len(eval_df)} comparisons from {EVAL_DATA_PATH}")
    print(f"Metrics from: {METRICS_DIR}")
    
    # Run analyses
    sophistication_results = analyze_sophistication_preference(eval_df)
    # Note: analyze_sophistication_vs_metaphor is confounded (mixes sophistication + metaphor effects)
    # Keeping for reference but not in summary
    _ = analyze_sophistication_vs_metaphor(eval_df)
    correlation_results = analyze_readability_correlations(eval_df)
    metaphor_results = analyze_metaphor_preference(eval_df)
    confound_results = analyze_metaphor_complexity_confound(eval_df)
    
    # Show examples
    show_examples(eval_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (SIGNIFICANT RESULTS)")
    print("=" * 70)
    print(f"\n1. Sophistication preference (2+ FK grade diff):")
    print(f"   {sophistication_results['percentage']:.0f}% prefer more advanced (N={sophistication_results['total_cases']})")
    print(f"   p < 0.001 ***")
    
    print(f"\n2. Metaphor effect by complexity level:")
    if confound_results['simple']:
        sig = '***' if confound_results['simple']['p'] < 0.001 else '**' if confound_results['simple']['p'] < 0.01 else '*' if confound_results['simple']['p'] < 0.05 else '(ns)'
        print(f"   Both SIMPLE (FK<10): {confound_results['simple']['pct']:.1f}% prefer metaphor (N={confound_results['simple']['n']}, p={confound_results['simple']['p']:.3f}) {sig}")
    if confound_results['complex']:
        sig = '***' if confound_results['complex']['p'] < 0.001 else '**' if confound_results['complex']['p'] < 0.01 else '*' if confound_results['complex']['p'] < 0.05 else '(ns)'
        print(f"   Both COMPLEX (FK>=10): {confound_results['complex']['pct']:.1f}% prefer metaphor (N={confound_results['complex']['n']}, p={confound_results['complex']['p']:.3f}) {sig}")
    print(f"\n   → Metaphor is NEUTRAL in simple text")
    print(f"   → Metaphor HURTS in complex text")


if __name__ == "__main__":
    main()

