#!/usr/bin/env python3
"""
Wilcoxon signed-rank tests for model comparisons in Experiment A.
Computes p-values for paired comparisons on per-question aggregate scores.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
METRICS = ['jargon', 'flesch_reading_ease', 'flesch_kincaid', 'ari', 'dale_chall', 
           'scaffolding_v2', 'analogy_v2', 'metaphor_v8', 'humor_v5']

# Normalization ranges (from aggregate_v2.py)
NORMALIZATION_RANGES = {
    "jargon": (0.65, 1.0),
    "flesch_kincaid": (6, 16),
    "ari": (6, 16),
    "dale_chall": (7, 12),
    "flesch_reading_ease": (0.3, 0.7),
}
LOWER_IS_BETTER = ["ari", "dale_chall", "flesch_kincaid"]

# Model name mapping (CSV column names -> paper names)
MODEL_MAP = {
    'human': 'Human',
    'gpt-3.5-turbo-0125': 'GPT-3.5 (unprompted)',
    'gpt-3.5-turbo-0125_cot': 'GPT-3.5 (Teacher) prompted',
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit': 'Base-Llama (unprompted)',
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt': 'Base-Llama (prompted)',
    'SciComma-3.1-8B_y': 'SFT (unprompted)',
    'SciComma-3.1-8B_prompt': 'SFT (prompted)',
    'scicomma-3.1-dpo': 'SFT+Synth-DPO (unprompted)',
    'scicomma-3.1-dpo_prompt': 'SFT+Synth-DPO (prompted)',
    'organic_sft': 'Human-SFT (unprompted)',
    'organic_sft_prompt': 'Human-SFT (prompted)',
    'organic_dpo': 'SFT+Human-DPO (unprompted)',
    'organic_dpo_prompt': 'SFT+Human-DPO (prompted)',
}


def normalize_score(value: float, metric: str) -> float:
    """Normalize a score to [0, 1] range."""
    if pd.isna(value):
        return np.nan
    
    if metric in NORMALIZATION_RANGES:
        min_val, max_val = NORMALIZATION_RANGES[metric]
    else:
        min_val, max_val = 0, 1
    
    if min_val == max_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    
    if metric in LOWER_IS_BETTER:
        normalized = 1 - normalized
    
    return normalized


def load_per_question_scores():
    """Load all metrics and compute per-question aggregate scores for each model."""
    
    # Get list of models from first metric file
    first_metric = pd.read_csv(DATA_DIR / f"{METRICS[0]}.csv")
    score_cols = [c for c in first_metric.columns if c.endswith('__score')]
    models = [c.replace('__score', '') for c in score_cols]
    
    n_questions = len(first_metric)
    print(f"Found {len(models)} models, {n_questions} questions")
    
    # Initialize storage for per-question scores
    model_scores = {model: np.zeros(n_questions) for model in models}
    model_counts = {model: np.zeros(n_questions) for model in models}
    
    # Load each metric and accumulate normalized scores
    for metric in METRICS:
        metric_path = DATA_DIR / f"{metric}.csv"
        if not metric_path.exists():
            print(f"  Warning: {metric}.csv not found, skipping")
            continue
        
        df = pd.read_csv(metric_path)
        print(f"  Loaded {metric}: {len(df)} rows")
        
        for model in models:
            score_col = f"{model}__score"
            if score_col in df.columns:
                for i, val in enumerate(df[score_col]):
                    if pd.notna(val):
                        norm_val = normalize_score(val, metric)
                        model_scores[model][i] += norm_val
                        model_counts[model][i] += 1
    
    # Compute average scores per question
    result = {}
    for model in models:
        # Only include questions where we have at least some metrics
        scores = model_scores[model] / np.maximum(model_counts[model], 1)
        # Set to NaN where we have no data
        scores[model_counts[model] == 0] = np.nan
        result[model] = scores
    
    return result


def run_wilcoxon_test(scores_a, scores_b, name_a, name_b):
    """Run Wilcoxon signed-rank test and return results."""
    # Remove pairs where either is NaN
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    a = scores_a[mask]
    b = scores_b[mask]
    
    if len(a) < 10:
        return {'n': len(a), 'mean_a': np.nan, 'mean_b': np.nan, 
                'diff': np.nan, 'statistic': np.nan, 'p_value': np.nan}
    
    # Wilcoxon signed-rank test (two-sided)
    try:
        stat, p = wilcoxon(a, b, alternative='two-sided')
    except ValueError as e:
        print(f"  Warning: {e}")
        stat, p = np.nan, np.nan
    
    return {
        'model_a': name_a,
        'model_b': name_b,
        'n': len(a),
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
        'diff': np.mean(a) - np.mean(b),
        'statistic': stat,
        'p_value': p,
    }


def main():
    print("Loading per-question scores...")
    scores = load_per_question_scores()
    
    # Define comparisons from the paper
    comparisons = [
        # Finding 1: Human vs GPT unprompted
        ('human', 'gpt-3.5-turbo-0125'),
        
        # Finding 2: Synthetic vs Organic
        ('SciComma-3.1-8B_prompt', 'organic_sft_prompt'),  # SFT synthetic vs organic (prompted)
        ('scicomma-3.1-dpo_prompt', 'organic_dpo_prompt'),  # DPO synthetic vs organic (prompted)
        
        # Finding 3: SFT prompted vs Base-Llama prompted
        ('SciComma-3.1-8B_prompt', 'Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt'),
        
        # Finding 3b: SFT prompted vs GPT prompted (teacher)
        ('SciComma-3.1-8B_prompt', 'gpt-3.5-turbo-0125_cot'),
        
        # Finding 4: SFT unprompted vs Base-Llama unprompted
        ('SciComma-3.1-8B_y', 'Meta-Llama-3.1-8B-Instruct-bnb-4bit'),
        
        # Finding 5: SFT+Synth-DPO unprompted vs GPT prompted (internalization)
        ('scicomma-3.1-dpo', 'gpt-3.5-turbo-0125_cot'),
    ]
    
    print("\n" + "=" * 80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("=" * 80)
    
    results = []
    for model_a, model_b in comparisons:
        if model_a not in scores or model_b not in scores:
            print(f"  Warning: Model not found: {model_a} or {model_b}")
            continue
        
        result = run_wilcoxon_test(scores[model_a], scores[model_b], model_a, model_b)
        results.append(result)
        
        name_a = MODEL_MAP.get(model_a, model_a)
        name_b = MODEL_MAP.get(model_b, model_b)
        
        # Format p-value
        p = result['p_value']
        if pd.isna(p):
            p_str = "N/A"
        elif p < 0.001:
            p_str = "p < 0.001"
        elif p < 0.01:
            p_str = f"p < 0.01 (p={p:.4f})"
        elif p < 0.05:
            p_str = f"p < 0.05 (p={p:.4f})"
        else:
            p_str = f"p = {p:.4f} (n.s.)"
        
        print(f"\n{name_a} vs {name_b}")
        print(f"  N = {result['n']} paired observations")
        print(f"  Mean A: {result['mean_a']*100:.1f}%  |  Mean B: {result['mean_b']*100:.1f}%")
        print(f"  Difference: {result['diff']*100:+.1f}%")
        print(f"  {p_str}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = DATA_DIR / "wilcoxon_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

