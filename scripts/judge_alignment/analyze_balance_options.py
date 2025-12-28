#!/usr/bin/env python3
"""
Analyze and generate balanced survey datasets from ask_science_human_metrics.csv.

This script:
1. Analyzes existing metrics data to find the best achievable balance
2. Can save balanced datasets to CSV files
3. Can generate more metric data if needed (with --generate flag)

Usage:
    python analyze_balance_options.py              # Analyze only
    python analyze_balance_options.py --save       # Analyze and save best datasets
    python analyze_balance_options.py --generate   # Generate more data if needed
    
    # Run specific metrics with smart stratified sampling:
    python analyze_balance_options.py --generate --metrics metaphor_v3 --smart-sample --batch-size 20
    
    # Smart sampling: For a metric like metaphor_v3, samples half from rows where
    # metaphor_v2_score > 0.5 (positive) and half from rows where metaphor_v2_score <= 0.5 (negative).
    # Falls back to source CSV if not enough examples in one group.
"""
import os
import sys
import argparse
import asyncio
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Set random seed for deterministic output
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Constants
THRESHOLD = 0.5
LARGE_SURVEY_SIZE = 30
SMALL_SURVEY_SIZE = 10
DEFAULT_BATCH_SIZE = 100
GEVAL_RETRIES = 3
MAX_ANSWER_LENGTH = 2560  # Exclude answers longer than this for labeling tasks

# Mapping from metric name to previous versions (for smart sampling)
# List is ordered by preference: try first, then fallback to next
METRIC_PREVIOUS_VERSIONS = {
    'metaphor_v12_1': ['metaphor_v12', 'metaphor_v11', 'metaphor_v10', 'metaphor_v9', 'metaphor_v8', 'metaphor_v7', 'metaphor_v6', 'metaphor_v5', 'metaphor_v4', 'metaphor_v3', 'metaphor_v2'],
    'metaphor_v12': ['metaphor_v11', 'metaphor_v10', 'metaphor_v9', 'metaphor_v8', 'metaphor_v7', 'metaphor_v6', 'metaphor_v5', 'metaphor_v4', 'metaphor_v3', 'metaphor_v2'],
    'metaphor_v8': ['metaphor_v7', 'metaphor_v6', 'metaphor_v5', 'metaphor_v4', 'metaphor_v3', 'metaphor_v2'],
    'metaphor_v6': ['metaphor_v5', 'metaphor_v4', 'metaphor_v3', 'metaphor_v2'],
    'metaphor_v5': ['metaphor_v4', 'metaphor_v3', 'metaphor_v2'],
    'metaphor_v4': ['metaphor_v3', 'metaphor_v2'],
    'metaphor_v3': ['metaphor_v2'],
    'humor_v5': ['humor_v4', 'humor_v3', 'humor_v2'],
    'humor_v4': ['humor_v3', 'humor_v2'],
    'humor_v3': ['humor_v2'],
    'analogy_v4': ['analogy_v3', 'analogy_v2'],
    'analogy_v3': ['analogy_v2'],
    'connection_to_everyday_life_v5': ['connection_to_everyday_life_v4', 'connection_to_everyday_life_v3', 'connection_to_everyday_life_v2'],
    'connection_to_everyday_life_v4': ['connection_to_everyday_life_v3', 'connection_to_everyday_life_v2'],
    'connection_to_everyday_life_v3': ['connection_to_everyday_life_v2'],
}

ALL_METRICS = [
    'humor_v2_score',
    'metaphor_v8_score',
    'analogy_v2_score',
    'connection_to_everyday_life_v2_score',
    'scaffolding_score'
]

# Reason columns corresponding to each metric
ALL_REASON_COLUMNS = [
    'humor_v2_reason',
    'metaphor_v8_reason',
    'analogy_v2_reason',
    'connection_to_everyday_life_v2_reason',
    'scaffolding_reason'
]

# Column rename mapping for output (to match balanced_dataset.csv format)
OUTPUT_COLUMN_RENAME = {
    'Question': 'question',
    'Human Answer': 'answer'
}

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "balanced_dataset_v2_human"
METRICS_PATH = OUTPUT_DIR / "ask_science_human_metrics.csv"
SOURCE_PATH = SCRIPT_DIR.parent.parent / "SFT" / "data" / "ask_science_human.csv"
EXCLUDE_INDICES_PATH = SCRIPT_DIR / "exclude_indices.csv"


def load_exclude_indices():
    """Load indices to exclude from CSV file."""
    if EXCLUDE_INDICES_PATH.exists():
        exclude_df = pd.read_csv(EXCLUDE_INDICES_PATH)
        return exclude_df['index'].tolist()
    return []


def add_to_exclude_indices(new_indices):
    """Add new indices to the exclusion list.
    
    Args:
        new_indices: List of indices to add to exclusion list
    
    Returns:
        Number of new indices actually added (excludes duplicates)
    """
    existing = set(load_exclude_indices())
    new_to_add = set(new_indices) - existing
    
    if not new_to_add:
        return 0
    
    # Combine and save
    all_indices = sorted(existing | new_to_add)
    exclude_df = pd.DataFrame({'index': all_indices})
    exclude_df.to_csv(EXCLUDE_INDICES_PATH, index=False)
    
    return len(new_to_add)


def short_name(m):
    return m.replace('_score', '').replace('_v2', '').replace('connection_to_everyday_life', 'conn')


# =============================================================================
# Balance Finding Algorithms
# =============================================================================

def try_perfect_balance(df_values, valid_indices, target_size, metrics_list, max_seeds=1000):
    """
    Try to form a perfectly balanced dataset for ALL metrics.
    Uses a smarter approach: prioritize samples with more positives first.
    """
    half_size = target_size // 2
    
    def count_positives(idx):
        return sum(1 for m in metrics_list if df_values[idx][m] > THRESHOLD)
    
    for seed in range(max_seeds):
        random.seed(seed)
        
        indices_with_counts = [(idx, count_positives(idx)) for idx in valid_indices]
        random.shuffle(indices_with_counts)
        indices_with_counts.sort(key=lambda x: -x[1])
        sorted_indices = [idx for idx, _ in indices_with_counts]
        
        pos_count = {m: 0 for m in metrics_list}
        neg_count = {m: 0 for m in metrics_list}
        selected = []
        
        for idx in sorted_indices:
            if len(selected) >= target_size:
                break
            
            can_add = True
            for m in metrics_list:
                is_pos = df_values[idx][m] > THRESHOLD
                if is_pos and pos_count[m] >= half_size:
                    can_add = False
                    break
                if not is_pos and neg_count[m] >= half_size:
                    can_add = False
                    break
            
            if can_add:
                selected.append(idx)
                for m in metrics_list:
                    if df_values[idx][m] > THRESHOLD:
                        pos_count[m] += 1
                    else:
                        neg_count[m] += 1
        
        if len(selected) == target_size:
            if all(pos_count[m] == half_size for m in metrics_list):
                return selected, pos_count
    
    return None, None


def try_relaxed_balance(df_values, valid_indices, target_size, metrics_list, tolerance=0.1, max_seeds=1000):
    """
    Try to form a dataset with relaxed balance tolerance.
    Finds the BEST balance across all seeds (closest to 50/50).
    """
    min_pos = int(target_size * (0.5 - tolerance))
    max_pos = int(target_size * (0.5 + tolerance))
    max_neg = target_size - min_pos
    half_size = target_size // 2
    
    def count_positives(idx):
        return sum(1 for m in metrics_list if df_values[idx][m] > THRESHOLD)
    
    def balance_score(pos_count):
        return sum(abs(pos_count[m] - half_size) for m in metrics_list)
    
    best_selected = None
    best_pos_count = None
    best_score = float('inf')
    
    for seed in range(max_seeds):
        random.seed(seed)
        
        indices_with_counts = [(idx, count_positives(idx)) for idx in valid_indices]
        random.shuffle(indices_with_counts)
        indices_with_counts.sort(key=lambda x: -x[1])
        sorted_indices = [idx for idx, _ in indices_with_counts]
        
        pos_count = {m: 0 for m in metrics_list}
        neg_count = {m: 0 for m in metrics_list}
        selected = []
        
        for idx in sorted_indices:
            if len(selected) >= target_size:
                break
            
            can_add = True
            for m in metrics_list:
                is_pos = df_values[idx][m] > THRESHOLD
                if is_pos and pos_count[m] >= max_pos:
                    can_add = False
                    break
                if not is_pos and neg_count[m] >= max_neg:
                    can_add = False
                    break
            
            if can_add:
                selected.append(idx)
                for m in metrics_list:
                    if df_values[idx][m] > THRESHOLD:
                        pos_count[m] += 1
                    else:
                        neg_count[m] += 1
        
        if len(selected) == target_size:
            if all(min_pos <= pos_count[m] <= max_pos for m in metrics_list):
                score = balance_score(pos_count)
                if score < best_score:
                    best_score = score
                    best_selected = selected.copy()
                    best_pos_count = pos_count.copy()
                    if score == 0:
                        return best_selected, best_pos_count
    
    return best_selected, best_pos_count


def find_best_balance(df_values, valid_indices, target_size, metrics_list, max_seeds=2000):
    """
    Find the best achievable balance by trying progressively looser tolerances.
    Starts tight (perfect 50/50) and relaxes until a solution is found.
    """
    half_size = target_size // 2
    
    def balance_score(pos_count):
        return sum(abs(pos_count[m] - half_size) for m in metrics_list)
    
    # Try progressively looser tolerances, starting with perfect balance
    tolerances = [0.0, 0.03, 0.05, 0.07, 0.1, 0.13, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    for tolerance in tolerances:
        if tolerance == 0.0:
            # Try perfect balance first
            result, counts = try_perfect_balance(
                df_values, valid_indices, target_size, metrics_list, max_seeds=max_seeds
            )
        else:
            result, counts = try_relaxed_balance(
                df_values, valid_indices, target_size, metrics_list, 
                tolerance=tolerance, max_seeds=max_seeds
            )
        
        if result:
            worst_deviation = max(abs(counts[m] / target_size - 0.5) for m in metrics_list)
            print(f"    Found with tolerance={tolerance} ({int((0.5-tolerance)*100)}-{int((0.5+tolerance)*100)}%)")
            return result, counts, worst_deviation
    
    return None, None, None


# =============================================================================
# Survey Analysis Functions
# =============================================================================

def analyze_surveys(df, survey_configs, use_relaxed=False, tolerance=0.1, label="surveys"):
    """Try to form non-overlapping surveys with given configurations."""
    balance_type = f"relaxed ({int((0.5-tolerance)*100)}-{int((0.5+tolerance)*100)}%)" if use_relaxed else "perfect (50/50)"
    print(f"\n=== Trying to form {label} with {balance_type} balance ===\n")
    
    valid_df = df.dropna()
    all_indices = set(valid_df.index.tolist())
    df_values = {idx: {m: valid_df.at[idx, m] for m in ALL_METRICS} for idx in all_indices}
    
    balance_fn = try_relaxed_balance if use_relaxed else try_perfect_balance
    
    surveys = []
    remaining_indices = list(all_indices)
    
    for name, size in tqdm(survey_configs, desc="Forming surveys"):
        if use_relaxed:
            result, counts = balance_fn(df_values, remaining_indices, size, ALL_METRICS, tolerance)
        else:
            result, counts = balance_fn(df_values, remaining_indices, size, ALL_METRICS)
        
        if result:
            surveys.append((name, size, result, counts))
            remaining_indices = [idx for idx in remaining_indices if idx not in result]
            print(f"  ✓ {name}: Found balanced set")
        else:
            print(f"  ✗ {name}: FAILED - not enough balanced data")
            return None
    
    return surveys


def analyze_four_surveys(df, use_relaxed=False, tolerance=0.1):
    survey_configs = [
        (f"Survey 1 ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE),
        (f"Survey 2 ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE),
        (f"Survey 3 ({SMALL_SURVEY_SIZE} questions)", SMALL_SURVEY_SIZE),
        (f"Survey 4 ({SMALL_SURVEY_SIZE} questions)", SMALL_SURVEY_SIZE),
    ]
    return analyze_surveys(df, survey_configs, use_relaxed, tolerance, "4 surveys")


def analyze_two_surveys(df, use_relaxed=False, tolerance=0.1):
    survey_configs = [
        (f"Large Survey ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE),
        (f"Small Survey ({SMALL_SURVEY_SIZE} questions)", SMALL_SURVEY_SIZE),
    ]
    return analyze_surveys(df, survey_configs, use_relaxed, tolerance, "2 surveys")


def analyze_best_achievable(df):
    """Find the best achievable balance for the 2-survey plan."""
    print("\n=== Finding BEST achievable balance (no tolerance constraint) ===\n")
    
    valid_df = df.dropna()
    all_indices = set(valid_df.index.tolist())
    df_values = {idx: {m: valid_df.at[idx, m] for m in ALL_METRICS} for idx in all_indices}
    
    remaining_indices = list(all_indices)
    surveys = []
    
    print(f"Finding best balance for {LARGE_SURVEY_SIZE}-question survey...")
    result, counts, tol = find_best_balance(df_values, remaining_indices, LARGE_SURVEY_SIZE, ALL_METRICS)
    if result:
        surveys.append((f"Large Survey ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE, result, counts))
        remaining_indices = [idx for idx in remaining_indices if idx not in result]
        print(f"  ✓ Found! Worst metric deviation: {tol*100:.1f}% from 50%")
        
        print(f"Finding best balance for {SMALL_SURVEY_SIZE}-question survey...")
        result2, counts2, tol2 = find_best_balance(df_values, remaining_indices, SMALL_SURVEY_SIZE, ALL_METRICS)
        if result2:
            surveys.append((f"Small Survey ({SMALL_SURVEY_SIZE} questions)", SMALL_SURVEY_SIZE, result2, counts2))
            print(f"  ✓ Found! Worst metric deviation: {tol2*100:.1f}% from 50%")
            return surveys
    
    return None


def analyze_large_only(df, metrics_list=None):
    """Find the best achievable balance for a single large survey only.
    
    Args:
        df: DataFrame with metrics data
        metrics_list: List of metric column names to balance on. If None, uses ALL_METRICS.
    
    Returns:
        Tuple of (surveys, score_cols, achieved_tolerance) or (None, score_cols, None) if not found.
    """
    if metrics_list is None:
        metrics_list = ALL_METRICS
    
    # Convert metric names to score column names if needed
    score_cols = [m if m.endswith('_score') else f"{m}_score" for m in metrics_list]
    
    print("\n=== Finding BEST achievable balance for LARGE SURVEY ONLY ===\n")
    print(f"Balancing on metrics: {[short_name(m) for m in score_cols]}")
    
    # Check if all required columns exist
    missing_cols = [col for col in score_cols if col not in df.columns]
    if missing_cols:
        print(f"  ✗ Missing columns in dataset: {missing_cols}")
        print(f"    Run --generate to create data for these metrics first.")
        return None, score_cols, None
    
    # Only keep rows that have values for ALL selected metrics
    valid_df = df.dropna(subset=score_cols)
    print(f"Rows with all selected metrics evaluated: {len(valid_df)}")
    
    all_indices = set(valid_df.index.tolist())
    
    # Exclude indices already used in previous surveys
    exclude_indices = load_exclude_indices()
    if exclude_indices:
        excluded = set(exclude_indices) & all_indices
        all_indices = all_indices - excluded
        print(f"Excluding {len(excluded)} indices from previous survey: {sorted(excluded)}")
    
    if len(all_indices) < LARGE_SURVEY_SIZE:
        print(f"  ✗ Not enough samples! Have {len(all_indices)}, need {LARGE_SURVEY_SIZE}")
        return None, score_cols, None
    
    df_values = {idx: {m: valid_df.at[idx, m] for m in score_cols} for idx in all_indices}
    
    # Debug: show positive/negative distribution per metric after exclusions
    half_size = LARGE_SURVEY_SIZE // 2
    print(f"Distribution after exclusions (need {half_size} pos + {half_size} neg):")
    for m in score_cols:
        pos_count = sum(1 for idx in all_indices if df_values[idx][m] > THRESHOLD)
        neg_count = len(all_indices) - pos_count
        pos_status = "✓" if pos_count >= half_size else f"✗ NEED {half_size - pos_count} MORE"
        neg_status = "✓" if neg_count >= half_size else f"✗ NEED {half_size - neg_count} MORE"
        print(f"  {short_name(m)}: {pos_count} pos ({pos_status}), {neg_count} neg ({neg_status})")
    
    print(f"Finding best balance for {LARGE_SURVEY_SIZE}-question survey...")
    result, counts, tol = find_best_balance(df_values, list(all_indices), LARGE_SURVEY_SIZE, score_cols)
    if result:
        print(f"  ✓ Found! Worst metric deviation: {tol*100:.1f}% from 50%")
        return [(f"Large Survey ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE, result, counts)], score_cols, tol
    
    print(f"  ✗ Could not find any balanced dataset")
    return None, score_cols, None


def print_survey_stats(surveys, label="surveys", metrics_list=None):
    """Print detailed stats for each survey."""
    if metrics_list is None:
        metrics_list = ALL_METRICS
        
    print("\n" + "="*60)
    print(f"SUCCESS! All {len(surveys)} {label} formed successfully!")
    print("="*60 + "\n")
    
    for name, size, indices, counts in surveys:
        half = size // 2
        print(f"{name}:")
        print(f"  Indices: {len(indices)} questions")
        for m in metrics_list:
            pos = counts[m]
            neg = size - pos
            pct = 100 * pos / size
            status = "✓" if pos == half else f"({pct:.0f}%)"
            print(f"    {short_name(m)}: {pos} pos / {neg} neg {status}")
        print()


def try_scenario(df, analyze_fn, label, tolerances=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]):
    """Try a scenario with perfect balance first, then relaxed."""
    surveys = analyze_fn(df, use_relaxed=False)
    
    if surveys:
        print_survey_stats(surveys, label)
        return surveys, 0.0
    
    print(f"\nPerfect balance failed for {label}. Trying relaxed constraints...")
    for tol in tolerances:
        print(f"\n--- Tolerance: {tol} ({int((0.5-tol)*100)}-{int((0.5+tol)*100)}% balance) ---")
        surveys = analyze_fn(df, use_relaxed=True, tolerance=tol)
        if surveys:
            print_survey_stats(surveys, label)
            return surveys, tol
    
    return None, None


# =============================================================================
# Save Datasets
# =============================================================================

def save_balanced_datasets(df, surveys, output_dir, metrics_list=None):
    """Save the balanced datasets to CSV files.
    
    Args:
        df: DataFrame with all data
        surveys: List of (name, size, indices, counts) tuples
        output_dir: Directory to save files
        metrics_list: List of metric column names used for balance. If None, uses ALL_METRICS.
    """
    if metrics_list is None:
        metrics_list = ALL_METRICS
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which columns to include in output
    # Base columns that are always included
    base_columns = ['Question', 'Human Answer']
    
    # Include selected score columns
    output_columns = base_columns + metrics_list
    
    # Include reason columns for selected metrics if they exist
    reason_columns = [m.replace('_score', '_reason') for m in metrics_list]
    available_reason_columns = [col for col in reason_columns if col in df.columns]
    output_columns += available_reason_columns
    
    if available_reason_columns:
        print(f"Including {len(available_reason_columns)} reason columns in output")
    else:
        print("Note: No reason columns found in source data")
    
    # Build metric suffix for filename if specific metrics were requested
    metric_suffix = ""
    if metrics_list != ALL_METRICS:
        # Create a compact suffix from metric names (e.g., "_humor_v4" or "_metaphor_v3_humor_v2")
        metric_names = [short_name(m).replace('_score', '') for m in metrics_list]
        metric_suffix = "_" + "_".join(metric_names)
    
    for name, size, indices, counts in surveys:
        # Create filename from survey name
        if "Large" in name:
            filename = f"balanced_{LARGE_SURVEY_SIZE}{metric_suffix}.csv"
        elif "Small" in name:
            filename = f"balanced_{SMALL_SURVEY_SIZE}{metric_suffix}.csv"
        else:
            # For numbered surveys (4-survey plan)
            filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("questions", "q") + f"{metric_suffix}.csv"
        
        output_path = output_dir / filename
        # Select only the columns we want to include
        survey_df = df.loc[indices, [col for col in output_columns if col in df.columns]].copy()

        # Add index column
        survey_df['Index'] = indices

        # Rename columns to match balanced_dataset.csv format
        survey_df = survey_df.rename(columns=OUTPUT_COLUMN_RENAME)

        # Shuffle the dataset to mix positive and negative labels
        survey_df = survey_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        survey_df.to_csv(output_path, index=False)
        print(f"Saved {output_path.name} with {len(survey_df)} rows")
    
    # Print verification
    print("\nBalance verification:")
    for name, size, indices, counts in surveys:
        print(f"\n{name}:")
        for m in metrics_list:
            pos = counts[m]
            neg = size - pos
            print(f"  {short_name(m)}: {pos} pos / {neg} neg ({100*pos/size:.0f}%)")
    
    # Add all saved indices to exclusion list
    all_saved_indices = []
    for name, size, indices, counts in surveys:
        all_saved_indices.extend(indices)
    
    added_count = add_to_exclude_indices(all_saved_indices)
    if added_count > 0:
        print(f"\n✓ Added {added_count} indices to exclusion list ({EXCLUDE_INDICES_PATH.name})")
    else:
        print(f"\n• All indices already in exclusion list")


# =============================================================================
# Data Generation (for when more data is needed)
# =============================================================================

def setup_deepeval(metrics_to_run=None):
    """Setup deepeval imports and environment.
    
    Args:
        metrics_to_run: Optional list of metric names to load. If None, loads all v2 metrics.
    
    Returns:
        LLMTestCase class and dictionary of metric name -> metric function
    """
    sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "Benchmarking" / "deep_eval"))
    
    from config import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("main_logger").setLevel(logging.WARNING)
    
    from deepeval.test_case import LLMTestCase
    from custom_metrics.metrics import (
        humor_metric_explicit_v2,
        humor_metric_explicit_v3,
        humor_metric_explicit_v4,
        humor_metric_explicit_v5,
        metaphor_metric_explicit_v2,
        metaphor_metric_explicit_v3,
        metaphor_metric_explicit_v4,
        metaphor_metric_explicit_v5,
        metaphor_metric_explicit_v6,
        metaphor_metric_explicit_v7,
        metaphor_metric_explicit_v8,
        metaphor_metric_explicit_v9,
        metaphor_metric_explicit_v10,
        metaphor_metric_explicit_v11,
        metaphor_metric_explicit_v12,
        analogy_metric_explicit_v2,
        connection_to_everyday_life_metric_explicit_v2,
        connection_to_everyday_life_metric_explicit_v3,
        connection_to_everyday_life_metric_explicit_v4,
        scaffolding_metric,
    )
    
    # All available metrics
    ALL_AVAILABLE_METRICS = {
        "humor_v2": humor_metric_explicit_v2,
        "humor_v3": humor_metric_explicit_v3,
        "humor_v4": humor_metric_explicit_v4,
        "humor_v5": humor_metric_explicit_v5,
        "metaphor_v2": metaphor_metric_explicit_v2,
        "metaphor_v3": metaphor_metric_explicit_v3,
        "metaphor_v4": metaphor_metric_explicit_v4,
        "metaphor_v5": metaphor_metric_explicit_v5,
        "metaphor_v6": metaphor_metric_explicit_v6,
        "metaphor_v7": metaphor_metric_explicit_v7,
        "metaphor_v8": metaphor_metric_explicit_v8,
        "metaphor_v9": metaphor_metric_explicit_v9,
        "metaphor_v10": metaphor_metric_explicit_v10,
        "metaphor_v11": metaphor_metric_explicit_v11,
        "metaphor_v12": metaphor_metric_explicit_v12,
        "analogy_v2": analogy_metric_explicit_v2,
        "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
        "connection_to_everyday_life_v3": connection_to_everyday_life_metric_explicit_v3,
        "connection_to_everyday_life_v4": connection_to_everyday_life_metric_explicit_v4,
        "scaffolding": scaffolding_metric,
    }
    
    # Filter to requested metrics if specified
    if metrics_to_run:
        selected_metrics = {}
        for metric_name in metrics_to_run:
            if metric_name in ALL_AVAILABLE_METRICS:
                selected_metrics[metric_name] = ALL_AVAILABLE_METRICS[metric_name]
            else:
                print(f"Warning: Unknown metric '{metric_name}'. Available: {list(ALL_AVAILABLE_METRICS.keys())}")
        return LLMTestCase, selected_metrics
    
    # Default: return v2 metrics only
    V2_METRICS = {
        "humor_v2": humor_metric_explicit_v2,
        "metaphor_v2": metaphor_metric_explicit_v2,
        "analogy_v2": analogy_metric_explicit_v2,
        "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
        "scaffolding": scaffolding_metric,
    }
    
    return LLMTestCase, V2_METRICS


async def evaluate_row(index, row, metric_name, metric_function, scores, reasons, semaphore, LLMTestCase, pbar=None):
    """Evaluate a single row with a single metric."""
    async with semaphore:
        test_case = LLMTestCase(
            input=row["Question"],
            actual_output=row["Human Answer"],
        )
        success = False
        for i in range(GEVAL_RETRIES):
            try:
                await metric_function.a_measure(test_case)
                success = True
                break
            except ValueError:
                print(f"Row {index}, {metric_name}: Retry {i+1}/{GEVAL_RETRIES} - Invalid JSON")
                continue
            except Exception as e:
                print(f"Row {index}, {metric_name}: Error - {e}")
                break
        
        if success:
            scores[index] = metric_function.score
            if hasattr(metric_function, "reason") and metric_function.reason:
                reasons[index] = metric_function.reason
            else:
                reasons[index] = None
        else:
            scores[index] = None
            reasons[index] = None
            print(f"Warning: Row {index} failed for {metric_name}")
        
        if pbar:
            pbar.set_description(f"Processing {metric_name}")
            pbar.update(1)


async def process_batch(batch_df, semaphore, LLMTestCase, V2_METRICS):
    """Process a batch of questions with all metrics."""
    indices = list(batch_df.index)
    
    total_tasks = len(batch_df) * len(V2_METRICS)
    pbar = tqdm(total=total_tasks, desc="Processing batch")
    
    for metric_name, metric_function in V2_METRICS.items():
        scores = {}
        reasons = {}
        tasks = [
            evaluate_row(index, row, metric_name, metric_function, scores, reasons, semaphore, LLMTestCase, pbar)
            for index, row in batch_df.iterrows()
        ]
        await asyncio.gather(*tasks)
        
        # Convert dicts to lists with consistent ordering
        scores_list = [scores.get(i, None) for i in indices]
        reasons_list = [reasons.get(i, None) for i in indices]
        
        batch_df[f"{metric_name}_score"] = scores_list
        batch_df[f"{metric_name}_reason"] = reasons_list
    
    pbar.close()
    return batch_df


def smart_sample(metrics_df, source_df, metric_name, batch_size):
    """
    Sample a stratified dataset based on previous versions of a metric.
    
    For example, for metaphor_v5:
    1. First try to sample positives/negatives from metaphor_v4_score
    2. If depleted, fall back to metaphor_v3_score for more examples
    3. If still depleted, fall back to metaphor_v2_score
    4. Finally fall back to source_df for completely new samples
    
    Samples half from positive (>0.5) and half from negative (<=0.5).
    
    Excludes:
    - Rows where the target metric already has a value
    - Rows in exclude_indices.csv
    - Rows with answers too long (> MAX_ANSWER_LENGTH)
    """
    prev_metrics = METRIC_PREVIOUS_VERSIONS.get(metric_name)
    if not prev_metrics:
        print(f"Warning: No previous version mapping for '{metric_name}'. Using sequential sampling.")
        return None
    
    # Find which previous metric columns exist
    available_prev_cols = []
    for candidate in prev_metrics:
        candidate_col = f"{candidate}_score"
        if candidate_col in metrics_df.columns:
            available_prev_cols.append(candidate_col)
    
    if not available_prev_cols:
        tried = [f"{m}_score" for m in prev_metrics]
        print(f"Warning: No previous metric columns found (tried: {tried}). Using sequential sampling.")
        return None
    
    target_score_col = f"{metric_name}_score"
    print(f"Smart sampling with fallback chain: {available_prev_cols}")
    
    # Get indices already in metrics_df
    existing_indices = set(metrics_df.index.tolist())
    
    # Load excluded indices
    exclude_indices = set(load_exclude_indices())
    
    # Start with all rows
    available_df = metrics_df.copy()
    
    # Filter out rows where target metric already has a value
    if target_score_col in available_df.columns:
        unevaluated_mask = available_df[target_score_col].isna()
        already_evaluated = (~unevaluated_mask).sum()
        available_df = available_df[unevaluated_mask]
        print(f"Excluding {already_evaluated} rows where {target_score_col} already exists")
    
    # Filter out rows with answers too long
    if 'Human Answer' in available_df.columns:
        long_answer_mask = available_df['Human Answer'].str.len() > MAX_ANSWER_LENGTH
        long_answers = long_answer_mask.sum()
        if long_answers > 0:
            available_df = available_df[~long_answer_mask]
            print(f"Excluding {long_answers} rows with answers > {MAX_ANSWER_LENGTH} chars")
    
    # Filter out excluded indices
    excluded_in_df = set(available_df.index.tolist()) & exclude_indices
    if excluded_in_df:
        available_df = available_df.drop(index=list(excluded_in_df), errors='ignore')
        print(f"Excluding {len(excluded_in_df)} rows from exclude_indices.csv")
    
    half_batch = batch_size // 2
    sampled_positive = []
    sampled_negative = []
    used_indices = set()
    
    print(f"Smart sampling for {metric_name}:")
    print(f"  Target: {half_batch} positive + {half_batch} negative = {batch_size} total")
    
    # Debug: show what's available for each version
    print(f"  Available rows per version (before sampling):")
    for prev_score_col in available_prev_cols:
        has_metric = available_df[prev_score_col].notna()
        version_df = available_df[has_metric]
        pos_count = (version_df[prev_score_col] > THRESHOLD).sum()
        neg_count = (version_df[prev_score_col] <= THRESHOLD).sum()
        print(f"    {prev_score_col}: {pos_count} pos, {neg_count} neg (total: {len(version_df)})")
    
    # Prioritize rows evaluated by newer metrics first
    # For each metric version (newest to oldest), take rows that have THAT version's score
    # but haven't been sampled yet
    for prev_score_col in available_prev_cols:
        if len(sampled_positive) >= half_batch and len(sampled_negative) >= half_batch:
            break  # Already have enough
        
        # Get rows that have this metric evaluated and aren't already sampled
        has_metric = available_df[prev_score_col].notna()
        candidates_df = available_df[has_metric & ~available_df.index.isin(used_indices)]
        
        if len(candidates_df) == 0:
            continue
        
        # Split by positive/negative using THIS metric's score
        positive_mask = candidates_df[prev_score_col] > THRESHOLD
        negative_mask = candidates_df[prev_score_col] <= THRESHOLD
        
        positive_indices = candidates_df[positive_mask].index.tolist()
        negative_indices = candidates_df[negative_mask].index.tolist()
        
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)
        
        # Take what we need (up to the remaining quota)
        pos_needed = half_batch - len(sampled_positive)
        neg_needed = half_batch - len(sampled_negative)
        
        new_positive = positive_indices[:pos_needed]
        new_negative = negative_indices[:neg_needed]
        
        if new_positive or new_negative:
            print(f"  From {prev_score_col}: +{len(new_positive)} pos, +{len(new_negative)} neg")
        
        sampled_positive.extend(new_positive)
        sampled_negative.extend(new_negative)
        used_indices.update(new_positive)
        used_indices.update(new_negative)
    
    # Check shortfall after trying all previous versions
    pos_shortfall = half_batch - len(sampled_positive)
    neg_shortfall = half_batch - len(sampled_negative)
    shortfall = pos_shortfall + neg_shortfall
    
    if pos_shortfall > 0:
        print(f"  Shortfall in positive after all versions: {pos_shortfall}")
    if neg_shortfall > 0:
        print(f"  Shortfall in negative after all versions: {neg_shortfall}")
    
    sampled_indices = sampled_positive + sampled_negative
    
    # Fallback to source_df for shortfall
    if shortfall > 0:
        print(f"  Falling back to source CSV for {shortfall} additional samples...")
        # Get indices not yet in metrics_df, excluding long answers and excluded indices
        all_source_indices = set(range(len(source_df)))
        available_source_indices = all_source_indices - existing_indices - exclude_indices
        
        # Filter out long answers from source
        if 'Human Answer' in source_df.columns:
            valid_source_indices = []
            for idx in available_source_indices:
                if idx < len(source_df) and len(str(source_df.iloc[idx]['Human Answer'])) <= MAX_ANSWER_LENGTH:
                    valid_source_indices.append(idx)
            available_source_indices = valid_source_indices
        else:
            available_source_indices = list(available_source_indices)
        
        random.shuffle(available_source_indices)
        
        fallback_indices = available_source_indices[:shortfall]
        print(f"  Added {len(fallback_indices)} samples from source CSV (after filtering)")
        
        # Return both: sampled from metrics_df and new from source_df
        return {
            'existing_indices': sampled_indices,
            'new_source_indices': fallback_indices
        }
    
    return {
        'existing_indices': sampled_indices,
        'new_source_indices': []
    }


async def generate_more_data(num_batches=1, metrics_to_run=None, smart_sample_enabled=False, batch_size=None):
    """Generate more metric data from source CSV.
    
    Args:
        num_batches: Number of batches to process
        metrics_to_run: List of specific metrics to run (e.g., ['metaphor_v3'])
        smart_sample_enabled: If True, use stratified sampling based on previous metric version
        batch_size: Number of samples per batch
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        
    print("\n" + "="*60)
    print("GENERATING MORE DATA")
    print("="*60)
    
    LLMTestCase, METRICS = setup_deepeval(metrics_to_run)
    
    if not METRICS:
        print("Error: No valid metrics to run.")
        return None
    
    print(f"Metrics to evaluate: {list(METRICS.keys())}")
    
    source_df = pd.read_csv(SOURCE_PATH)
    print(f"Source dataset has {len(source_df)} rows")
    
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        print(f"Existing metrics file has {len(metrics_df)} rows")
    else:
        metrics_df = pd.DataFrame()
        print("No existing metrics file found. Starting fresh.")
    
    semaphore = asyncio.Semaphore(40)
    
    for batch_num in range(num_batches):
        print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")
        
        # Determine which samples to process
        if smart_sample_enabled and metrics_to_run and len(metrics_to_run) == 1 and not metrics_df.empty:
            metric_name = metrics_to_run[0]
            sample_result = smart_sample(metrics_df, source_df, metric_name, batch_size)
            
            if sample_result:
                existing_indices = sample_result['existing_indices']
                new_source_indices = sample_result['new_source_indices']
                
                # Process existing rows (just add new metric columns)
                if existing_indices:
                    print(f"Processing {len(existing_indices)} existing rows with new metric...")
                    batch_df = metrics_df.loc[existing_indices].copy()
                    processed_batch = await process_batch(batch_df, semaphore, LLMTestCase, METRICS)
                    
                    # Update the existing rows in metrics_df with new metric columns
                    for col in processed_batch.columns:
                        if col.endswith('_score') or col.endswith('_reason'):
                            if col not in metrics_df.columns:
                                metrics_df[col] = None
                            metrics_df.loc[existing_indices, col] = processed_batch[col].values
                
                # Process new source rows
                if new_source_indices:
                    print(f"Processing {len(new_source_indices)} new rows from source...")
                    new_batch_df = source_df.iloc[new_source_indices].copy()
                    processed_new = await process_batch(new_batch_df, semaphore, LLMTestCase, METRICS)
                    metrics_df = pd.concat([metrics_df, processed_new], ignore_index=True)
            else:
                # Fallback to sequential processing
                print("Smart sampling not available. Using sequential processing.")
                processed_count = len(metrics_df)
                start_idx = processed_count
                end_idx = min(start_idx + batch_size, len(source_df))
                
                if start_idx >= len(source_df):
                    print("\nProcessed all available data!")
                    break
                
                batch_df = source_df.iloc[start_idx:end_idx].copy()
                processed_batch = await process_batch(batch_df, semaphore, LLMTestCase, METRICS)
                
                if metrics_df.empty:
                    metrics_df = processed_batch
                else:
                    metrics_df = pd.concat([metrics_df, processed_batch], ignore_index=True)
        else:
            # Original sequential processing
            processed_count = len(metrics_df)
            start_idx = processed_count
            end_idx = min(start_idx + batch_size, len(source_df))
            
            if start_idx >= len(source_df):
                print("\nProcessed all available data!")
                break
            
            print(f"Processing rows {start_idx} to {end_idx-1}")
            
            batch_df = source_df.iloc[start_idx:end_idx].copy()
            processed_batch = await process_batch(batch_df, semaphore, LLMTestCase, METRICS)
            
            if metrics_df.empty:
                metrics_df = processed_batch
            else:
                metrics_df = pd.concat([metrics_df, processed_batch], ignore_index=True)
        
        # Save after each batch
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"Saved {len(metrics_df)} rows to {METRICS_PATH}")
    
    return metrics_df


# =============================================================================
# Main
# =============================================================================

def main():
    global LARGE_SURVEY_SIZE
    
    parser = argparse.ArgumentParser(description="Analyze and generate balanced survey datasets")
    parser.add_argument("--save", action="store_true", help="Save the best achievable datasets to CSV")
    parser.add_argument("--large", action="store_true", default=True,
                        help="Generate only the large survey (no small survey) [default: True]")
    parser.add_argument("--no-large", action="store_false", dest="large",
                        help="Generate both large and small surveys")
    parser.add_argument("--large-size", type=int, default=LARGE_SURVEY_SIZE,
                        help=f"Size of the large survey (default: {LARGE_SURVEY_SIZE})")
    parser.add_argument("--generate", type=int, nargs="?", const=1, metavar="N",
                        help="Generate N more batches of data (default: 1)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated list of metrics to run (e.g., 'metaphor_v3,humor_v2')")
    parser.add_argument("--smart-sample", action="store_true",
                        help="Use stratified sampling based on previous metric version scores")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of samples per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--min-tolerance", type=float, default=0.1,
                        help="Minimum acceptable balance tolerance (default: 0.1 = 40-60%% split)")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum generation iterations before giving up (default: 50)")
    args = parser.parse_args()
    
    # Update LARGE_SURVEY_SIZE if specified
    LARGE_SURVEY_SIZE = args.large_size
    
    # Parse metrics list
    metrics_to_run = None
    if args.metrics:
        metrics_to_run = [m.strip() for m in args.metrics.split(',')]
        print(f"Metrics specified: {metrics_to_run}")
    
    # Convert to score column names for balance analysis
    balance_metrics = None
    if metrics_to_run:
        balance_metrics = [m if m.endswith('_score') else f"{m}_score" for m in metrics_to_run]
    
    # Load data first to check if balance is already achievable
    if not METRICS_PATH.exists():
        if args.generate:
            print(f"Note: {METRICS_PATH} not found. Will create during generation.")
        else:
            print(f"Error: {METRICS_PATH} not found. Run with --generate first.")
            return
        df = None
    else:
        print(f"Loading data from: {METRICS_PATH}")
        df = pd.read_csv(METRICS_PATH)
        print(f"Total rows: {len(df)}")
        
        # Filter out answers that are too long for labeling tasks
        original_len = len(df)
        df = df[df['Human Answer'].str.len() <= MAX_ANSWER_LENGTH]
        filtered_count = original_len - len(df)
        print(f"Filtered out {filtered_count} rows with answers > {MAX_ANSWER_LENGTH} chars")
        print(f"Remaining rows: {len(df)}")
        
        if args.large:
            print(f"Large-only plan needs: {LARGE_SURVEY_SIZE} questions\n")
        else:
            total_2 = LARGE_SURVEY_SIZE + SMALL_SURVEY_SIZE
            print(f"2-survey plan needs: {total_2} questions (1×{LARGE_SURVEY_SIZE} + 1×{SMALL_SURVEY_SIZE})\n")
        
        # Show distribution for selected metrics
        display_metrics = balance_metrics if balance_metrics else ALL_METRICS
        print(f"Metric distribution in {METRICS_PATH.name} (threshold={THRESHOLD}):")
        for m in display_metrics:
            if m in df.columns:
                valid_count = df[m].notna().sum()
                pos = (df[m] > THRESHOLD).sum()
                neg = (df[m] <= THRESHOLD).sum()
                print(f"  {short_name(m)}: {pos} pos / {neg} neg ({100*pos/valid_count:.1f}% pos) [{valid_count} evaluated]")
            else:
                print(f"  {short_name(m)}: [not in dataset]")
        
        # ========== TRY BALANCE FIRST ==========
        achieved_tolerance = None
        if args.large:
            print("\n" + "#"*70)
            print("# LARGE SURVEY ONLY (1×{})".format(LARGE_SURVEY_SIZE))
            print("#"*70)
            best_surveys, used_metrics, achieved_tolerance = analyze_large_only(df, metrics_list=balance_metrics)
        else:
            print("\n" + "#"*70)
            print("# BEST ACHIEVABLE BALANCE (2 surveys: 1×{} + 1×{})".format(LARGE_SURVEY_SIZE, SMALL_SURVEY_SIZE))
            print("#"*70)
            best_surveys = analyze_best_achievable(df)
            used_metrics = ALL_METRICS
        
        # Check if balance is acceptable (within min_tolerance)
        balance_acceptable = best_surveys and (achieved_tolerance is None or achieved_tolerance <= args.min_tolerance)
        
        if best_surveys:
            print_survey_stats(best_surveys, "best achievable", metrics_list=used_metrics)
            
            if not balance_acceptable:
                print(f"\n⚠ Balance found but tolerance ({achieved_tolerance*100:.1f}%) exceeds --min-tolerance ({args.min_tolerance*100:.0f}%)")
                print("  Need more data for better balance.")
            elif args.save:
                print("\n" + "-"*60)
                print("SAVING DATASETS")
                print("-"*60)
                save_balanced_datasets(df, best_surveys, OUTPUT_DIR, metrics_list=used_metrics)
        
        # If balance is acceptable, we're done
        if balance_acceptable:
            if args.save:
                print(f"\n✓ Datasets saved to {OUTPUT_DIR}/")
            print("\nDone.")
            return
        
        # If balance not acceptable and generation requested, continue to generation
        if args.generate:
            print("\nProceeding with generation to improve balance...")
    
    # ========== GENERATE DATA (with loop until balance achieved) ==========
    if args.generate:
        async def run_generation_loop():
            iteration = 0
            balance_acceptable = False
            nonlocal df, best_surveys
            
            while not balance_acceptable and iteration < args.max_iterations:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"GENERATION ITERATION {iteration}/{args.max_iterations}")
                print(f"{'='*60}")
                
                await generate_more_data(
                    num_batches=args.generate,
                    metrics_to_run=metrics_to_run,
                    smart_sample_enabled=args.smart_sample,
                    batch_size=args.batch_size
                )
                
                # Reload and re-analyze after generation
                print("\n" + "-"*40)
                print("RE-ANALYZING AFTER GENERATION")
                print("-"*40)
                
                df = pd.read_csv(METRICS_PATH)
                original_len = len(df)
                df = df[df['Human Answer'].str.len() <= MAX_ANSWER_LENGTH]
                print(f"Loaded {len(df)} rows (filtered {original_len - len(df)} long answers)")
                
                if args.large:
                    best_surveys, used_metrics, achieved_tolerance = analyze_large_only(df, metrics_list=balance_metrics)
                else:
                    best_surveys = analyze_best_achievable(df)
                    used_metrics_loop = ALL_METRICS
                    achieved_tolerance = None
                
                if best_surveys:
                    used_metrics_loop = used_metrics if args.large else ALL_METRICS
                    print_survey_stats(best_surveys, "best achievable", metrics_list=used_metrics_loop)
                    
                    balance_acceptable = achieved_tolerance is None or achieved_tolerance <= args.min_tolerance
                    if balance_acceptable:
                        print(f"\n✓ Balance achieved! (tolerance {achieved_tolerance*100:.1f}% <= {args.min_tolerance*100:.0f}%)")
                        if args.save:
                            print("\n" + "-"*60)
                            print("SAVING DATASETS")
                            print("-"*60)
                            save_balanced_datasets(df, best_surveys, OUTPUT_DIR, metrics_list=used_metrics_loop)
                    else:
                        print(f"\n⚠ Balance not yet acceptable (tolerance {achieved_tolerance*100:.1f}% > {args.min_tolerance*100:.0f}%)")
                        print(f"  Continuing... ({args.max_iterations - iteration} iterations remaining)")
                else:
                    print("\n✗ Could not find any balanced dataset. Continuing generation...")
            
            return balance_acceptable
        
        balance_acceptable = asyncio.run(run_generation_loop())
        
        if not balance_acceptable:
            print(f"\n✗ Could not achieve acceptable balance after {args.max_iterations} iterations.")
            print("  Try increasing --max-iterations or relaxing --min-tolerance.")
        
        print("\nDone.")
        return
    
    # No generation requested and balance not found
    if df is not None and not best_surveys:
        print("\n✗ Cannot find balanced dataset. Run with --generate to add more data.")


if __name__ == "__main__":
    main()
