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

# Constants
THRESHOLD = 0.5
LARGE_SURVEY_SIZE = 30
SMALL_SURVEY_SIZE = 10
BATCH_SIZE = 100
GEVAL_RETRIES = 3
MAX_ANSWER_LENGTH = 2560  # Exclude answers longer than this for labeling tasks

# Indexes to exclude (already used in balanced_10.csv test tagging)
EXCLUDE_INDICES = [1217, 549, 1188, 1575, 128]

ALL_METRICS = [
    'humor_v2_score', 
    'metaphor_v2_score', 
    'analogy_v2_score', 
    'connection_to_everyday_life_v2_score', 
    'scaffolding_score'
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


def analyze_large_only(df):
    """Find the best achievable balance for a single large survey only."""
    print("\n=== Finding BEST achievable balance for LARGE SURVEY ONLY ===\n")
    
    valid_df = df.dropna()
    all_indices = set(valid_df.index.tolist())
    
    # Exclude indices already used in balanced_10.csv
    if EXCLUDE_INDICES:
        excluded = set(EXCLUDE_INDICES) & all_indices
        all_indices = all_indices - excluded
        print(f"Excluding {len(excluded)} indices from previous survey: {sorted(excluded)}")
    
    df_values = {idx: {m: valid_df.at[idx, m] for m in ALL_METRICS} for idx in all_indices}
    
    print(f"Finding best balance for {LARGE_SURVEY_SIZE}-question survey...")
    result, counts, tol = find_best_balance(df_values, list(all_indices), LARGE_SURVEY_SIZE, ALL_METRICS)
    if result:
        print(f"  ✓ Found! Worst metric deviation: {tol*100:.1f}% from 50%")
        return [(f"Large Survey ({LARGE_SURVEY_SIZE} questions)", LARGE_SURVEY_SIZE, result, counts)]
    
    return None


def print_survey_stats(surveys, label="surveys"):
    """Print detailed stats for each survey."""
    print("\n" + "="*60)
    print(f"SUCCESS! All {len(surveys)} {label} formed successfully!")
    print("="*60 + "\n")
    
    for name, size, indices, counts in surveys:
        half = size // 2
        print(f"{name}:")
        print(f"  Indices: {len(indices)} questions")
        for m in ALL_METRICS:
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

def save_balanced_datasets(df, surveys, output_dir):
    """Save the balanced datasets to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, size, indices, counts in surveys:
        # Create filename from survey name
        if "Large" in name:
            filename = f"balanced_{LARGE_SURVEY_SIZE}.csv"
        elif "Small" in name:
            filename = f"balanced_{SMALL_SURVEY_SIZE}.csv"
        else:
            # For numbered surveys (4-survey plan)
            filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("questions", "q") + ".csv"
        
        output_path = output_dir / filename
        survey_df = df.loc[indices].copy()
        
        # Rename columns to match balanced_dataset.csv format
        survey_df = survey_df.rename(columns=OUTPUT_COLUMN_RENAME)
        
        survey_df.to_csv(output_path, index=False)
        print(f"Saved {output_path.name} with {len(survey_df)} rows")
    
    # Print verification
    print("\nBalance verification:")
    for name, size, indices, counts in surveys:
        print(f"\n{name}:")
        for m in ALL_METRICS:
            pos = counts[m]
            neg = size - pos
            print(f"  {short_name(m)}: {pos} pos / {neg} neg ({100*pos/size:.0f}%)")


# =============================================================================
# Data Generation (for when more data is needed)
# =============================================================================

def setup_deepeval():
    """Setup deepeval imports and environment."""
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
        metaphor_metric_explicit_v2,
        analogy_metric_explicit_v2,
        connection_to_everyday_life_metric_explicit_v2,
        scaffolding_metric,
    )
    
    V2_METRICS = {
        "humor_v2": humor_metric_explicit_v2,
        "metaphor_v2": metaphor_metric_explicit_v2,
        "analogy_v2": analogy_metric_explicit_v2,
        "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
        "scaffolding": scaffolding_metric,
    }
    
    return LLMTestCase, V2_METRICS


async def evaluate_row(index, row, metric_name, metric_function, scores, semaphore, LLMTestCase, pbar=None):
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
        else:
            scores[index] = None
            print(f"Warning: Row {index} failed for {metric_name}")
        
        if pbar:
            pbar.set_description(f"Processing {metric_name}")
            pbar.update(1)


async def process_batch(batch_df, semaphore, LLMTestCase, V2_METRICS):
    """Process a batch of questions with all metrics."""
    SCORE_COLUMNS = [f"{name}_score" for name in V2_METRICS.keys()]
    
    for col in SCORE_COLUMNS:
        batch_df[col] = None

    total_tasks = len(batch_df) * len(V2_METRICS)
    pbar = tqdm(total=total_tasks, desc="Processing batch")
    
    for metric_name, metric_function in V2_METRICS.items():
        scores = {}
        tasks = [
            evaluate_row(index, row, metric_name, metric_function, scores, semaphore, LLMTestCase, pbar)
            for index, row in batch_df.iterrows()
        ]
        await asyncio.gather(*tasks)
        
        for idx, score in scores.items():
            batch_df.at[idx, f"{metric_name}_score"] = score
    
    pbar.close()
    return batch_df


async def generate_more_data(num_batches=1):
    """Generate more metric data from source CSV."""
    print("\n" + "="*60)
    print("GENERATING MORE DATA")
    print("="*60)
    
    LLMTestCase, V2_METRICS = setup_deepeval()
    
    source_df = pd.read_csv(SOURCE_PATH)
    print(f"Source dataset has {len(source_df)} rows")
    
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        processed_count = len(metrics_df)
        print(f"Already processed: {processed_count} rows")
    else:
        metrics_df = pd.DataFrame()
        processed_count = 0
    
    semaphore = asyncio.Semaphore(40)
    
    for batch_num in range(num_batches):
        start_idx = processed_count
        end_idx = min(start_idx + BATCH_SIZE, len(source_df))
        
        if start_idx >= len(source_df):
            print("\nProcessed all available data!")
            break
        
        print(f"\nBatch {batch_num + 1}: Processing rows {start_idx} to {end_idx-1}")
        
        batch_df = source_df.iloc[start_idx:end_idx].copy()
        processed_batch = await process_batch(batch_df, semaphore, LLMTestCase, V2_METRICS)
        
        if metrics_df.empty:
            metrics_df = processed_batch
        else:
            metrics_df = pd.concat([metrics_df, processed_batch], ignore_index=True)
        
        processed_count = len(metrics_df)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"Saved {processed_count} rows to {METRICS_PATH}")
    
    return metrics_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze and generate balanced survey datasets")
    parser.add_argument("--save", action="store_true", help="Save the best achievable datasets to CSV")
    parser.add_argument("--large", action="store_true", help="Generate only the large survey (no small survey)")
    parser.add_argument("--generate", type=int, nargs="?", const=1, metavar="N",
                        help="Generate N more batches of data (default: 1)")
    args = parser.parse_args()
    
    # Generate more data if requested
    if args.generate:
        asyncio.run(generate_more_data(args.generate))
    
    # Load data
    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} not found. Run with --generate first.")
        return
    
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
    
    # Show distribution
    print("Current metric distribution:")
    for m in ALL_METRICS:
        pos = (df[m] > THRESHOLD).sum()
        neg = (df[m] <= THRESHOLD).sum()
        print(f"  {short_name(m)}: {pos} pos, {neg} neg ({100*pos/len(df):.1f}% pos)")
    
    # ========== ANALYSIS ==========
    if args.large:
        print("\n" + "#"*70)
        print("# LARGE SURVEY ONLY (1×{})".format(LARGE_SURVEY_SIZE))
        print("#"*70)
        best_surveys = analyze_large_only(df)
    else:
        print("\n" + "#"*70)
        print("# BEST ACHIEVABLE BALANCE (2 surveys: 1×{} + 1×{})".format(LARGE_SURVEY_SIZE, SMALL_SURVEY_SIZE))
        print("#"*70)
        best_surveys = analyze_best_achievable(df)
    
    if best_surveys:
        print_survey_stats(best_surveys, "best achievable")
        
        if args.save:
            print("\n" + "-"*60)
            print("SAVING DATASETS")
            print("-"*60)
            save_balanced_datasets(df, best_surveys, OUTPUT_DIR)
    
    # # ========== ORIGINAL PLAN: 2 surveys ==========
    # print("\n" + "#"*70)
    # print("# ORIGINAL PLAN: 2 surveys (1×{} + 1×{})".format(LARGE_SURVEY_SIZE, SMALL_SURVEY_SIZE))
    # print("#"*70)
    #
    # surveys_2, tol_2 = try_scenario(df, analyze_two_surveys, "2 surveys")
    #
    # # ========== EXPANDED PLAN: 4 surveys ==========
    # print("\n" + "#"*70)
    # print("# EXPANDED PLAN: 4 surveys (2×{} + 2×{})".format(LARGE_SURVEY_SIZE, SMALL_SURVEY_SIZE))
    # print("#"*70)
    #
    # surveys_4, tol_4 = try_scenario(df, analyze_four_surveys, "4 surveys")
    
    # ========== SUMMARY ==========
    # print("\n" + "="*70)
    # print("SUMMARY")
    # print("="*70)
    #
    # if surveys_2:
    #     if tol_2 == 0.0:
    #         print(f"✓ 2-survey plan: PERFECT BALANCE achieved")
    #     else:
    #         print(f"✓ 2-survey plan: Achieved with {int((0.5-tol_2)*100)}-{int((0.5+tol_2)*100)}% tolerance")
    # else:
    #     print("✗ 2-survey plan: Could not be achieved even with 10-90% tolerance")
    #
    # if surveys_4:
    #     if tol_4 == 0.0:
    #         print(f"✓ 4-survey plan: PERFECT BALANCE achieved")
    #     else:
    #         print(f"✓ 4-survey plan: Achieved with {int((0.5-tol_4)*100)}-{int((0.5+tol_4)*100)}% tolerance")
    # else:
    #     print("✗ 4-survey plan: Could not be achieved even with 10-90% tolerance")
    #
    if args.save and best_surveys:
        print(f"\n✓ Datasets saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
