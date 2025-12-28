#!/usr/bin/env python3
"""
Run connection_to_everyday_life v3 metric on all rows and add to CSV.
Uses async gather for parallel execution.
"""
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add the Benchmarking directory to the path
SCRIPT_DIR = Path(__file__).parent
BENCHMARKING_DIR = SCRIPT_DIR.parent.parent / "Benchmarking" / "deep_eval"
sys.path.insert(0, str(BENCHMARKING_DIR))

# Setup API keys before importing deepeval
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Suppress debug logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Constants
GEVAL_RETRIES = 3
CONCURRENCY_LIMIT = 10


async def evaluate_row(index, row, metric_function, scores, reasons, semaphore, LLMTestCase, pbar=None):
    """Evaluate a single row with the metric."""
    async with semaphore:
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
        )
        success = False
        for i in range(GEVAL_RETRIES):
            try:
                await metric_function.a_measure(test_case)
                success = True
                break
            except ValueError:
                print(f"Row {index}: Retry {i+1}/{GEVAL_RETRIES} - Invalid JSON")
                continue
            except Exception as e:
                print(f"Row {index}: Error - {e}")
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
            print(f"Warning: Row {index} failed")
        
        if pbar:
            pbar.update(1)


async def main():
    from deepeval.test_case import LLMTestCase
    from custom_metrics.metrics import connection_to_everyday_life_metric_explicit_v3
    
    # Load the balanced dataset
    data_path = SCRIPT_DIR / "balanced_dataset_v2_human" / "balanced_30_formatted.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Running connection_to_everyday_life v3 metric on all rows (parallel)...")
    print()
    
    # Setup for parallel execution
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    scores = {}
    reasons = {}
    
    pbar = tqdm(total=len(df), desc="Processing v3 metric")
    
    # Create tasks for all rows
    tasks = [
        evaluate_row(
            row['Index'], row, connection_to_everyday_life_metric_explicit_v3,
            scores, reasons, semaphore, LLMTestCase, pbar
        )
        for _, row in df.iterrows()
    ]
    
    # Run all tasks in parallel
    await asyncio.gather(*tasks)
    pbar.close()
    
    # Add columns to dataframe (maintain order by Index)
    df['connection_v3_score'] = df['Index'].map(scores)
    df['connection_v3_reason'] = df['Index'].map(reasons)
    
    # Save back to CSV
    df.to_csv(data_path, index=False)
    print(f"\nSaved updated CSV with v3 columns to {data_path}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON: v2 vs v3 alignment with human coders")
    print("="*80)
    
    v2_matches = 0
    v3_matches = 0
    total = len(df)
    
    for i, row in df.iterrows():
        mattan = row['connection_mattan_yeroushalmi']
        nir = row['connection_nirgrn']
        human_consensus = round((mattan + nir) / 2)
        
        v2_score = row['connection_to_everyday_life_v2_score']
        v3_score = row['connection_v3_score']
        
        v2_binary = 1 if v2_score > 0.5 else 0
        v3_binary = 1 if v3_score and v3_score > 0.5 else 0
        
        if v2_binary == human_consensus:
            v2_matches += 1
        if v3_binary == human_consensus:
            v3_matches += 1
    
    print(f"\nv2 agreement with human consensus: {v2_matches}/{total} ({100*v2_matches/total:.1f}%)")
    print(f"v3 agreement with human consensus: {v3_matches}/{total} ({100*v3_matches/total:.1f}%)")
    
    # Show cases where v2 and v3 differ
    print("\n" + "="*80)
    print("CASES WHERE v2 AND v3 DIFFER")
    print("="*80)
    
    for i, row in df.iterrows():
        v2_score = row['connection_to_everyday_life_v2_score']
        v3_score = row['connection_v3_score'] if row['connection_v3_score'] else 0
        
        v2_binary = 1 if v2_score > 0.5 else 0
        v3_binary = 1 if v3_score > 0.5 else 0
        
        if v2_binary != v3_binary:
            mattan = row['connection_mattan_yeroushalmi']
            nir = row['connection_nirgrn']
            q = row['question'][:50] + "..." if len(row['question']) > 50 else row['question']
            print(f"\nIndex {row['Index']}: {q}")
            print(f"  Human: Mattan={mattan}, Nir={nir}")
            print(f"  v2={v2_score:.2f} → {v2_binary}, v3={v3_score:.2f} → {v3_binary}")
            reason = row['connection_v3_reason'] or ""
            print(f"  v3 reason: {reason[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
