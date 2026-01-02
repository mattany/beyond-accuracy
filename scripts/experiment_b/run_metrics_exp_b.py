#!/usr/bin/env python3
"""
Run metrics on experiment_b sampled data.
Uses the existing deep_eval infrastructure with checkpointing (run 10).
"""

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Benchmarking" / "deep_eval"))

from config import PROJECT_DIR, OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from custom_metrics.metrics import (
    scaffolding_metric_v2,
    metaphor_metric_explicit_v8,
    humor_metric_explicit_v5,
    analogy_metric_explicit_v2,
    jargon_metric,
    flesch_kincaid,
    flesch_reading_ease,
    dale_chall,
    ari,
)
from custom_metrics.run import generate_metric_report

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_NUMBER = 10  # Separate run for experiment_b
DATA_DIR = Path(__file__).parent / "data"
SAMPLED_CSV = DATA_DIR / "experiment_b_sampled.csv"

# Metrics to run (same as experiment A)
METRICS = {
    "jargon": jargon_metric,
    "metaphor_v8": metaphor_metric_explicit_v8,
    "humor_v5": humor_metric_explicit_v5,
    "analogy_v2": analogy_metric_explicit_v2,
    "scaffolding_v2": scaffolding_metric_v2,
    "flesch_kincaid": flesch_kincaid,
    "flesch_reading_ease": flesch_reading_ease,
    "dale_chall": dale_chall,
    "ari": ari,
}


def prepare_data_for_metrics():
    """
    Prepare the sampled data for metric evaluation.
    Creates a CSV with 'question', 'explanation_a', 'explanation_b' columns.
    """
    if not SAMPLED_CSV.exists():
        print(f"Error: {SAMPLED_CSV} not found. Run sample_for_metrics.py first.")
        return None
    
    df = pd.read_csv(SAMPLED_CSV)
    print(f"Loaded {len(df)} sampled comparisons")
    
    # Create evaluation dataset with the columns expected by run.py
    # run.py expects: question, <model_name_1>, <model_name_2>, etc.
    eval_df = pd.DataFrame()
    eval_df['question'] = df['question']
    eval_df['explanation_a'] = df['explanation_a_match']  # Column name = model name
    eval_df['explanation_b'] = df['explanation_b_match']  # Column name = model name
    
    # Keep metadata for later correlation analysis
    eval_df['comparison_id'] = df.apply(lambda r: f"{r['qid']}_{r.name}", axis=1)
    eval_df['human_choice'] = df['answer']
    eval_df['cluster'] = df['cluster']
    eval_df['model_a'] = df['explanation_a_SOURCE']
    eval_df['model_b'] = df['explanation_b_SOURCE']
    
    # Save evaluation dataset
    eval_path = DATA_DIR / "experiment_b_eval_dataset.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation dataset to: {eval_path}")
    print(f"  - {len(eval_df)} comparisons")
    print(f"  - Will evaluate both explanation_a and explanation_b")
    
    return eval_df, eval_path


async def run_metrics_on_explanations():
    """
    Run all metrics on both explanation_a and explanation_b.
    Uses the existing infrastructure with checkpointing.
    """
    result = prepare_data_for_metrics()
    if result is None:
        return
    
    eval_df, eval_path = result
    
    # Run metrics using the standard infrastructure
    # We evaluate both 'explanation_a' and 'explanation_b' columns
    print("\n" + "=" * 60)
    print(f"Running metrics (run {RUN_NUMBER})")
    print("=" * 60)
    
    await generate_metric_report(
        metrics=METRICS,
        evaluation_dataset=str(eval_path),
        models_to_evaluate=['explanation_a', 'explanation_b'],  # Both columns
        run_number=RUN_NUMBER,
    )
    
    print("\n" + "=" * 60)
    print("Metrics complete!")
    print("=" * 60)
    print("\nNEXT STEP: Run python metric_correlation.py")


def main():
    asyncio.run(run_metrics_on_explanations())


if __name__ == "__main__":
    main()
