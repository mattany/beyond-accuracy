import os
import sys
import pandas as pd
import asyncio
import random
import numpy as np
from tqdm import tqdm
import logging


# Suppress DEBUG logs from deepeval
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("main_logger").setLevel(logging.INFO)

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROJECT_DIR, OPENAI_API_KEY
from deepeval.test_case import LLMTestCase

# Set OpenAI Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Import Metrics (Baram Tsabari Cluster)
from custom_metrics.metrics import (
    jargon_metric,
    explanation_type_metric_explicit,
    metaphor_metric_explicit,
    content_units_metric_explicit,
    humor_metric_explicit,
    analogy_metric_explicit,
    connection_to_everyday_life_metric_explicit,
)

# Metrics Map
METRICS = {
    "jargon": jargon_metric,
    "explanation_type": explanation_type_metric_explicit,
    "metaphor_explicit": metaphor_metric_explicit,
    "content_units_explicit": content_units_metric_explicit,
    "humor_explicit": humor_metric_explicit,
    "analogy_explicit": analogy_metric_explicit,
    "connection_to_everyday_life": connection_to_everyday_life_metric_explicit,
}

# Run 7 Models
MODELS = [
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt',
    'SciComma-3.1-8B_y',
    'SciComma-3.1-8B_prompt',
    'scicomma-3.1-dpo',
    'scicomma-3.1-dpo_prompt'
]

DATASET_PATH = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv"

REPETITIONS = 10  # Number of times to run each metric per pair
SAMPLE_SIZE = 15

OUTPUT_DIR = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/consistency_check/{SAMPLE_SIZE}_samples_{REPETITIONS}_tries/"

async def evaluate_pair(metric_name, metric_func, question, answer, model_name, row_idx, repetition_idx):
    """Evaluates a single pair with a single metric."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )
    
    try:
        await metric_func.a_measure(test_case)
        score = metric_func.score
        reason = metric_func.reason
        return {
            "question_idx": row_idx,
            "model": model_name,
            "metric": metric_name,
            "repetition": repetition_idx,
            "score": score,
            "explanation": reason,
            "question": question,
            "answer": answer[:100] + "..." if len(answer) > 100 else answer # Truncate for CSV readability if needed, but keeping full is better for debugging.
        }
    except Exception as e:
        print(f"Error evaluating {metric_name} for model {model_name} (idx {row_idx}, rep {repetition_idx}): {e}")
        return None

async def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # 1. Select SAMPLE_SIZE Random Pairs
    print(f"Selecting {SAMPLE_SIZE} random pairs...")
    selected_pairs = []
    
    # Randomly sample SAMPLE_SIZE indices
    indices = random.sample(range(len(df)), SAMPLE_SIZE)
    
    for idx in indices:
        row = df.iloc[idx]
        question = row['question']
        # Randomly select a model
        model = random.choice(MODELS)
        
        # Check if model column exists
        if model not in df.columns:
            print(f"Warning: Model {model} not found in dataset columns. Picking another...")
            # Fallback to finding a valid model column
            valid_models = [m for m in MODELS if m in df.columns]
            if not valid_models:
                print(f"No valid models found for row {idx}. Skipping.")
                continue
            model = random.choice(valid_models)
            
        answer = row[model]
        
        # Handle NaN answers
        if pd.isna(answer):
             print(f"Answer is NaN for model {model} at row {idx}. Skipping.")
             continue

        selected_pairs.append({
            "idx": idx,
            "question": question,
            "model": model,
            "answer": answer
        })

    print(f"Selected {len(selected_pairs)} pairs.")
    
    # 2. Run Metrics
    all_results = []
    
    print(f"Running metrics ({REPETITIONS} repetitions each)...")
    
    tasks = []
    # Create tasks for all metrics * all pairs * all repetitions
    # To avoid rate limits, we might want to batch or use a semaphore.
    semaphore = asyncio.Semaphore(20) # Limit concurrency
    
    async def wrapped_eval(metric_name, metric_func, pair, rep):
        async with semaphore:
            return await evaluate_pair(
                metric_name, metric_func, 
                pair['question'], pair['answer'], pair['model'], pair['idx'], rep
            )

    for metric_name, metric_func in METRICS.items():
        for pair in selected_pairs:
            for i in range(REPETITIONS):
                tasks.append(wrapped_eval(metric_name, metric_func, pair, i+1))
    
    # Run with progress bar
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await f
        if res:
            results.append(res)
            
    # 3. Save Intermediate Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    intermediate_df = pd.DataFrame(results)
    intermediate_path = os.path.join(OUTPUT_DIR, "intermediate_results.csv")
    intermediate_df.to_csv(intermediate_path, index=False)
    print(f"Intermediate results saved to {intermediate_path}")
    
    # 4. Calculate Statistics
    print("Calculating statistics...")
    stats = []
    
    # Group by Metric, Question, Model
    grouped = intermediate_df.groupby(['metric', 'question_idx', 'model'])
    
    for (metric, q_idx, model), group in grouped:
        scores = group['score'].values
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores, ddof=1) # Sample std dev
        
        # Confidence Interval (95%) for the Mean using t-distribution or just simple normal approx since N is small
        # SE = s / sqrt(n)
        se = std_dev / np.sqrt(len(scores))
        # 95% CI approx Mean +/- 1.96 * SE (or use t-value for N-1 dof, e.g. 2.776 for N=5)
        # Using 1.96 for simplicity or t-value for N=5 is better.
        # t_value for 4 dof, 95% is 2.776
        t_value = 2.776 if len(scores) == 5 else 1.96 # Approx
        
        ci_lower = mean_score - (t_value * se)
        ci_upper = mean_score + (t_value * se)
        
        # Consistency Score (Inverse of Std Dev, or just report Std Dev)
        # We'll report Std Dev as "Inconsistency"
        
        stats.append({
            "metric": metric,
            "question_idx": q_idx,
            "model": model,
            "n_samples": len(scores),
            "mean_score": mean_score,
            "std_dev": std_dev,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "scores_list": str(list(scores))
        })
        
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(OUTPUT_DIR, "consistency_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    
    # 5. Unified Report (Average Consistency per Metric)
    metric_summary = stats_df.groupby('metric')[['std_dev', 'se']].mean().reset_index()
    metric_summary.columns = ['metric', 'avg_std_dev (inconsistency)', 'avg_se']
    summary_path = os.path.join(OUTPUT_DIR, "metric_consistency_summary.csv")
    metric_summary.to_csv(summary_path, index=False)
    
    print(f"Stats saved to {stats_path}")
    print(f"Summary saved to {summary_path}")
    print("\nMetric Consistency Summary:")
    print(metric_summary)

if __name__ == "__main__":
    asyncio.run(main())

