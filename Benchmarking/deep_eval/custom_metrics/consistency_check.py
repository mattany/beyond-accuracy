"""
Test metric consistency/stability by running the same metric multiple times on the same inputs.
Measures variance to understand how deterministic the LLM-based metric is.

Usage:
    cd Benchmarking/deep_eval && poetry run python custom_metrics/consistency_check.py
    
    # Run all metrics on random samples (default)
    poetry run python custom_metrics/consistency_check.py
    
    # Run only metaphor_v2 metric
    poetry run python custom_metrics/consistency_check.py --metric metaphor_v2
    
    # Use disagreement examples instead of random sampling
    poetry run python custom_metrics/consistency_check.py --use_disagreements --metric metaphor_v2
    
    # Custom sample size and repetitions
    poetry run python custom_metrics/consistency_check.py --samples 20 --reps 5
"""
import os
import sys
import argparse
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

from config import PROJECT_DIR, OPENAI_API_KEY, DEEPSEEK_API_KEY
from deepeval.test_case import LLMTestCase

# Set API Keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    # DeepSeek pricing for deepeval cost tracking
    os.environ["OPENAI_COST_PER_INPUT_TOKEN"] = "0.00000028"
    os.environ["OPENAI_COST_PER_OUTPUT_TOKEN"] = "0.00000042"

# Import Metrics
from custom_metrics.metrics import (
    jargon_metric,
    metaphor_metric_explicit,
    metaphor_metric_explicit_v2,
    metaphor_metric_explicit_v3,
    metaphor_metric_explicit_v6,
    metaphor_metric_explicit_v7,
    metaphor_metric_explicit_v8,
    metaphor_metric_explicit_v8_deepseek,
    metaphor_metric_explicit_v8_deepseek_chat,
    metaphor_metric_explicit_v8_gpt4o_mini,
    metaphor_metric_explicit_v9,
    metaphor_metric_explicit_v10,
    metaphor_metric_explicit_v11,
    metaphor_metric_explicit_v12,
    metaphor_metric_explicit_v12_deepseek,
    metaphor_metric_explicit_v12_deepseek_chat,
    metaphor_metric_explicit_v12_gpt4o_mini,
    metaphor_metric_explicit_v12_1,
    humor_metric_explicit_v2,
    analogy_metric_explicit_v2,
    connection_to_everyday_life_metric_explicit_v2,
    explanation_type_metric_explicit_v2,
    scaffolding_metric,
)

# All available metrics
ALL_METRICS = {
    "jargon": jargon_metric,
    "metaphor_v1": metaphor_metric_explicit,
    "metaphor_v2": metaphor_metric_explicit_v2,
    "metaphor_v3": metaphor_metric_explicit_v3,
    "metaphor_v6": metaphor_metric_explicit_v6,
    "metaphor_v7": metaphor_metric_explicit_v7,
    "metaphor_v8": metaphor_metric_explicit_v8,
    "metaphor_v8_deepseek": metaphor_metric_explicit_v8_deepseek,
    "metaphor_v8_deepseek_chat": metaphor_metric_explicit_v8_deepseek_chat,
    "metaphor_v8_gpt4o_mini": metaphor_metric_explicit_v8_gpt4o_mini,
    "metaphor_v9": metaphor_metric_explicit_v9,
    "metaphor_v10": metaphor_metric_explicit_v10,
    "metaphor_v11": metaphor_metric_explicit_v11,
    "metaphor_v12": metaphor_metric_explicit_v12,
    "metaphor_v12_deepseek": metaphor_metric_explicit_v12_deepseek,
    "metaphor_v12_deepseek_chat": metaphor_metric_explicit_v12_deepseek_chat,
    "metaphor_v12_gpt4o_mini": metaphor_metric_explicit_v12_gpt4o_mini,
    "metaphor_v12_1": metaphor_metric_explicit_v12_1,
    "humor_v2": humor_metric_explicit_v2,
    "analogy_v2": analogy_metric_explicit_v2,
    "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
    "explanation_type_v2": explanation_type_metric_explicit_v2,
    "scaffolding": scaffolding_metric,
}

# Default metrics (all v2)
DEFAULT_METRICS = ALL_METRICS.copy()

# Models for random sampling
MODELS = [
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    'Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt',
    'SciComma-3.1-8B_y',
    'SciComma-3.1-8B_prompt',
    'scicomma-3.1-dpo',
    'scicomma-3.1-dpo_prompt'
]

DATASET_PATH = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv"
HUMAN_ANSWERS_PATH = f"{PROJECT_DIR}/scripts/judge_alignment/balanced_dataset_v2_human/ask_science_human_metrics.csv"
DISAGREEMENT_DATA_PATH = f"{PROJECT_DIR}/scripts/judge_alignment/metaphor_metric_disagreement_analysis.csv"


def get_eval_model_name(metric_func):
    """Extract the evaluation model name from a metric."""
    try:
        # For GEval metrics with custom model wrapper
        if hasattr(metric_func, 'model') and hasattr(metric_func.model, 'get_model_name'):
            return metric_func.model.get_model_name()
        # For GEval metrics with string model name
        if hasattr(metric_func, 'model') and isinstance(metric_func.model, str):
            return metric_func.model
        # For GEval using native model
        if hasattr(metric_func, 'using_native_model') and metric_func.using_native_model:
            if hasattr(metric_func.model, 'model_name'):
                return metric_func.model.model_name
        return "unknown"
    except Exception:
        return "unknown"


async def evaluate_pair(metric_name, metric_func, question, answer, row_idx, repetition_idx, model_name=None):
    """Evaluates a single pair with a single metric."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer
    )
    
    try:
        await metric_func.a_measure(test_case)
        score = metric_func.score
        reason = metric_func.reason
        result = {
            "question_idx": row_idx,
            "metric": metric_name,
            "repetition": repetition_idx,
            "score": score,
            "explanation": reason,
            "question": question,
            "answer": answer,
            "eval_model": get_eval_model_name(metric_func)
        }
        if model_name:
            result["model"] = model_name
        return result
    except Exception as e:
        print(f"Error evaluating {metric_name} (idx {row_idx}, rep {repetition_idx}): {e}")
        return None


def get_random_pairs(sample_size):
    """Select random question-answer pairs from the evaluation dataset."""
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Selecting {sample_size} random pairs...")
    selected_pairs = []
    
    # Randomly sample indices
    indices = random.sample(range(len(df)), min(sample_size, len(df)))
    
    for idx in indices:
        row = df.iloc[idx]
        question = row['question']
        # Randomly select a model
        model = random.choice(MODELS)
        
        # Check if model column exists
        if model not in df.columns:
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
    
    return selected_pairs


def get_disagreement_pairs():
    """Load question-answer pairs from the disagreement analysis."""
    print(f"Loading disagreement data from {DISAGREEMENT_DATA_PATH}...")
    df = pd.read_csv(DISAGREEMENT_DATA_PATH)
    
    selected_pairs = []
    for _, row in df.iterrows():
        selected_pairs.append({
            "idx": row['index'],
            "question": row['question'],
            "answer": row['answer'],
            "model": None  # No model info for disagreement data
        })
    
    return selected_pairs


def get_human_answer_pairs(sample_size):
    """Select random question-answer pairs from the human answers dataset (ask_science)."""
    print(f"Loading human answers from {HUMAN_ANSWERS_PATH}...")
    df = pd.read_csv(HUMAN_ANSWERS_PATH)
    
    print(f"Selecting {sample_size} random pairs from human answers...")
    selected_pairs = []
    
    # Randomly sample indices
    indices = random.sample(range(len(df)), min(sample_size, len(df)))
    
    for idx in indices:
        row = df.iloc[idx]
        question = row['Question']
        answer = row['Human Answer']
        original_idx = row['Index'] if 'Index' in df.columns else idx
        
        # Handle NaN answers
        if pd.isna(answer):
            print(f"Answer is NaN at row {idx}. Skipping.")
            continue

        selected_pairs.append({
            "idx": original_idx,
            "question": question,
            "answer": answer,
            "model": "human"  # Mark as human answer
        })
    
    return selected_pairs


def get_custom_csv_pairs(csv_path):
    """Load question-answer pairs from a custom CSV file."""
    print(f"Loading pairs from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Try to find question and answer columns
    q_col = None
    a_col = None
    
    for col in ['question', 'Question', 'original_question']:
        if col in df.columns:
            q_col = col
            break
    
    for col in ['answer', 'Answer', 'original_answer', 'Human Answer']:
        if col in df.columns:
            a_col = col
            break
    
    if not q_col or not a_col:
        raise ValueError(f"Could not find question/answer columns. Available: {list(df.columns)}")
    
    print(f"Using columns: question='{q_col}', answer='{a_col}'")
    
    selected_pairs = []
    for idx, row in df.iterrows():
        question = row[q_col]
        answer = row[a_col]
        
        # Handle NaN
        if pd.isna(question) or pd.isna(answer):
            continue
        
        selected_pairs.append({
            "idx": idx,
            "question": question,
            "answer": answer,
            "model": "custom"
        })
    
    print(f"Loaded {len(selected_pairs)} pairs")
    return selected_pairs


async def run_consistency_check(metrics, selected_pairs, repetitions, output_dir):
    """Run the consistency check with given metrics and pairs."""
    print(f"Using {len(selected_pairs)} pairs for consistency check.")
    print(f"Running {len(metrics)} metrics ({repetitions} repetitions each)...")
    
    # Create tasks for all metrics * all pairs * all repetitions
    semaphore = asyncio.Semaphore(20)  # Limit concurrency
    
    async def wrapped_eval(metric_name, metric_func, pair, rep):
        async with semaphore:
            return await evaluate_pair(
                metric_name, metric_func, 
                pair['question'], pair['answer'], pair['idx'], rep,
                model_name=pair.get('model')
            )

    tasks = []
    for metric_name, metric_func in metrics.items():
        for pair in selected_pairs:
            for i in range(repetitions):
                tasks.append(wrapped_eval(metric_name, metric_func, pair, i+1))
    
    # Run with progress bar
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await f
        if res:
            results.append(res)
            
    # Save Intermediate Results
    os.makedirs(output_dir, exist_ok=True)
    intermediate_df = pd.DataFrame(results)
    intermediate_path = os.path.join(output_dir, "intermediate_results.csv")
    intermediate_df.to_csv(intermediate_path, index=False)
    print(f"Intermediate results saved to {intermediate_path}")
    
    # Calculate Statistics
    print("Calculating statistics...")
    stats = []
    
    # Group by Metric, Question (and Model if available)
    group_cols = ['metric', 'question_idx']
    if 'model' in intermediate_df.columns and intermediate_df['model'].notna().any():
        group_cols.append('model')
    
    grouped = intermediate_df.groupby(group_cols)
    
    for group_key, group in grouped:
        scores = group['score'].values
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores, ddof=1) if len(scores) > 1 else 0
        
        # SE = s / sqrt(n)
        se = std_dev / np.sqrt(len(scores)) if len(scores) > 1 else 0
        
        # 95% CI using t-value
        t_values = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093}
        t_value = t_values.get(len(scores), 1.96)
        
        ci_lower = mean_score - (t_value * se)
        ci_upper = mean_score + (t_value * se)
        
        # Binary consistency: how often does the binary decision (>0.5) agree?
        binary_scores = (scores > 0.5).astype(int)
        binary_mode = 1 if binary_scores.mean() > 0.5 else 0
        binary_agreement = (binary_scores == binary_mode).mean()
        
        stat_row = {
            "metric": group_key[0],
            "question_idx": group_key[1],
            "n_samples": len(scores),
            "mean_score": mean_score,
            "std_dev": std_dev,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "binary_agreement": binary_agreement,
            "scores_list": str(list(scores))
        }
        if len(group_cols) > 2:
            stat_row["model"] = group_key[2]
        
        stats.append(stat_row)
        
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(output_dir, "consistency_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    
    # Unified Report (Average Consistency per Metric)
    metric_summary = stats_df.groupby('metric').agg({
        'std_dev': 'mean',
        'se': 'mean',
        'binary_agreement': 'mean'
    }).reset_index()
    metric_summary.columns = ['metric', 'avg_std_dev (inconsistency)', 'avg_se', 'avg_binary_agreement']
    summary_path = os.path.join(output_dir, "metric_consistency_summary.csv")
    metric_summary.to_csv(summary_path, index=False)
    
    print(f"\nStats saved to {stats_path}")
    print(f"Summary saved to {summary_path}")
    print("\nMetric Consistency Summary:")
    print(metric_summary)
    
    # Print detailed per-question stats
    print("\n" + "="*80)
    print("PER-QUESTION CONSISTENCY")
    print("="*80)
    for _, row in stats_df.iterrows():
        model_info = f" (model: {row['model']})" if 'model' in row and pd.notna(row.get('model')) else ""
        print(f"\nQuestion {row['question_idx']}{model_info}:")
        print(f"  Metric: {row['metric']}")
        print(f"  Mean: {row['mean_score']:.3f} ± {row['std_dev']:.3f}")
        print(f"  95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        print(f"  Binary agreement: {row['binary_agreement']*100:.1f}%")
    
    return stats_df, metric_summary


def main():
    parser = argparse.ArgumentParser(description="Test metric consistency/stability")
    parser.add_argument("--metric", type=str, default=None,
                        help=f"Run only a specific metric. Available: {', '.join(ALL_METRICS.keys())}")
    parser.add_argument("--use_disagreements", action="store_true",
                        help="Use disagreement examples instead of random sampling")
    parser.add_argument("--human_answers", action="store_true",
                        help="Sample from human answers (ask_science) instead of model outputs")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to custom CSV file with question/answer columns")
    parser.add_argument("--samples", type=int, default=15,
                        help="Number of random samples (ignored if --use_disagreements)")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of repetitions per sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number suffix for output directory (for sister experiments)")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Select metrics
    if args.metric:
        if args.metric not in ALL_METRICS:
            print(f"Error: Unknown metric '{args.metric}'")
            print(f"Available metrics: {', '.join(ALL_METRICS.keys())}")
            return
        metrics = {args.metric: ALL_METRICS[args.metric]}
        metric_suffix = args.metric
    else:
        metrics = DEFAULT_METRICS
        metric_suffix = "all"
    
    # Get pairs
    if args.csv:
        selected_pairs = get_custom_csv_pairs(args.csv)
        # Use CSV filename as suffix
        csv_name = os.path.basename(args.csv).replace('.csv', '')
        source_suffix = f"csv_{csv_name}"
    elif args.use_disagreements:
        selected_pairs = get_disagreement_pairs()
        source_suffix = "disagreements"
    elif args.human_answers:
        selected_pairs = get_human_answer_pairs(args.samples)
        source_suffix = f"human_{args.samples}_samples"
    else:
        selected_pairs = get_random_pairs(args.samples)
        source_suffix = f"{args.samples}_samples"
    
    # Output directory
    run_suffix = f"_run{args.run}" if args.run else ""
    output_dir = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/consistency_check/{metric_suffix}_{source_suffix}_{args.reps}_tries{run_suffix}/"
    
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK CONFIGURATION")
    print(f"{'='*60}")
    print(f"Metrics: {', '.join(metrics.keys())}")
    data_source = "disagreement examples" if args.use_disagreements else ("human answers (ask_science)" if args.human_answers else "model outputs")
    print(f"Data source: {data_source}")
    print(f"Number of pairs: {len(selected_pairs)}")
    print(f"Repetitions: {args.reps}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run
    asyncio.run(run_consistency_check(metrics, selected_pairs, args.reps, output_dir))
    
    print("\n" + "="*60)
    print("CONSISTENCY CHECK COMPLETE")
    print("="*60)
    
    # Force clean exit to avoid asyncio hanging on unclosed connections
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
