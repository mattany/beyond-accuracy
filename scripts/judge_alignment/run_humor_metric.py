"""
Generic runner for humor metrics on balanced_30_formatted.csv.
Usage: python run_humor_metric.py --version v4
"""

import os
import sys
import asyncio
import argparse
import pandas as pd

# Add the deep_eval directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Benchmarking/deep_eval")))

from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from deepeval.test_case import LLMTestCase

GEVAL_RETRIES = 3
MAX_CONCURRENT = 5  # Limit concurrent API calls


def create_metric(version: str):
    """Create a fresh metric instance for a given version."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    from custom_metrics.metrics.constants import g_eval_default_params
    from custom_metrics.metrics import (
        humor_metric_explicit_v2,
        humor_metric_explicit_v3,
        humor_metric_explicit_v4,
    )
    
    # Get the evaluation steps from the original metric
    templates = {
        "v2": humor_metric_explicit_v2,
        "v3": humor_metric_explicit_v3,
        "v4": humor_metric_explicit_v4,
    }
    
    if version not in templates:
        raise ValueError(f"Unknown version: {version}. Available: {list(templates.keys())}")
    
    template = templates[version]
    
    # Create a fresh instance with the same config
    return GEval(
        name=template.name,
        evaluation_steps=template.evaluation_steps,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        **g_eval_default_params
    )


async def evaluate_single(
    idx: int,
    original_index: int,
    question: str,
    answer: str,
    version: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, float, str]:
    """Evaluate a single row and return (idx, score, reason)."""
    async with semaphore:
        # Create fresh metric instance for this evaluation
        metric = create_metric(version)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )
        
        for attempt in range(GEVAL_RETRIES):
            try:
                await metric.a_measure(test_case)
                score = metric.score
                reason = getattr(metric, 'reason', None)
                print(f"  Row {idx+1}/30 (Index {original_index}): Score={score}")
                return idx, score, reason
            except ValueError as e:
                print(f"  Row {idx+1} Attempt {attempt + 1}: Invalid JSON, retrying...")
                continue
            except Exception as e:
                print(f"  Row {idx+1} Attempt {attempt + 1}: Error - {e}")
                continue
        
        print(f"  Row {idx+1} (Index {original_index}): FAILED")
        return idx, None, None


async def run_evaluation(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Run evaluation on all rows using async gather."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    tasks = [
        evaluate_single(
            idx=idx,
            original_index=row['Index'],
            question=row['question'],
            answer=row['answer'],
            version=version,
            semaphore=semaphore,
        )
        for idx, row in df.iterrows()
    ]
    
    print(f"Running {len(tasks)} evaluations with max {MAX_CONCURRENT} concurrent...")
    results = await asyncio.gather(*tasks)
    
    # Apply results to dataframe
    score_col = f'humor_{version}_score'
    reason_col = f'humor_{version}_reason'
    df[score_col] = None
    df[reason_col] = None
    
    for idx, score, reason in results:
        df.at[idx, score_col] = score
        df.at[idx, reason_col] = reason
    
    return df


def print_comparison(df: pd.DataFrame, version: str):
    """Print comparison of all humor metrics."""
    print(f"\n{'=' * 60}")
    print("COMPARISON: All Humor Metrics vs Human Consensus")
    print(f"{'=' * 60}")
    
    df['humor_human_consensus'] = ((df['humor_mattan_yeroushalmi'] + df['humor_nirgrn']) / 2).round()
    
    metric_cols = [
        ('humor_v2_score', 'GPT-4o v2'),
        ('humor_v3_score', 'GPT-4o v3'),
        ('humor_v4_score', 'GPT-4o v4'),
        ('humor_claude_score', 'Claude 4 Opus'),
    ]
    
    for col, name in metric_cols:
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                binary = (scores > 0.5).astype(int)
                agreement = (binary == df.loc[scores.index, 'humor_human_consensus']).sum()
                marker = " ← NEW" if version in col else ""
                print(f"{name}: {agreement}/{len(scores)} ({100*agreement/len(scores):.1f}%){marker}")


async def main_async(version: str):
    csv_path = "/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/balanced_30_formatted.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Running humor_{version} metric...")
    print("-" * 60)
    
    df = await run_evaluation(df, version)
    
    # Save
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved to {csv_path}")
    print(f"Added 'humor_{version}_score' and 'humor_{version}_reason' columns")
    
    # Print comparison
    print_comparison(df, version)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run humor metric evaluation")
    parser.add_argument("--version", "-v", default="v4", help="Metric version (v2, v3, v4)")
    args = parser.parse_args()
    
    # Run with proper event loop handling
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_async(args.version))
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

