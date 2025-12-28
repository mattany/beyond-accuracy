"""
Run humor_v3 metric (calibrated version) on balanced_30_formatted.csv.
Add humor_v3_score and humor_v3_reason columns.
"""

import os
import sys
import asyncio
import pandas as pd

# Add the deep_eval directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Benchmarking/deep_eval")))

from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from deepeval.test_case import LLMTestCase
from custom_metrics.metrics import humor_metric_explicit_v3

GEVAL_RETRIES = 3


async def evaluate_humor(question: str, answer: str) -> tuple[float, str]:
    """Evaluate humor metric v3 and return (score, reason)."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )
    
    for attempt in range(GEVAL_RETRIES):
        try:
            await humor_metric_explicit_v3.a_measure(test_case)
            score = humor_metric_explicit_v3.score
            reason = getattr(humor_metric_explicit_v3, 'reason', None)
            return score, reason
        except ValueError as e:
            print(f"Attempt {attempt + 1}: Invalid JSON, retrying... ({e})")
            continue
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
            continue
    
    return None, None


async def main():
    # Load the dataset
    csv_path = "/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/balanced_30_formatted.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Running humor_v3 metric (calibrated version with GPT-4o)...")
    print("-" * 60)
    
    # Initialize the new columns
    df['humor_v3_score'] = None
    df['humor_v3_reason'] = None
    
    # Process all rows
    for idx in range(len(df)):
        row = df.iloc[idx]
        question = row['question']
        answer = row['answer']
        original_index = row['Index']
        
        print(f"\nRow {idx+1}/30 (Index {original_index})")
        print(f"  Question: {question[:80]}...")
        
        score, reason = await evaluate_humor(question, answer)
        
        if score is not None:
            df.at[idx, 'humor_v3_score'] = score
            df.at[idx, 'humor_v3_reason'] = reason
            print(f"  Score: {score}")
            if reason:
                print(f"  Reason: {reason[:200]}...")
        else:
            print(f"  Failed to get score/reason")
    
    # Save the updated dataset
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved updated dataset to {csv_path}")
    print(f"Added 'humor_v3_score' and 'humor_v3_reason' columns")
    
    # Quick summary comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON: v2 vs v3 vs Claude vs Human")
    print(f"{'=' * 60}")
    
    df['humor_human_consensus'] = ((df['humor_mattan_yeroushalmi'] + df['humor_nirgrn']) / 2).round()
    
    for col, name in [('humor_v2_score', 'GPT-4o v2'), 
                       ('humor_v3_score', 'GPT-4o v3'),
                       ('humor_claude_score', 'Claude 4 Opus')]:
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                binary = (scores > 0.5).astype(int)
                agreement = (binary == df.loc[scores.index, 'humor_human_consensus']).sum()
                print(f"{name}: {agreement}/{len(scores)} agreement with human consensus ({100*agreement/len(scores):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())

