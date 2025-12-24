"""
Run humor_v3 metric on rows where at least one annotator marked humor as true.
Add humor_v3_score and humor_v3_reason columns.

v3 is a balanced version aligned with Baram-Tsabari (2012) definition:
- Less strict than v2 (which required "obvious" and "unmistakable" humor)
- More strict than v1 (which allowed score of 5 when unsure)
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
from custom_metrics.metrics import humor_metric_explicit_v2

GEVAL_RETRIES = 3

async def evaluate_humor(question: str, answer: str) -> tuple[float, str]:
    """Evaluate humor metric (v3) and return (score, reason)."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )
    
    for attempt in range(GEVAL_RETRIES):
        try:
            await humor_metric_explicit_v2.a_measure(test_case)
            score = humor_metric_explicit_v2.score
            reason = getattr(humor_metric_explicit_v2, 'reason', None)
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
    csv_path = "/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2/balanced_dataset_v2.csv"
    df = pd.read_csv(csv_path)
    
    # Find rows where at least one annotator marked humor as true
    annotator_cols = ['humor_mattan_yeroushalmi', 'humor_maximbr', 'humor_nirgrn']
    humor_mask = (df[annotator_cols].sum(axis=1) >= 1)
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with at least 1 humor annotation: {humor_mask.sum()}")
    
    # Initialize the new columns with None
    df['humor_v3_score'] = None
    df['humor_v3_reason'] = None
    
    # Get the rows to evaluate
    rows_to_evaluate = df[humor_mask].index.tolist()
    
    print(f"\nEvaluating {len(rows_to_evaluate)} rows...")
    print("-" * 60)
    
    for idx in rows_to_evaluate:
        row = df.loc[idx]
        question = row['question']
        answer = row['answer']
        model = row['model']
        
        # Show which annotators marked it
        annotators = [col.replace('humor_', '') for col in annotator_cols if row[col] == 1]
        
        print(f"\nRow {idx}: question_id={row['question_id']}, model={model}")
        print(f"  Annotators who marked humor: {annotators}")
        print(f"  Question: {question[:80]}...")
        
        score, reason = await evaluate_humor(question, answer)
        
        if score is not None:
            df.at[idx, 'humor_v3_score'] = score
            df.at[idx, 'humor_v3_reason'] = reason
            print(f"  Score: {score}")
            print(f"  Reason: {reason[:200] if reason else 'None'}...")
        else:
            print(f"  Failed to get score/reason")
    
    # Save the updated dataset
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved updated dataset to {csv_path}")
    print(f"Added 'humor_v3_reason' column")


if __name__ == "__main__":
    asyncio.run(main())

