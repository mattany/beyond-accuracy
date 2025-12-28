"""
Run humor metric with Claude 4 Opus as the judge model on balanced_30_formatted.csv.
Add humor_claude_score and humor_claude_reason columns.
"""

import os
import sys
import asyncio
import pandas as pd

# Add the deep_eval directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Benchmarking/deep_eval")))

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Required by deepeval even if not using OpenAI

from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from custom_metrics.metrics.claude_model import claude_opus_4

GEVAL_RETRIES = 3

# Create humor metric with Claude 4 Opus as the judge
humor_metric_claude = GEval(
    name="Humor Explicit (Claude)",
    evaluation_steps=[
        """1. HUMOR includes explicit jokes AND ironic language.
   
   Look for ANY of the following:
   - Explicit jokes (with or without a formal punchline)
   - Puns or wordplay meant to amuse
   - Ironic language: understatement, tongue-in-cheek remarks, or wry observations
   
   Examples of humor PRESENT:
   - "...killing our own cells, which wouldn't be very wise" (ironic understatement)
   - "Atoms are like tiny LEGO blocks, except you can't step on them at 3 AM" (joke with punchline)
   - "Why did the electron leave the atom? Because it had no potential" (pun)
   - "Evolution doesn't plan ahead – if it did, it would have given us better knees" (ironic observation)""",
        """2. The following are NOT humor:
   - Creative or vivid analogies/metaphors without irony (e.g., "DNA is like a blueprint", "bacteria are like ninjas")
   - Engaging or enthusiastic tone without jokes or irony
   - Personification without irony (e.g., "the virus wants to replicate", "chemicals are villains")
   - Vivid or dramatic descriptions
   - Playful language that lacks actual jokes or ironic statements""",
        """3. The key question: Is there an explicit JOKE or IRONIC statement?
   If yes → return 10.
   If no (even if creative, vivid, or playful) → return 0.
   Do not use intermediate scores."""
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=claude_opus_4
)


async def evaluate_humor(question: str, answer: str) -> tuple[float, str]:
    """Evaluate humor metric with Claude and return (score, reason)."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )
    
    for attempt in range(GEVAL_RETRIES):
        try:
            await humor_metric_claude.a_measure(test_case)
            score = humor_metric_claude.score
            reason = getattr(humor_metric_claude, 'reason', None)
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
    print(f"Running humor metric with Claude 4 Opus as judge...")
    print("-" * 60)
    
    # Initialize the new columns
    df['humor_claude_score'] = None
    df['humor_claude_reason'] = None
    
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
            df.at[idx, 'humor_claude_score'] = score
            df.at[idx, 'humor_claude_reason'] = reason
            print(f"  Score: {score}")
            if reason:
                print(f"  Reason: {reason[:200]}...")
        else:
            print(f"  Failed to get score/reason")
    
    # Save the updated dataset
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved updated dataset to {csv_path}")
    print(f"Added 'humor_claude_score' and 'humor_claude_reason' columns")
    
    # Quick summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    scores = df['humor_claude_score'].dropna()
    if len(scores) > 0:
        binary_scores = (scores > 0.5).astype(int)
        print(f"Positive (>0.5): {binary_scores.sum()}/{len(scores)}")
        print(f"Negative (<=0.5): {(~binary_scores.astype(bool)).sum()}/{len(scores)}")


if __name__ == "__main__":
    asyncio.run(main())

