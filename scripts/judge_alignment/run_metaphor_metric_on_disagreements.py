"""
Run metaphor_v2 metric on the 16 metaphor disagreement examples to get scores with reasons.
"""
import os
import sys
import asyncio
import pandas as pd

# Add project paths
sys.path.insert(0, '/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/custom_metrics')
sys.path.insert(0, '/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval')

from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from deepeval.test_case import LLMTestCase
from metrics import metaphor_metric_explicit_v2

# The 16 indices from our metaphor disagreement analysis
DISAGREEMENT_INDICES = [1301, 1404, 984, 1205, 426, 1405, 875, 1281, 292, 570, 470, 1349, 402, 312, 1825, 889]

async def run_metric_with_reason(question: str, answer: str, metric) -> tuple:
    """Run a metric and return score and reason."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )
    
    try:
        await metric.a_measure(test_case)
        return metric.score, getattr(metric, 'reason', None)
    except Exception as e:
        print(f"Error: {e}")
        return None, str(e)

async def main():
    # Load the ORIGINAL unformatted data from ask_science_human_metrics.csv
    df = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/ask_science_human_metrics.csv')
    
    # Also load the human annotations from the formatted dataset
    df_human = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/balanced_30_formatted.csv')
    human_annotations = df_human.set_index('Index')[['metaphor_mattan_yeroushalmi', 'metaphor_nirgrn']].to_dict('index')
    
    # Filter to only the disagreement indices
    df_filtered = df[df['Index'].isin(DISAGREEMENT_INDICES)].copy()
    
    print(f"Found {len(df_filtered)} examples to evaluate")
    print(f"Indices: {df_filtered['Index'].tolist()}")
    
    results = []
    
    for idx, row in df_filtered.iterrows():
        index = row['Index']
        question = row['Question']  # Note: capital Q in original data
        answer = row['Human Answer']  # Note: 'Human Answer' column name
        
        print(f"\nProcessing Index {index}...")
        print(f"Question: {question[:80]}...")
        
        score, reason = await run_metric_with_reason(question, answer, metaphor_metric_explicit_v2)
        
        # Get human annotations if available
        human_data = human_annotations.get(index, {})
        
        results.append({
            'index': index,
            'question': question,
            'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer,
            'metaphor_v2_score': score,
            'metaphor_v2_reason': reason,
            # Include existing human annotations for comparison
            'metaphor_mattan': human_data.get('metaphor_mattan_yeroushalmi', None),
            'metaphor_nir': human_data.get('metaphor_nirgrn', None),
        })
        
        print(f"  Score: {score}")
        print(f"  Reason: {reason[:200] if reason else 'N/A'}...")
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = '/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/metaphor_metric_disagreement_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for _, row in results_df.iterrows():
        print(f"\nIndex {row['index']}:")
        print(f"  LLM Score: {row['metaphor_v2_score']}")
        print(f"  Mattan: {row['metaphor_mattan']}, Nir: {row['metaphor_nir']}")
        print(f"  Reason: {row['metaphor_v2_reason'][:300] if row['metaphor_v2_reason'] else 'N/A'}...")

if __name__ == "__main__":
    asyncio.run(main())

