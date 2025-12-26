"""
Run metaphor_v2 metric on the 16 metaphor disagreement examples to get scores with reasons.
Uses parallel async processing with asyncio.gather.

Usage:
    python run_metaphor_metric_on_disagreements.py          # Run metrics and analyze
    python run_metaphor_metric_on_disagreements.py --no_run # Just analyze existing CSV
"""
import os
import sys
import argparse
import asyncio
import logging
import pandas as pd
from tqdm import tqdm

# Add project paths
sys.path.insert(0, '/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/custom_metrics')
sys.path.insert(0, '/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval')

# Constants
THRESHOLD = 0.5
GEVAL_RETRIES = 3
OUTPUT_PATH = '/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/metaphor_metric_disagreement_analysis.csv'

# The 16 indices from our metaphor disagreement analysis
DISAGREEMENT_INDICES = [1301, 1404, 984, 1205, 426, 1405, 875, 1281, 292, 570, 470, 1349, 402, 312, 1825, 889]


def get_disagreement_type(llm_binary, mattan, nir):
    """Categorize the type of disagreement between annotators."""
    if llm_binary == mattan and llm_binary != nir:
        return "LLM+Mattan vs Nir"
    elif llm_binary == nir and llm_binary != mattan:
        return "LLM+Nir vs Mattan"
    elif mattan == nir and llm_binary != mattan:
        return "Mattan+Nir vs LLM"
    else:
        # Edge case: all three agree (shouldn't happen in disagreement set)
        return "No disagreement"


def add_analysis_columns(df):
    """Add llm_binary and disagreement_type columns to the dataframe."""
    # Add llm_binary column (1 if score > threshold, else 0)
    df['llm_binary'] = (df['metaphor_v2_score'] > THRESHOLD).astype(int)
    
    # Add disagreement_type column
    df['disagreement_type'] = df.apply(
        lambda row: get_disagreement_type(
            row['llm_binary'],
            row['metaphor_mattan'],
            row['metaphor_nir']
        ),
        axis=1
    )
    
    return df


def print_analysis(df):
    """Print detailed analysis of disagreements."""
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS")
    print("="*80)
    
    # Summary by disagreement type
    print("\n--- Disagreement Type Summary ---")
    type_counts = df['disagreement_type'].value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count}")
    
    # Detailed per-row analysis
    print("\n--- Per-Example Details ---")
    for _, row in df.iterrows():
        print(f"\nIndex {row['index']}:")
        print(f"  Question: {row['question'][:60]}...")
        print(f"  LLM Score: {row['metaphor_v2_score']:.3f} → Binary: {row['llm_binary']}")
        print(f"  Mattan: {row['metaphor_mattan']}, Nir: {row['metaphor_nir']}")
        print(f"  Disagreement Type: {row['disagreement_type']}")
        reason = row['metaphor_v2_reason']
        print(f"  LLM Reason: {reason[:200] if reason else 'N/A'}...")
    
    # Analysis insights
    print("\n" + "="*80)
    print("INSIGHTS")
    print("="*80)
    
    # Where LLM agrees with each human
    llm_mattan_agree = (df['llm_binary'] == df['metaphor_mattan']).sum()
    llm_nir_agree = (df['llm_binary'] == df['metaphor_nir']).sum()
    mattan_nir_agree = (df['metaphor_mattan'] == df['metaphor_nir']).sum()
    
    print(f"\nAgreement rates (out of {len(df)} examples):")
    print(f"  LLM-Mattan agreement: {llm_mattan_agree}/{len(df)} ({100*llm_mattan_agree/len(df):.1f}%)")
    print(f"  LLM-Nir agreement: {llm_nir_agree}/{len(df)} ({100*llm_nir_agree/len(df):.1f}%)")
    print(f"  Mattan-Nir agreement: {mattan_nir_agree}/{len(df)} ({100*mattan_nir_agree/len(df):.1f}%)")
    
    # LLM tendency
    llm_pos = df['llm_binary'].sum()
    mattan_pos = df['metaphor_mattan'].sum()
    nir_pos = df['metaphor_nir'].sum()
    
    print(f"\nPositive labels (metaphor present):")
    print(f"  LLM: {llm_pos}/{len(df)} ({100*llm_pos/len(df):.1f}%)")
    print(f"  Mattan: {mattan_pos}/{len(df)} ({100*mattan_pos/len(df):.1f}%)")
    print(f"  Nir: {nir_pos}/{len(df)} ({100*nir_pos/len(df):.1f}%)")


async def evaluate_row(index, question, answer, human_data, metric, scores, reasons, semaphore, pbar=None):
    """Evaluate a single row with the metric."""
    async with semaphore:
        from deepeval.test_case import LLMTestCase
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )
        
        success = False
        for i in range(GEVAL_RETRIES):
            try:
                await metric.a_measure(test_case)
                success = True
                break
            except ValueError:
                print(f"Index {index}: Retry {i+1}/{GEVAL_RETRIES} - Invalid JSON")
                continue
            except Exception as e:
                print(f"Index {index}: Error - {e}")
                break
        
        if success:
            scores[index] = {
                'index': index,
                'question': question,
                'answer': answer,  # Full answer
                'metaphor_v2_score': metric.score,
                'metaphor_v2_reason': getattr(metric, 'reason', None),
                'metaphor_mattan': human_data.get('metaphor_mattan_yeroushalmi', None),
                'metaphor_nir': human_data.get('metaphor_nirgrn', None),
            }
            reasons[index] = getattr(metric, 'reason', None)
        else:
            scores[index] = {
                'index': index,
                'question': question,
                'answer': answer,  # Full answer
                'metaphor_v2_score': None,
                'metaphor_v2_reason': 'EVALUATION FAILED',
                'metaphor_mattan': human_data.get('metaphor_mattan_yeroushalmi', None),
                'metaphor_nir': human_data.get('metaphor_nirgrn', None),
            }
            reasons[index] = None
            print(f"Warning: Index {index} failed")
        
        if pbar:
            pbar.update(1)


async def run_metrics():
    """Run metrics on all disagreement examples."""
    from config import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Suppress DEBUG logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    from metrics import metaphor_metric_explicit_v2
    
    # Load the ORIGINAL unformatted data from ask_science_human_metrics.csv
    df = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/ask_science_human_metrics.csv')
    
    # Also load the human annotations from the formatted dataset
    df_human = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/balanced_30_formatted.csv')
    human_annotations = df_human.set_index('Index')[['metaphor_mattan_yeroushalmi', 'metaphor_nirgrn']].to_dict('index')
    
    # Filter to only the disagreement indices
    df_filtered = df[df['Index'].isin(DISAGREEMENT_INDICES)].copy()
    
    print(f"Found {len(df_filtered)} examples to evaluate")
    print(f"Indices: {df_filtered['Index'].tolist()}")
    
    # Setup for parallel processing
    semaphore = asyncio.Semaphore(40)
    scores = {}
    reasons = {}
    
    pbar = tqdm(total=len(df_filtered), desc="Evaluating metaphor metric")
    
    # Create tasks for parallel execution
    tasks = [
        evaluate_row(
            index=row['Index'],
            question=row['Question'],
            answer=row['Human Answer'],
            human_data=human_annotations.get(row['Index'], {}),
            metric=metaphor_metric_explicit_v2,
            scores=scores,
            reasons=reasons,
            semaphore=semaphore,
            pbar=pbar
        )
        for _, row in df_filtered.iterrows()
    ]
    
    # Run all tasks in parallel
    await asyncio.gather(*tasks)
    pbar.close()
    
    # Convert scores dict to ordered list based on DISAGREEMENT_INDICES
    results = [scores[idx] for idx in DISAGREEMENT_INDICES if idx in scores]
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Add analysis columns
    results_df = add_analysis_columns(results_df)
    
    # Reorder columns for better readability
    column_order = ['index', 'question', 'answer', 'metaphor_v2_score', 'llm_binary',
                    'metaphor_mattan', 'metaphor_nir', 'disagreement_type', 'metaphor_v2_reason']
    results_df = results_df[column_order]
    
    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
    
    return results_df


def analyze_existing():
    """Load and analyze existing CSV without running metrics."""
    print(f"Loading existing results from {OUTPUT_PATH}")
    df = pd.read_csv(OUTPUT_PATH)
    
    # If we only have answer_preview, load full answers from source
    if 'answer_preview' in df.columns and 'answer' not in df.columns:
        print("Loading full answers from source data...")
        source_df = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/scripts/judge_alignment/balanced_dataset_v2_human/ask_science_human_metrics.csv')
        # Create mapping from Index to full answer
        answer_map = source_df.set_index('Index')['Human Answer'].to_dict()
        df['answer'] = df['index'].map(answer_map)
        df = df.drop(columns=['answer_preview'])
    
    # Add/update analysis columns
    df = add_analysis_columns(df)
    
    # Reorder columns for better readability
    column_order = ['index', 'question', 'answer', 'metaphor_v2_score', 'llm_binary',
                    'metaphor_mattan', 'metaphor_nir', 'disagreement_type', 'metaphor_v2_reason']
    df = df[[col for col in column_order if col in df.columns]]
    
    # Save updated results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Updated results saved to {OUTPUT_PATH}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run metaphor metric on disagreement examples")
    parser.add_argument("--no_run", action="store_true", 
                        help="Skip running metrics, just analyze existing CSV")
    args = parser.parse_args()
    
    if args.no_run:
        print("Running in analysis-only mode (--no_run)")
        results_df = analyze_existing()
    else:
        print("Running metrics and analysis")
        results_df = asyncio.run(run_metrics())
    
    # Print analysis
    print_analysis(results_df)


if __name__ == "__main__":
    main()
