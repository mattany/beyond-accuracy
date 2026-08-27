"""
Script to re-run readability metrics using textstat library.
This updates all readability scores (flesch_kincaid, flesch_reading_ease, dale_chall, ari)
for the specified run numbers.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import scireadability

# Add parents to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))  # deep_eval/
sys.path.insert(0, str(script_dir))  # custom_metrics/

from evaluation.rubrics.settings import PROJECT_ROOT, result_directory


# Readability metric functions using scireadability directly
# scireadability is optimized for scientific text with better syllable counting
READABILITY_METRICS = {
    'flesch_kincaid': lambda text: scireadability.flesch_kincaid_grade(text),
    'flesch_reading_ease': lambda text: scireadability.flesch_reading_ease(text) / 100,  # Normalize to 0-1
    'dale_chall': lambda text: scireadability.dale_chall_readability_score(text),
    'ari': lambda text: scireadability.automated_readability_index(text),
}


def rerun_readability_for_run(run_number: int, models: list = None):
    """
    Re-run all readability metrics for a given run number.
    
    Args:
        run_number: The run number to process
        models: List of model columns to process. If None, auto-detects from existing data.
    """
    run_dir = result_directory(run_number)
    
    # Get the first metric file to detect models
    first_metric_file = run_dir / "flesch_kincaid.csv"
    if not first_metric_file.exists():
        print(f"Error: {first_metric_file} not found")
        return
    
    existing_df = pd.read_csv(first_metric_file)
    
    # Detect model columns
    if models is None:
        models = [c.replace('__score', '') for c in existing_df.columns if c.endswith('__score')]
    
    print(f"\n{'='*60}")
    print(f"Re-running readability metrics for run {run_number}")
    print(f"Models: {models}")
    print(f"{'='*60}\n")
    
    # Load the evaluation dataset to get the actual text
    if run_number == 10:
        # Experiment B uses experiment_b_eval_dataset
        eval_df = pd.read_csv(
            PROJECT_ROOT / "human_study/preferences/data/experiment_b_eval_dataset.csv"
        )
    else:
        # Standard runs use corrected_evaluation_dataset
        eval_df = pd.read_csv(PROJECT_ROOT / "evaluation/model_outputs/main/all_models_joined.csv")
    
    for metric_name, metric_func in READABILITY_METRICS.items():
        print(f"\nProcessing {metric_name}...")
        output_path = run_dir / f"{metric_name}.csv"
        
        # Create new DataFrame for results with question for alignment
        output_df = pd.DataFrame()
        if 'question' in eval_df.columns:
            output_df['question'] = eval_df['question'].values
        
        for model in models:
            print(f"  Evaluating {model}...")
            scores = []
            
            # The text column is always the model name
            text_column = model
            
            for idx, row in eval_df.iterrows():
                try:
                    if pd.isna(row[text_column]):
                        scores.append(None)
                    else:
                        text = str(row[text_column])
                        if text.strip() == '':
                            scores.append(None)
                        else:
                            # Use textstat directly
                            score = metric_func(text)
                            scores.append(score)
                except Exception as e:
                    print(f"    Row {idx}: Error - {e}")
                    scores.append(None)
            
            output_df[f"{model}__score"] = scores
            
            # Count non-null
            non_null = sum(1 for s in scores if s is not None)
            print(f"    Completed: {non_null}/{len(scores)} successful")
        
        output_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
    
    print(f"\n✓ Completed run {run_number}")


def main():
    parser = argparse.ArgumentParser(description='Re-run readability metrics with textstat')
    parser.add_argument(
        '--run',
        type=int,
        nargs='+',
        default=[9, 10],
        help='Run numbers to process (default: 9 10)'
    )
    args = parser.parse_args()
    
    for run_num in args.run:
        rerun_readability_for_run(run_num)


if __name__ == "__main__":
    main()

