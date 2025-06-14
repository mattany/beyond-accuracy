import pandas as pd
import numpy as np
import os
import random

def get_model_names(run_dir):
    """Get list of model names from model.txt file."""
    with open(os.path.join(run_dir, "model.txt"), "r") as f:
        return f.read().strip().split(",")

def load_metric_data(run_dir, metric_name):
    """Load a metric's data from its CSV file and get all model scores."""
    file_path = os.path.join(run_dir, f"{metric_name}.csv")
    df = pd.read_csv(file_path)
    # Extract score columns for all models
    score_cols = [col for col in df.columns if col.endswith('__score')]
    return pd.DataFrame({model.replace('__score', ''): df[model] for model in score_cols})

def binarize_metric(scores, threshold=0.5):
    """Convert metric scores to binary values based on threshold."""
    return (scores >= threshold).astype(int)

def create_side_by_side_dataset(run_dir, eval_dataset_path, output_path, threshold=0.5, num_questions=30, random_seed=42):
    """
    Create a dataset comparing base model answers with finetuned (SciComma) model answers.
    
    Args:
        run_dir: Directory containing metric CSV files
        eval_dataset_path: Path to the evaluation dataset
        output_path: Where to save the output CSV
        threshold: Threshold for binarizing metrics (default 0.5)
        num_questions: Number of questions to sample (default 30)
        random_seed: Seed for reproducibility (default 42)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load evaluation dataset
    eval_df = pd.read_csv(eval_dataset_path)
    
    # Load metrics
    metrics = ['humor_explicit', 'metaphor_explicit', 'analogy_explicit', 'connection_to_everyday_life']
    metric_data = {}
    
    # Get all model scores for each metric
    for metric in metrics:
        scores_df = load_metric_data(run_dir, metric)
        metric_data[metric] = {
            'raw_scores': {model: scores_df[model] for model in scores_df.columns},
            'binary_scores': {model: binarize_metric(scores_df[model], threshold) for model in scores_df.columns}
        }
    
    # Define model pairs
    model_pairs = [
        ('llama-2-7b', 'SciComma-2-7b'),
        ('llama3.1-instruct', 'SciComma-3.1-8B'),
        ('llama-3.3-70b', 'SciComma-3.3-70B')
    ]
    
    # Find all valid pairs (where grades differ)
    valid_pairs = []
    for idx in range(len(eval_df)):
        for model_1, model_2 in model_pairs:
            # Calculate grades for both models
            grade_1 = sum(metric_data[metric]['binary_scores'][model_1][idx] for metric in metrics)
            grade_2 = sum(metric_data[metric]['binary_scores'][model_2][idx] for metric in metrics)
            
            # Only include pairs where grades differ
            if grade_1 != grade_2:
                valid_pairs.append({
                    'question_id': idx,
                    'model_1': model_1,
                    'model_2': model_2,
                    'grade_1': grade_1,
                    'grade_2': grade_2
                })
    
    if len(valid_pairs) < num_questions:
        print(f"Warning: Only found {len(valid_pairs)} valid pairs with different grades. Using all of them.")
        selected_pairs = valid_pairs
    else:
        print(f"Found {len(valid_pairs)} valid pairs with different grades. Sampling {num_questions} pairs.")
        selected_pairs = random.sample(valid_pairs, num_questions)
    
    # Create rows for the selected pairs
    rows = []
    for pair in selected_pairs:
        idx = pair['question_id']
        model_1 = pair['model_1']
        model_2 = pair['model_2']
        if random.choice([True, False]):
            model_1, model_2 = model_2, model_1
        row = {
            'question_id': idx,
            'question': eval_df.iloc[idx]['question'],
            'answer_a': eval_df.iloc[idx][model_1],
            'answer_b': eval_df.iloc[idx][model_2],
            'model_a': model_1,
            'model_b': model_2,
            'grade_a': pair['grade_1'],
            'grade_b': pair['grade_2']
        }
        
        # Add binary scores
        for metric in metrics:
            row[f"{metric}_a"] = metric_data[metric]['binary_scores'][model_1][idx]
        for metric in metrics:
            row[f"{metric}_b"] = metric_data[metric]['binary_scores'][model_2][idx]
        for metric in metrics:
            row[f"{metric}_score_a"] = metric_data[metric]['raw_scores'][model_1][idx]
        for metric in metrics:
            row[f"{metric}_score_b"] = metric_data[metric]['raw_scores'][model_2][idx]
        
        rows.append(row)
    
    # Create final dataframe and shuffle
    final_df = pd.DataFrame(rows).sample(frac=1, random_state=random_seed)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)

def create_boolean_dataset(run_dir, eval_dataset_path, output_path, threshold=0.5, samples_per_group=15, random_seed=42):
    """
    Create a dataset with binarized metrics and balanced sampling.
    
    Args:
        run_dir: Directory containing metric CSV files
        eval_dataset_path: Path to the evaluation dataset
        output_path: Where to save the output CSV
        threshold: Threshold for binarizing metrics (default 0.5)
        samples_per_group: Number of samples to select from each group
        random_seed: Seed for reproducibility (default 42)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load evaluation dataset
    eval_df = pd.read_csv(eval_dataset_path)
    
    # Load metrics
    metrics = ['humor_explicit', 'metaphor_explicit', 'analogy_explicit', 'connection_to_everyday_life']
    metric_data = {}
    
    # Get all model scores for each metric
    for metric in metrics:
        scores_df = load_metric_data(run_dir, metric)
        metric_data[metric] = {
            'raw_scores': {model: scores_df[model] for model in scores_df.columns},
            'binary_scores': {model: binarize_metric(scores_df[model], threshold) for model in scores_df.columns}
        }
    
    # Create rows for all question-model combinations
    rows = []
    models = scores_df.columns  # Using models from the last loaded metric
    
    for idx in range(len(eval_df)):
        for model in models:
            row = {
                'question_id': idx,
                'question': eval_df.iloc[idx]['question'],
                'answer': eval_df.iloc[idx][model],
                'model': model
            }
            # Add both raw and binary metric values
            for metric in metrics:
                row[metric] = metric_data[metric]['binary_scores'][model][idx]
            for metric in metrics:
                # Add raw score
                row[f"{metric}_score"] = metric_data[metric]['raw_scores'][model][idx]
                # Add binary score
            rows.append(row)
    
    # Create combined dataframe
    combined_df = pd.DataFrame(rows)
    
    # Calculate sum of binary metrics for each row
    binary_metrics = [m for m in metrics]  # Using original metric names for binary columns
    combined_df['metric_sum'] = combined_df[binary_metrics].sum(axis=1)
    
    # Calculate median of metric sums
    median_sum = combined_df['metric_sum'].median()
    
    # Split into high and low groups
    high_group = combined_df[combined_df['metric_sum'] > median_sum]
    low_group = combined_df[combined_df['metric_sum'] <= median_sum]
    
    # Randomly sample from each group
    sampled_high = high_group.sample(n=samples_per_group, random_state=42)
    sampled_low = low_group.sample(n=samples_per_group, random_state=42)
    
    # Combine samples
    final_df = pd.concat([sampled_high, sampled_low]).sample(frac=1, random_state=42)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    run_dir = "Benchmarking/deep_eval/data/run_5"
    eval_dataset_path = "Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv"
    output_path_1 = "Benchmarking/deep_eval/data/boolean_dataset_1.csv"
    output_path_2 = "Benchmarking/deep_eval/data/side_by_side_dataset.csv"
    
    # Set random seed for reproducibility
    random_seed = 42
    
    # Create both datasets
    create_boolean_dataset(run_dir, eval_dataset_path, output_path_1, random_seed=random_seed)
    create_side_by_side_dataset(run_dir, eval_dataset_path, output_path_2, random_seed=random_seed)
