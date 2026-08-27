import pandas as pd
import itertools
import random
from evaluation.rubrics.settings import PROJECT_ROOT

# Load the original dataset
df = pd.read_csv(PROJECT_ROOT / "evaluation/model_outputs/main/all_models_joined.csv")

# Identify model columns dynamically
non_model_columns = {'index', 'full_dataset_index', 'question'}
model_columns = [col for col in df.columns if col not in non_model_columns]

# Prepare the outputs
comparison_rows = []
pair_metadata_rows = []
random.seed(42)
limit = 20
# Loop through each row in the dataset
for i, row in df.iterrows():
    if i >= limit:
        break
    question = row['question']
    question_index = row['index']

    # Shuffle models for this row
    shuffled_models = model_columns.copy()
    random.shuffle(shuffled_models)

    # Generate all unique unordered model pairs
    model_pairs = list(itertools.combinations(shuffled_models, 2))
    random.shuffle(model_pairs)
    for loop_index, (model_a, model_b) in enumerate(model_pairs):
        answer_a, answer_b = row[model_a], row[model_b]

        # Randomly flip order for A/B comparison
        if random.random() < 0.5:
            answer_a, answer_b = answer_b, answer_a
            model_a, model_b = model_b, model_a

        # Build task ID
        task_id = f"{question_index}_{loop_index}"

        # Append to outputs
        comparison_rows.append({
            "task_id": task_id,
            "question_index": question_index,
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b
        })

        pair_metadata_rows.append({
            "task_id": task_id,
            "model_a": model_a,
            "model_b": model_b
        })
random.shuffle(comparison_rows)
# Convert to DataFrames
comparison_df = pd.DataFrame(comparison_rows)
metadata_df = pd.DataFrame(pair_metadata_rows)

# Save to CSV
comparison_df.to_csv(PROJECT_ROOT / "evaluation/model_outputs/main/comparison_tasks.csv", index=False)
metadata_df.to_csv(PROJECT_ROOT / "evaluation/model_outputs/main/model_pairs_metadata.csv", index=False)
