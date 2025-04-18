import pandas as pd
import itertools
from config import PROJECT_DIR, RUN_NUMBER


def generate_pairwise_comparisons(scores_path, answers_path, output_path, include_metadata=True, top_k=None):
    # Load aggregate scores
    scores_df = pd.read_csv(scores_path, index_col="Index")

    # Load model answers
    answers_df = pd.read_csv(answers_path, index_col="index")

    # Extract question text and model names
    prompt_column = "question"
    model_columns = [col for col in answers_df.columns if col not in [prompt_column, 'full_dataset_index']]

    rows = []

    for idx, row in answers_df.iterrows():
        prompt = row[prompt_column]
        question_id = idx
        top_k_models = None
        if top_k:
            top_k_models = scores_df.iloc[question_id].sort_values(ascending=False)[:top_k].index

        for model1, model2 in itertools.combinations(model_columns, 2):
            if model1 not in top_k_models and model2 not in top_k_models:
                continue
            score1 = scores_df.at[question_id, model1]
            score2 = scores_df.at[question_id, model2]

            if pd.isna(score1) or pd.isna(score2):
                continue  # skip pairs with missing scores

            if score1 == score2:
                continue  # skip ties (or you could include both directions if needed)

            # Choose the higher scoring model as chosen
            if score1 > score2:
                chosen_model, rejected_model = model1, model2
                chosen_score, rejected_score = score1, score2
            else:
                chosen_model, rejected_model = model2, model1
                chosen_score, rejected_score = score2, score1
            row_data = {
                "prompt": prompt,
                "chosen": row[chosen_model],
                "rejected": row[rejected_model],
            }
            if include_metadata:
                row_data = {
                    **row_data,
                    "chosen_model": chosen_model,
                    "rejected_model": rejected_model,
                    "question_id": question_id,
                    "chosen_model_score": chosen_score,
                    "rejected_model_score": rejected_score,
                }
            rows.append(row_data)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"Pairwise comparison dataset written to {output_path}")


if __name__ == "__main__":
    generate_pairwise_comparisons(
        scores_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{RUN_NUMBER}/aggregations/aggregate_scores.csv",
        answers_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv",
        output_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{RUN_NUMBER}/aggregations/preference_dataset_top_3.csv",
        top_k=3
    )
