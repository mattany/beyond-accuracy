import random
from datasets import Dataset, DatasetDict

import pandas as pd
import itertools
from config import PROJECT_DIR, RUN_NUMBER


def generate_pairwise_comparisons(
    scores_path,
    answers_path,
    output_path,
    include_metadata=True,
    top_k=None,
    min_diff=0,
    shuffle=True,
    push_to_hub=False,
):
    """

    :param scores_path:
    :param answers_path:
    :param output_path:
    :param include_metadata: include the scores and model names
    in addition to the "prompt","chosen" & "rejected" columns
    :param top_k: the amount of models to take "chosen" answers from
    :param min_diff: minimum standard deviations difference in score between
    two answers in order for them to be included in the dataset
    :param shuffle: shuffle the rows before writing the dataframe
    :return:
    """
    # Load aggregate scores
    scores_df = pd.read_csv(scores_path, index_col="Index")
    std_dev = scores_df.std(axis=1).mean(axis=0)

    # Load model answers
    answers_df = pd.read_csv(answers_path, index_col="index")
    # Extract question text and model names
    prompt_column = "question"
    model_columns = [
        col
        for col in answers_df.columns
        if col not in [prompt_column, "full_dataset_index"]
    ]

    rows = []

    for idx, row in answers_df.iterrows():
        prompt = row[prompt_column]
        question_id = idx
        if top_k:
            top_k_models = (
                scores_df.iloc[question_id].sort_values(ascending=False)[:top_k].index
            )
        else:
            top_k_models = (
                scores_df.iloc[question_id].sort_values(ascending=False).index
            )

        for model1, model2 in itertools.combinations(model_columns, 2):
            if model1 not in top_k_models and model2 not in top_k_models:
                continue
            score1 = scores_df.at[question_id, model1]
            score2 = scores_df.at[question_id, model2]

            if pd.isna(score1) or pd.isna(score2):
                continue  # skip pairs with missing scores

            if abs(score1 - score2) < min_diff * std_dev:
                continue  # skip ties (or you could include both directions if needed)
            # Choose the higher scoring model as chosen
            score_model = {score1: model1, score2: model2}
            chosen_score, rejected_score = max(score1, score2), min(score1, score2)
            chosen_model, rejected_model = (
                score_model[chosen_score],
                score_model[rejected_score],
            )
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
    if shuffle:
        result_df = result_df.sample(frac=1).reset_index(drop=True)

    if push_to_hub:
        unique_ids = result_df["question_id"].unique().tolist()
        random.seed(42)
        random.shuffle(unique_ids)
        train_split_index = int(0.8 * len(unique_ids))  # 80% train
        test_split_index = int(0.9 * len(unique_ids))  # 10% test, 10% eval
        train_ids = set(unique_ids[:train_split_index])
        test_ids = set(unique_ids[train_split_index:test_split_index])
        eval_ids = set(unique_ids[test_split_index:])
        train_df = result_df[result_df["question_id"].isin(train_ids)]
        test_df = result_df[result_df["question_id"].isin(test_ids)]
        eval_df = result_df[result_df["question_id"].isin(eval_ids)]
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
        final_dataset = DatasetDict(
            {
                "train": train_dataset,  # ~80%
                "validation": test_dataset,  # ~10%
                "test": eval_dataset,  # ~10%
            }
        )
        final_dataset.push_to_hub("mattany/ask_science_preference_dataset_1_stddev")

    result_df.to_csv(output_path, index=False)
    print(
        f"Pairwise comparison dataset written to {output_path}.\nCount: {len(result_df)}"
    )


if __name__ == "__main__":
    generate_pairwise_comparisons(
        scores_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{RUN_NUMBER}/aggregations/aggregate_scores.csv",
        answers_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv",
        output_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{RUN_NUMBER}/aggregations/preference_dataset_1_stddev.csv",
        # top_k=3,
        min_diff=1,
        push_to_hub=False,
    )
