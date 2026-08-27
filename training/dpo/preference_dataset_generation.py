"""Prepare pairwise preference data from per-answer scores and model outputs.

This module constructs DPO-compatible ``prompt``, ``chosen``, and ``rejected``
records. It does not train a model; no end-to-end DPO trainer is tracked here.
"""

from argparse import ArgumentParser
import itertools
from pathlib import Path
import random

from datasets import Dataset, DatasetDict
import pandas as pd


def generate_pairwise_comparisons(
    scores_path,
    answers_path,
    output_path,
    include_metadata=True,
    top_k=None,
    min_diff=0,
    shuffle=True,
    push_to_hub=False,
    hub_repo=None,
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
        if not hub_repo:
            raise ValueError("hub_repo is required when push_to_hub is enabled")
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
        final_dataset.push_to_hub(hub_repo)

    result_df.to_csv(output_path, index=False)
    print(
        f"Pairwise comparison dataset written to {output_path}.\nCount: {len(result_df)}"
    )


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="per-question score CSV indexed by Index, with one column per model",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="answer CSV indexed by index, with question and model-answer columns",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-diff", type=float, default=1.0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", help="Hugging Face dataset repository ID")
    args = parser.parse_args(argv)
    if args.push_to_hub and not args.hub_repo:
        parser.error("--hub-repo is required with --push-to-hub")
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_pairwise_comparisons(
        scores_path=args.scores,
        answers_path=args.answers,
        output_path=args.output,
        top_k=args.top_k,
        min_diff=args.min_diff,
        shuffle=not args.no_shuffle,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )
