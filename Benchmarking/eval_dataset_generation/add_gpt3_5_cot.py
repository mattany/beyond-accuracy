from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--answers",
        type=Path,
        default=ROOT / "data" / "qa_pairs" / "ask_science_gpt_answers.csv",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "Benchmarking" / "deep_eval" / "data" / "evaluation_dataset.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    answer_df = pd.read_csv(args.answers)
    answer_df.columns = [col.lower() for col in answer_df.columns]
    evaluation_df = pd.read_csv(args.evaluation)
    joined_df = evaluation_df.merge(answer_df, on="index", how="left")
    evaluation_df["gpt_3_5_cot"] = joined_df["answer"]
    evaluation_df.to_csv(args.evaluation, index=False)