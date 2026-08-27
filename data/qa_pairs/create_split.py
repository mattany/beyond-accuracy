#!/usr/bin/env python3
"""Split an answers CSV into train/test/eval using the exact same Index
membership as the reference ask_science_gpt_answers_{train,test,eval}.csv splits,
so every model version shares identical questions per fold.

Examples:
    # GPT-5 answers (original behavior)
    python data/qa_pairs/create_split.py \
        --answers data/qa_pairs/ask_science_gpt_5_answers.csv

    # Kimi answers, dropping truncated rows
    python data/qa_pairs/create_split.py \
        --answers data/qa_pairs/ask_science_kimi_answers.csv --drop-truncated
"""
import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
REFERENCE_PREFIX = "ask_science_gpt_answers"


def load_reference_split_indices():
    train = pd.read_csv(DATA_DIR / f"{REFERENCE_PREFIX}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{REFERENCE_PREFIX}_test.csv")
    eval_df = pd.read_csv(DATA_DIR / f"{REFERENCE_PREFIX}_eval.csv")
    return set(train["Index"]), set(test["Index"]), set(eval_df["Index"])


def create_split(answers_csv, drop_truncated=False, dataset_name="ask_science"):
    answers_path = Path(answers_csv)
    if not answers_path.is_absolute():
        # Allow paths relative to the repo root as well as bare filenames.
        answers_path = answers_path if answers_path.exists() else DATA_DIR / answers_path.name
    df = pd.read_csv(answers_path)

    n_start = len(df)
    df = df[df["Answer"] != "No answer available"].copy()
    dropped_no_answer = n_start - len(df)

    dropped_truncated = 0
    if drop_truncated and "Truncated" in df.columns:
        before = len(df)
        df = df[df["Truncated"].astype(int) != 1].copy()
        dropped_truncated = before - len(df)

    train_idx, test_idx, eval_idx = load_reference_split_indices()
    splits = {
        "train": df[df["Index"].isin(train_idx)].copy(),
        "test": df[df["Index"].isin(test_idx)].copy(),
        "eval": df[df["Index"].isin(eval_idx)].copy(),
    }

    stem = answers_path.with_suffix("")  # e.g. .../ask_science_kimi_answers
    for name, split_df in splits.items():
        split_df["Dataset"] = dataset_name
        out_path = Path(f"{stem}_{name}.csv")
        split_df.to_csv(out_path, index=False)

    covered = train_idx | test_idx | eval_idx
    missing = set(df["Index"]) - covered

    print(f"Answers file: {answers_path}")
    print(f"Dropped: {dropped_no_answer} 'No answer available', {dropped_truncated} truncated")
    print(f"Full usable dataset: {len(df):,} rows")
    print(f"Train: {len(splits['train']):,} rows")
    print(f"Test:  {len(splits['test']):,} rows")
    print(f"Eval:  {len(splits['eval']):,} rows")
    print(f"Not in any reference split: {len(missing):,} rows")
    print(f"Wrote {stem}_{{train,test,eval}}.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers", default=str(DATA_DIR / "ask_science_gpt_5_answers.csv"),
                   help="answers CSV to split (columns: Index, Question, Answer[, Truncated])")
    p.add_argument("--drop-truncated", action="store_true",
                   help="exclude rows with Truncated == 1 (if the column exists)")
    p.add_argument("--dataset-name", default="ask_science", help="value for the Dataset column")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_split(args.answers, drop_truncated=args.drop_truncated, dataset_name=args.dataset_name)
