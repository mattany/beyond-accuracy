#!/usr/bin/env python3
"""
Split ask_science_gpt_5_answers.csv into train/test/eval using the exact same
Index membership as the existing ask_science_gpt_answers_{train,test,eval}.csv
splits, so both model versions share identical questions per split.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent


def load_reference_split_indices():
    train = pd.read_csv(DATA_DIR / 'ask_science_gpt_answers_train.csv')
    test = pd.read_csv(DATA_DIR / 'ask_science_gpt_answers_test.csv')
    eval_df = pd.read_csv(DATA_DIR / 'ask_science_gpt_answers_eval.csv')
    return set(train['Index']), set(test['Index']), set(eval_df['Index'])


def main():
    gpt5 = pd.read_csv(DATA_DIR / 'ask_science_gpt_5_answers.csv')
    train_idx, test_idx, eval_idx = load_reference_split_indices()

    # Question 5834 has no answer (batch request hit the max_tokens limit and
    # was written off rather than retried); drop it from every split.
    gpt5 = gpt5[gpt5['Answer'] != 'No answer available'].copy()

    train_df = gpt5[gpt5['Index'].isin(train_idx)].copy()
    test_df = gpt5[gpt5['Index'].isin(test_idx)].copy()
    eval_df = gpt5[gpt5['Index'].isin(eval_idx)].copy()

    for df in (train_df, test_df, eval_df):
        df['Dataset'] = 'ask_science'

    train_df.to_csv(DATA_DIR / 'ask_science_gpt_5_answers_train.csv', index=False)
    test_df.to_csv(DATA_DIR / 'ask_science_gpt_5_answers_test.csv', index=False)
    eval_df.to_csv(DATA_DIR / 'ask_science_gpt_5_answers_eval.csv', index=False)

    covered = train_idx | test_idx | eval_idx
    missing = set(gpt5['Index']) - covered

    print(f"Full gpt-5 dataset: {len(gpt5):,} rows")
    print(f"Train: {len(train_df):,} rows")
    print(f"Test:  {len(test_df):,} rows")
    print(f"Eval:  {len(eval_df):,} rows")
    print(f"Not in any split (matches ask_science_gpt's excluded rows): {len(missing):,} rows")


if __name__ == "__main__":
    main()
