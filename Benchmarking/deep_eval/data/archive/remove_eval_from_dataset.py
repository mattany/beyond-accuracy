import os

import pandas as pd

# eval_ds = pd.read_csv("./data/evaluation_dataset.csv")
# test_questions = pd.read_csv("../../SFT/data/ask_science_gpt_answers_test.csv")
# filtered = eval_ds[eval_ds["index"].isin(test_questions['Index'])]
# print(filtered.head())
# filtered.to_csv("./data/corrected_evaluation_dataset.csv", index=False)

eval_ds = "./data/test_data/corrected_evaluation_dataset.csv"
old_data_dir = "/Benchmarking/deep_eval/data/archive/old_data"
for results in os.listdir(old_data_dir):
    df = pd.read_csv(f"{old_data_dir}/{results}")
    df = df.reset_index()
    eval_df = pd.read_csv(eval_ds)
    filtered = df[df["index"].isin(eval_df["Index"])]
    filtered.to_csv(f"./data/test_data/{results}", index=False)