import pandas as pd
from config import PROJECT_DIR
path_to_cot_answers = f"{PROJECT_DIR}/SFT/data/ask_science_gpt_answers.csv"

answer_df = pd.read_csv(path_to_cot_answers)
answer_df.columns = [col.lower() for col in answer_df.columns]
evaluation_df = pd.read_csv(f"{PROJECT_DIR}/Benchmarking/deep_eval/data/evaluation_dataset.csv")
joined_df = evaluation_df.merge(answer_df, on="index", how="left")
evaluation_df["gpt_3_5_cot"] = joined_df["answer"]
evaluation_df.to_csv(f"{PROJECT_DIR}/Benchmarking/deep_eval/data/evaluation_dataset.csv", index=False)