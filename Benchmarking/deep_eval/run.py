import pandas as pd
from sklearn.model_selection import train_test_split
#"~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv"
# df = pd.read_csv("~/studies/thesis/SFT/data/ask_science_gpt_answers.csv")
df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv")
_, eval_df = train_test_split(df, test_size=0.02, random_state=42)
eval_df.to_csv("~/studies/thesis/Benchmarking/deep_eval/data/eval_dataset.csv", index=False)