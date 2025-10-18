import pandas as pd
input_path = "/Users/mattan.yeroushalmi/studies/thesis/scripts/generations"
df = pd.read_csv(f"{input_path}/all_models_joined.csv")
main_df = pd.read_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv")


df = df.rename(columns={
    "Question": "question",
    "SciComma-3.1-8B-DPO": "scicomma-3.1-dpo",
    "SciComma-3.1-8B-DPO_prompt": "scicomma-3.1-dpo_prompt",
})

# Ensure both question columns have the same data type (string)
df['question'] = df['question'].astype(str)
main_df['question'] = main_df['question'].astype(str)

# Use merge instead of join for better control
joined_df = main_df.merge(df, on="question", how="inner")
joined_df.to_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv", index=False)