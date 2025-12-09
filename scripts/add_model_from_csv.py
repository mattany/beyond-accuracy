import pandas as pd
input_path = "/Users/mattan.yeroushalmi/studies/thesis/scripts/generations_2/"
df = pd.read_csv(f"{input_path}/output_dpo_v2_512 tokens_short prompt.csv")
main_df = pd.read_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv")


df = df.rename(columns={
    "Question": "question",
    "Answer": "scicomma-3.1-dpo_real_512_short",
})

# Ensure both question columns have the same data type (string)
df['question'] = df['question'].astype(str)
main_df['question'] = main_df['question'].astype(str)

# Use merge instead of join for better control
joined_df = main_df.merge(df, on="question", how="inner")
joined_df.to_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv", index=False)