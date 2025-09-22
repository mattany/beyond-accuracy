import pandas as pd

df = pd.read_csv("input1dir/scripts/output_3.csv")
main_df = pd.read_csv("input2dir/data/test_data/corrected_evaluation_dataset.csv")


df = df.rename(columns={"Question": "question", "Answer": "scicomma-3.1-dpo"})

# Ensure both question columns have the same data type (string)
df['question'] = df['question'].astype(str)
main_df['question'] = main_df['question'].astype(str)

# Use merge instead of join for better control
joined_df = main_df.merge(df, on="question", how="inner")
joined_df.to_csv("outputdir/corrected_evaluation_dataset_with_answers_2.csv", index=False)