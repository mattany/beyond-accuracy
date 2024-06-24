import json

import pandas as pd

# df = pd.read_csv("./data/GPT_Questions.csv")
# df['Number'] = range(0, len(df))
# df.to_csv("./outputs/numbered_questions.csv", index=False)

def merge():

    df = pd.read_csv("./outputs/numbered_questions.csv")

    output_dict = json.load(open("outputs/output_dict_2.jsonl"))
    answer_df = pd.DataFrame(output_dict, index=[0]).transpose()
    answer_df["Number"] = answer_df.index
    df["Number"] = df["Number"].astype(int)
    answer_df["Number"] = answer_df["Number"].dropna().astype(int)
    merged = pd.merge(df, answer_df, left_on="Number", right_on="Number")
    # print(merged)
    merged.to_csv("./outputs/numbered_questions_with_2nd_batch_answers.csv", index=False)
    # print(output_dict)
merge()
df1 = pd.read_csv("./outputs/numbered_questions_with_first_batch_answers.csv")
df2 = pd.read_csv("./outputs/numbered_questions_with_2nd_batch_answers.csv")
df3 = pd.read_csv("./outputs/numbered_questions_with_3rd_batch_answers.csv")
df4 = pd.read_csv("./outputs/numbered_questions_with_4th_batch_answers.csv")
merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
merged_df.to_csv("final.csv")
# merged = pd.merge(df1, df2,  left_on="Number", right_on="Number")
# merged = pd.merge(merged, df3,  left_on="Number", right_on="Number")
# merged.to_csv("./outputs/final.csv")