import csv
import re
import pandas as pd

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read()

    # Regular expression to capture index, question, and answer
    pattern = re.compile(
        r"### Index: (\d+)\n<\|begin_of_text\|>.*?### Question:\n(.*?)\n\n### Answer:\n(.*?)(?=\n### Index:|<\|eot_id\|>|$)",
        re.DOTALL
    )

    # Extract matches
    matches = pattern.findall(data)

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["row_index", "index", "question", "answer"])

        for row_index, match in enumerate(matches, start=1):
            index, question, answer = match
            # Clean up special tokens from question and answer
            question = question.strip()
            answer = answer.strip()
            writer.writerow([row_index, index, question, answer])


#
# # # Input and output file paths
# input_file = "base_model_qa.txt"
# output_file = "base_model_output.csv"
# #
# process_file(input_file, output_file)
# print(f"Conversion completed. CSV saved to {output_file}")
#
#
# df = pd.read_csv("../data/test_data/corrected_evaluation_dataset.csv")
# llama_df = pd.read_csv("../scillama3/base_model_output.csv").drop(columns=["question"])
# out_df = df.merge(llama_df, right_on="index", left_on="index")
# out_df.to_csv("../scillama3/corrected_evaluation_dataset.csv")


df = pd.read_csv("../scillama3/corrected_evaluation_dataset.csv")
llama_df = pd.read_csv("../scillama3/output.csv").drop(columns=["question"])
out_df = df.merge(llama_df, right_on="index", left_on="index")
out_df.to_csv("../scillama3/corrected_evaluation_dataset.csv")