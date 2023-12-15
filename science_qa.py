import pandas as pd

# Specify the path to your Parquet file
input_files_dir = 'science-QA'
output_files_dir = 'science-QA_jsonl'
mapping = {
    f'./{input_files_dir}/train-00000-of-00001-1028f23e353fbe3e.parquet': f'./{output_files_dir}/train.jsonl',
    f'./{input_files_dir}/test-00000-of-00001-f0e719df791966ff.parquet': f'./{output_files_dir}/test.jsonl',
    f'./{input_files_dir}/validation-00000-of-00001-6c7328ff6c84284c.parquet': f'./{output_files_dir}/validation.jsonl',

}

for parquet_file_path, output_file in mapping.items():
    # Use pandas.read_parquet to load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file_path)
    df = df[pd.isna(df["image"])]
    json_data = df.to_json(orient='records', lines=True)

    # Write the JSON data to a JSON Lines file
    with open(output_file, 'w') as jsonl_file:
        jsonl_file.write(json_data)

# # Now, 'df' contains your data from the Parquet file
# with open(, "w") as f:
#     for line in df:
#         print(line)