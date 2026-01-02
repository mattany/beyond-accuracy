import pandas as pd
from pathlib import Path

# Configuration
INPUT_DIR = Path("/Users/mattan.yeroushalmi/studies/thesis/scripts/generations_3")
MAIN_DATASET_PATH = "/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv"

# Load the main dataset
main_df = pd.read_csv(MAIN_DATASET_PATH)
main_df['question'] = main_df['question'].astype(str)

# Get all CSV files from the input directory
csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {INPUT_DIR}")
    exit(1)

print(f"Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    print(f"  - {f.name}")

# Process each CSV file
for csv_file in csv_files:
    # Extract model name from filename (without .csv extension)
    model_name = csv_file.stem
    
    print(f"\nProcessing {csv_file.name} -> model column: '{model_name}'")
    
    # Read the model's CSV
    df = pd.read_csv(csv_file)
    
    # Rename columns: Question -> question, Answer -> model_name
    df = df.rename(columns={
        "Question": "question",
        "Answer": model_name,
    })
    
    # Keep only the columns we need
    df = df[["question", model_name]]
    
    # Ensure question column has string type
    df['question'] = df['question'].astype(str)
    
    # Merge with main dataset
    initial_rows = len(main_df)
    main_df = main_df.merge(df, on="question", how="left")
    
    # Report on merge results
    matched = main_df[model_name].notna().sum()
    print(f"  Matched {matched}/{initial_rows} questions")

# Save the updated dataset
main_df.to_csv(MAIN_DATASET_PATH, index=False)
print(f"\nSaved updated dataset to {MAIN_DATASET_PATH}")
print(f"Total columns: {len(main_df.columns)}")
print(f"New model columns added: {[f.stem for f in csv_files]}")
