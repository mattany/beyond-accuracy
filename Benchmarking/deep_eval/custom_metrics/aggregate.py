import os
import pandas as pd
from config import PROJECT_DIR

run_number = 5


def calculate_means(directory):
    means_file = os.path.join(directory, "means.csv")

    # Delete means.csv if it exists
    if os.path.exists(means_file):
        os.remove(means_file)
        print(f"Deleted existing {means_file}")

    rows = []

    # Iterate over CSV files in directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".csv") and filename != "means.csv":
            try:
                df = pd.read_csv(filepath)
                means = df.mean(numeric_only=True)
                means["metric"] = os.path.splitext(filename)[0]
                rows.append(means)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if rows:
        # Combine all means into a single DataFrame
        means_df = pd.DataFrame(rows)
        # Move 'metric' column to the front
        cols = ["metric"] + [col for col in means_df.columns if col != "metric"]
        means_df = means_df[cols]
        means_df.to_csv(means_file, index=False)
        print(f"Written means to {means_file}")
    else:
        print("No valid CSV files found.")


if __name__ == "__main__":
    directory = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}"
    calculate_means(directory)
