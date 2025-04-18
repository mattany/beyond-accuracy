import os
import pandas as pd
from config import PROJECT_DIR
import numpy as np
run_number = 5


def calculate_normalized_means(directory):
    def normalize_df(df, lower_is_better):
        """Normalize values in the DataFrame to [0, 1] range if not already normalized."""
        if (
            df.select_dtypes(include=[np.number])
            .map(lambda x: 0 <= x <= 1)
            .all()
            .all()
        ):
            return df  # Already normalized

        numeric_df = df.select_dtypes(include=[np.number])
        min_val = numeric_df.min().min()
        max_val = numeric_df.max().max()

        if min_val == max_val:
            return df  # Avoid division by zero; leave unchanged
        if lower_is_better:
            normalized = 1 - ((numeric_df - min_val) / (max_val - min_val))
        else:
            normalized = (numeric_df - min_val) / (max_val - min_val)
        df[normalized.columns] = normalized
        return df

    def process_csvs():
        output_file = os.path.join(directory, "normalized_means.csv")
        lower_is_better = {
            "ari", "dale_chall", "flesch_kincaid"
        }
        # Delete normalized_means.csv if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Deleted existing {output_file}")

        rows = []

        for filename in os.listdir(directory):
            if not filename.endswith(".csv") or filename == "normalized_means.csv":
                continue

            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                df = normalize_df(df, lower_is_better=filename in lower_is_better)
                means = df.mean(numeric_only=True)
                means["metric"] = os.path.splitext(filename)[0]
                rows.append(means)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

        if rows:
            result_df = pd.DataFrame(rows)
            cols = ["metric"] + [col for col in result_df.columns if col != "metric"]
            result_df = result_df[cols]
            result_df.to_csv(output_file, index=False)
            print(f"Wrote normalized means to {output_file}")
        else:
            print("No valid CSV files found.")

    process_csvs()  # Replace "." with your target directory if needed


if __name__ == "__main__":
    directory = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}"
    calculate_normalized_means(directory)
