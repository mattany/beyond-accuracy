import os
import pandas as pd
from config import PROJECT_DIR
import numpy as np

run_number = 5


def normalize_df(df, is_lower_better):
    """Normalize values in the DataFrame to [0, 1] range if not already normalized."""
    if df.select_dtypes(include=[np.number]).map(lambda x: 0 <= x <= 1).all().all():
        return df  # Already normalized

    numeric_df = df.select_dtypes(include=[np.number])
    min_val = numeric_df.min().min()
    max_val = numeric_df.max().max()

    if min_val == max_val:
        return df  # Avoid division by zero; leave unchanged
    if is_lower_better:
        normalized = 1 - ((numeric_df - min_val) / (max_val - min_val))
    else:
        normalized = (numeric_df - min_val) / (max_val - min_val)
    df.loc[:, normalized.columns] = normalized
    return df


def aggregate_over_model_question(
    directory: str, ignore_files: list[str], lower_is_better_metrics: list[str]
):
    metric_files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f not in ignore_files
    ]

    normalized_dfs = []
    for file_name in metric_files:
        full_path = os.path.join(directory, file_name)
        df = pd.read_csv(full_path)

        # Filter to score columns only (exclude "reason" columns)
        score_columns = [
            col for col in df.columns if not col.lower().endswith("reason")
        ]
        df_scores = df[score_columns].copy()
        is_lower_better = file_name in lower_is_better_metrics
        df_normalized = normalize_df(df_scores, is_lower_better)
        normalized_dfs.append(df_normalized)

    # Check that all dataframes have the same shape
    shape_set = set(df.shape for df in normalized_dfs)
    if len(shape_set) > 1:
        raise ValueError(f"Mismatch in shapes across CSVs: {shape_set}")

    # Average the scores
    aggregated_df = pd.concat(normalized_dfs).groupby(level=0).mean()
    output_path = os.path.join(directory, "aggregate_scores.csv")
    aggregated_df.index.name = "Index"
    aggregated_df.to_csv(output_path, index=True)
    print(f"Aggregate scores written to {output_path}")


def aggregate_over_model_metric(directory, lower_is_better_metrics):
    def process_csvs():
        output_file = os.path.join(directory, "normalized_means.csv")
        # Delete normalized_means.csv if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Deleted existing {output_file}")

        rows = []
        all_normalized_dfs = []

        for filename in os.listdir(directory):
            if not filename.endswith(".csv") or filename == "normalized_means.csv":
                continue

            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                df = normalize_df(
                    df, is_lower_better=filename in lower_is_better_metrics
                )
                all_normalized_dfs.append(df)
                means = df.mean(numeric_only=True, skipna=True)
                means["metric"] = os.path.splitext(filename)[0]
                rows.append(means)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

        if rows:
            means_df = pd.DataFrame(rows)
            cols = ["metric"] + [col for col in means_df.columns if col != "metric"]
            means_df = means_df[cols]
            means_df.to_csv(output_file, index=False)
            print(f"Wrote normalized means to {output_file}")

            # Also compute global means across all CSVs (column-wise), ignoring nulls
            combined_df = pd.concat(all_normalized_dfs, ignore_index=True)
            global_means = combined_df.mean(numeric_only=True, skipna=True)
            print("\nGlobal mean for each column across all CSVs (ignoring nulls):")
            print(global_means.sort_values())
        else:
            print("No valid CSV files found.")

    process_csvs()  # Replace "." with your target directory if needed


if __name__ == "__main__":
    csv_path = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}"
    lower_is_better_metrics = ["ari", "dale_chall", "flesch_kincaid"]
    # aggregate_over_model_metric(
    #     directory=csv_path,
    #     lower_is_better_metrics={"ari", "dale_chall", "flesch_kincaid"},
    # )
    aggregate_over_model_question(
        directory=csv_path,
        ignore_files=["normalized_means.csv", "aggregate_scores.csv"],
        lower_is_better_metrics=lower_is_better_metrics,
    )
