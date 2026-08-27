import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from evaluation.rubrics.settings import PROJECT_ROOT, result_directory

RUN_ID = 9


# Create the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Model', y='Score', inner='box', data=df_long)
#
# # Add a title and labels
# plt.title('Violin Plot of Model Metaphor Scores (higher is better)')
# plt.xlabel('Model')
# plt.ylabel('Score')
#
# # Display the plot
# plt.show()
def plot_figure(metric_name, data, type="swarm"):
    # Create the box plot
    plt.figure(figsize=(10, 6))
    if type == "box":
        sns.boxplot(
            x="Model", y="Score", data=data, whis=1.5, width=0.3, fliersize=4
        )  # fliersize=0 hides outliers
    # Overlay the swarm plot to show individual points
    elif type == "swarm":
        sns.swarmplot(x="Model", y="Score", data=data, color=".25")
    elif type == "strip":
        sns.stripplot(
            x="Model", y="Score", data=data, jitter=True, color="black", alpha=0.6
        )
    else:
        assert False
    # Add a title and labels
    sns.pointplot(
        x="Model",
        y="Score",
        data=data,
        estimator="mean",
        color="red",
        markers="o",
        scale=0.5,
        ci=None,
    )
    # sns.pointplot(
    #     x="Model",
    #     y="Score",
    #     data=data,
    #     estimator="median",
    #     color="blue",
    #     markers="o",
    #     scale=0.5,
    #     ci=None,
    # )

    plt.title(f"{metric_name.capitalize()} Scores")

    plt.xticks(rotation=45, verticalalignment="top", horizontalalignment="right")
    plt.xlabel("Model")
    plt.ylabel("Score")


# Load the data from CSV
# Update the file path as needed
def plot_scores(metric_name, plot_type="swarm", hide_models=None, run_number=9):
    df = pd.read_csv(
        result_directory(run_number) / f"{metric_name}.csv"
    )
    score_columns = [col for col in df.columns if "score" in col]
    df = df[score_columns]
    df = df.reindex(sorted(df.columns), axis=1)

    # Reshape the DataFrame from wide to long format for Seaborn
    # Assuming the column names are in the format '<metric1_type>_score__<model_name>'
    df_long = df.melt(var_name="Model", value_name="Score")

    # Optionally, you can clean up the 'Model' column to make it just model names
    df_long["Model"] = df_long["Model"].str.replace(r"^.*_score__", "", regex=True)
    if hide_models:
        df_long = df_long.where(~df_long["Model"].isin(hide_models)).dropna()
    plot_figure(metric_name=metric_name, data=df_long, type=plot_type)
    output_path = PROJECT_ROOT / "evaluation/visualization/images/plots" / f"{metric_name}_{plot_type}_plot.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.tight_layout()
    # Display the plot
    plt.show()


def correlation_heatmap_between_columns(metric_name):
    df = pd.read_csv(
        result_directory(RUN_ID) / f"{metric_name}_evaluation_scores.csv"
    )
    score_columns = [col for col in df.columns if "score" in col]
    df = df[score_columns]
    df.columns = [col.split("__")[1] for col in df.columns]
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Optional: Adjusts the size of the plot
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

    # Display the plot
    plt.title(f"Correlation Matrix Heatmap for {metric_name}")
    plt.show()


def winrate_matrix(metric_name, run_number=9):
    df = pd.read_csv(
        result_directory(run_number) / f"{metric_name}_evaluation_scores.csv"
    )
    score_columns = [col for col in df.columns if "score" in col]
    df = df[score_columns]
    df.columns = [col.split("__")[1] for col in df.columns]

    def calculate_filtered_winrate(df):
        # Create a 3D boolean array for win/loss by comparing columns
        win_matrix = df.values[:, :, None] > df.values[:, None, :]

        # Create a mask for non-ties (i.e., exclude where values are equal)
        non_ties = df.values[:, :, None] != df.values[:, None, :]

        # Calculate winrate only where there are no ties
        winrate_matrix = (win_matrix & non_ties).sum(axis=0) / non_ties.sum(axis=0)

        return pd.DataFrame(winrate_matrix, index=df.columns, columns=df.columns)

    # Calculate the winrate matrix after filtering out ties
    winrate_matrix = calculate_filtered_winrate(df)

    # Plot the winrate matrix as a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        winrate_matrix,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        linewidths=0.5,
        fmt=".2f",
    )

    # Add labels and title
    plt.title(
        f"{metric_name.capitalize()} Filtered Winrate Matrix Heatmap (Excluding Ties)"
    )
    plt.xlabel("Compared Against")
    plt.ylabel("Column")

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "evaluation/visualization/images/heatmaps" / f"{metric_name}_winrate_matrix.png"
    )
    plt.show()


def correlation_heatmap(metric_1, metric_2):
    # Create sample data for two dataframes
    np.random.seed(0)  # For reproducibility

    df1 = pd.read_csv(
        result_directory(9) / f"{metric_1}.csv"
    )
    score_columns = [col for col in df1.columns if "score" in col]
    df1 = df1[score_columns]
    df1.columns = [col.split("__")[1] for col in df1.columns]

    df2 = pd.read_csv(
        result_directory(9) / f"{metric_2}.csv"
    )
    score_columns = [col for col in df2.columns if "score" in col]
    df2 = df2[score_columns]
    df2.columns = [col.split("__")[1] for col in df2.columns]

    # Calculate the correlation matrix
    correlation_matrix = df1.corrwith(df2)

    # Create a heatmap to visualize the correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix.values.reshape(1, -1),
        annot=True,
        cmap="coolwarm",
        cbar=True,
        xticklabels=correlation_matrix.index,
        yticklabels=["Correlation"],
    )

    # Customize the plot
    plt.title("Correlation Between Two DataFrames Across 5 Columns")
    plt.xlabel("Columns")
    plt.ylabel("Correlation")
    plt.show()


def display_means(run_number, prefix=""):
    # Read the means.csv
    df = pd.read_csv(
        result_directory(run_number) / f"{prefix}means.csv"
    )

    # Separate out the metric labels and the numeric data
    metrics = df["metric"]
    data = df.drop(columns=["metric"])

    # Calculate the average score for each metric (row-wise mean)
    metric_averages = data.mean(axis=1)



    # Create a DataFrame with metric names and their average values
    avg_df = pd.DataFrame({
        "metric": metrics,
        "average": metric_averages
    })

    # Sort metrics by average value
    avg_df_sorted = avg_df.sort_values(by="average", ascending=True)

    # Split the metrics into 4 groups
    groups = np.array_split(avg_df_sorted, 4)

    # Create a separate figure for each group
    for group_index, group in enumerate(groups, start=1):
        group_metrics = group["metric"].values
        group_indices = df["metric"].isin(group_metrics)
        group_data = data[group_indices]

        # Compute the average score for each column (across the selected metrics in this group)
        column_means = group_data.mean(axis=0)

        # Sort the columns by average score in descending order (best to worst)
        sorted_columns = column_means.sort_values(ascending=False).index.tolist()
        sorted_data = group_data[sorted_columns]

        # Plot
        plt.figure(figsize=(10, 6))
        x_positions = range(len(sorted_columns))

        for i, row in sorted_data.iterrows():
            plt.scatter(x_positions, row.values, label=metrics[i])

        # Labeling
        plt.xticks(x_positions, sorted_columns, rotation=45, verticalalignment="top", horizontalalignment="right")
        plt.xlabel("Columns (Sorted by Average Score)")
        plt.ylabel("Mean Values")
        plt.title(f"Strip Plot of Means (Group {group_index})")
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    display_means(run_number=9, prefix="normalized_")
    # correlation_heatmap("humor", "metaphor")
    # correlation_heatmap_between_columns("completeness")
    for metric in [
        "jargon",
        "explanation_type",
        "metaphor_explicit",
        "content_units_explicit",
        "humor_explicit",
        "analogy_explicit",
        "connection_to_everyday_life",
        "internal_coherence_explicit",
        "completeness_explicit",
        "alternatives_explicit",
        "articulation_explicit",
        "perceived_truth_explicit",
        # 'flesch_kincaid',
        # 'flesch_reading_ease',
        # 'dale_chall',
        # 'ari',
        # 'perceived_truth'
        # 'correctness_reference:gpt_4o',
        # 'correctness_reference:gpt_4o_validation',
        # 'correctness_reference:llama_2_base',
    ]:
        pass
        # plot_scores(
        #     metric,
        #     plot_type="strip",
        #     hide_models=[
        #         # 'gpt_4',
        #         # 'gpt_3_5_turbo',
        #         # 'gpt_3_5_cot',
        #         # 'gpt_4o',
        #     ],
        #     run_number=5,
        # )
        # winrate_matrix(metric_name=metric, run_number=0)
