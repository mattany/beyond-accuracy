import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import PROJECT_DIR
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
    if type=="box":
        sns.boxplot(x='Model', y='Score', data=data, whis=1.5, width=0.3, fliersize=4)  # fliersize=0 hides outliers
    # Overlay the swarm plot to show individual points
    elif type=="swarm":
        sns.swarmplot(x='Model', y='Score', data=data, color=".25")
    elif type=="strip":
        sns.stripplot(x='Model', y='Score', data=data, jitter=True, color='black', alpha=0.6)
    else:
        assert False
    # Add a title and labels
    sns.pointplot(x="Model", y="Score", data=data, estimator='mean', color='red', markers='o', scale=0.5, ci=None)
    sns.pointplot(x="Model", y="Score", data=data, estimator='median', color='blue', markers='o', scale=0.5, ci=None)

    plt.title(f'{metric_name.capitalize()} Scores')
    plt.xlabel('Model')
    plt.ylabel('Score')


# Load the data from CSV
# Update the file path as needed
def plot_scores(metric_name, plot_type="swarm", hide_models=None, run_number=0):
    df = pd.read_csv(f'/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric_name}_evaluation_scores_run_{run_number}.csv')
    score_columns = [col for col in df.columns if 'score' in col]
    df = df[score_columns]
    df = df.reindex(sorted(df.columns), axis=1)

    # Reshape the DataFrame from wide to long format for Seaborn
    # Assuming the column names are in the format '<metric1_type>_score__<model_name>'
    df_long = df.melt(var_name='Model', value_name='Score')

    # Optionally, you can clean up the 'Model' column to make it just model names
    df_long['Model'] = df_long['Model'].str.replace(r'^.*_score__', '', regex=True)
    if hide_models:
        df_long = df_long.where(~df_long['Model'].isin(hide_models)).dropna()
    plot_figure(metric_name=metric_name, data=df_long, type=plot_type)
    # Display the plot
    plt.show()


def correlation_heatmap_between_columns(metric_name):
    df = pd.read_csv(f'{PROJECT_DIR}/Benchmarking/deep_eval/data/{metric_name}_evaluation_scores_run_0.csv')
    score_columns = [col for col in df.columns if 'score' in col]
    df = df[score_columns]
    df.columns = [col.split('__')[1] for col in df.columns]
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Optional: Adjusts the size of the plot
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    # Display the plot
    plt.title(f'Correlation Matrix Heatmap for {metric_name}')
    plt.show()


def correlation_heatmap(metric_1, metric_2):
    # Create sample data for two dataframes
    np.random.seed(0)  # For reproducibility

    df1 = pd.read_csv(f'/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric_1}_evaluation_scores_run_0.csv')
    score_columns = [col for col in df1.columns if 'score' in col]
    df1 = df1[score_columns]
    df1.columns = [col.split('__')[1] for col in df1.columns]

    df2 = pd.read_csv(f'/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric_2}_evaluation_scores_run_0.csv')
    score_columns = [col for col in df2.columns if 'score' in col]
    df2 = df2[score_columns]
    df2.columns = [col.split('__')[1] for col in df2.columns]

    # Calculate the correlation matrix
    correlation_matrix = df1.corrwith(df2)

    # Create a heatmap to visualize the correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix.values.reshape(1, -1), annot=True, cmap='coolwarm', cbar=True,
                xticklabels=correlation_matrix.index, yticklabels=['Correlation'])

    # Customize the plot
    plt.title("Correlation Between Two DataFrames Across 5 Columns")
    plt.xlabel("Columns")
    plt.ylabel("Correlation")
    plt.show()


if __name__ == "__main__":
    # correlation_heatmap("humor", "metaphor")
    # correlation_heatmap_between_columns("completeness")
    for metric in [
        # 'jargon',
        # 'metaphor',
        'explanation_type',
        # 'analogy',
        # 'humor',
        # 'connection_to_everyday_life',
        # 'content_units',
        # 'correctness_reference:gpt_4o',
        # 'correctness_reference:gpt_4o_validation',
        # 'correctness_reference:llama_2_base',
        # 'alternatives',
        # 'articulation',
        # 'completeness',
        # 'internal_coherence',
        # 'perceived_truth'
    ]:
        plot_scores(metric, plot_type="strip", hide_models=[
            # 'gpt_4',
            # 'gpt_3_5_turbo',
            # 'gpt_3_5_cot',
            # 'gpt_4o',
        ], run_number=1)
