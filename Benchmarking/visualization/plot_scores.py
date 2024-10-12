import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    elif type=="sinaplot":
        sns.stripplot(x='Model', y='Score', data=data, jitter=True, color='black', alpha=0.6)
    else:
        assert False
    # Add a title and labels
    plt.title(f'{metric_name.capitalize()} Scores')
    plt.xlabel('Model')
    plt.ylabel('Score')




# Load the data from CSV
# Update the file path as needed
def plot_scores(metric_name, plot_type="swarm"):
    df = pd.read_csv(f'/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric_name}_evaluation_scores_run_0.csv')
    score_columns = [col for col in df.columns if 'score' in col]
    df = df[score_columns]
    # Reshape the DataFrame from wide to long format for Seaborn
    # Assuming the column names are in the format '<metric1_type>_score__<model_name>'
    df_long = df.melt(var_name='Model', value_name='Score')

    # Optionally, you can clean up the 'Model' column to make it just model names
    df_long['Model'] = df_long['Model'].str.replace(r'^.*_score__', '', regex=True)
    plot_figure(metric_name=metric_name, data=df_long, type=plot_type)
    # Display the plot
    plt.show()


if __name__ == "__main__":
    # plot_scores("jargon", plot_type="box")
    # plot_scores("metaphor", plot_type="box")
    plot_scores("explanation_type", plot_type="sinaplot")
    # plot_scores("explanation_type", plot_type="box")
