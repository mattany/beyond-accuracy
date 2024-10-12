import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from CSV
# Update the file path as needed
df = pd.read_csv('/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/metaphor_evaluation_scores_run_0.csv')
score_columns = [col for col in df.columns if 'score' in col]
df = df[score_columns]
# Reshape the DataFrame from wide to long format for Seaborn
# Assuming the column names are in the format '<metric1_type>_score__<model_name>'
df_long = df.melt(var_name='Model', value_name='Score')

# Optionally, you can clean up the 'Model' column to make it just model names
df_long['Model'] = df_long['Model'].str.replace(r'^.*_score__', '', regex=True)

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


# Create the box plot
plt.figure(figsize=(10, 6))
# sns.boxplot(x='Model', y='Score', data=df_long, whis=1.5, width=0.3, fliersize=0)  # fliersize=0 hides outliers

# Overlay the swarm plot to show individual points
sns.swarmplot(x='Model', y='Score', data=df_long, color=".25")

# Add a title and labels
plt.title('Box Plot with Swarm of Model Scores (higher is better)')
plt.xlabel('Model')
plt.ylabel('Score')

# Display the plot
plt.show()