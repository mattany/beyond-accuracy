import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EVAL_DIR = "eval_results"

def flatten_json(d, parent_key='', sep='.'):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)

# Collect all results
model_results = {}
for file in os.listdir(EVAL_DIR):
    if file.endswith(".json"):
        model_name = file.replace("_eval_truthfulness.json", "")
        with open(os.path.join(EVAL_DIR, file), "r") as f:
            data = json.load(f)
        model_results[model_name] = flatten_json(data)

# Convert to DataFrame
df = pd.DataFrame(model_results).T

# Identify metrics where lower is better
lower_is_better = [col for col in df.columns if "sycophancy" in col]

# Normalize so better = 1, worse = 0
norm_df = pd.DataFrame(index=df.index)
for col in df.columns:
    if col in lower_is_better:
        norm_df[col] = df[col].max() - df[col]
    else:
        norm_df[col] = df[col]
    # scale 0-1
    if norm_df[col].max() != norm_df[col].min():
        norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())
    else:
        norm_df[col] = 0.0

# --- Optional: order models ---
# Provide a list of model names in desired order
model_order = [
    "llama2-7b",
    "scicomma-2-7b",
    "llama3.1-8b",
    "scicomma-3.1-8b",
    "llama3.3-70b-4bit",
    "scicomma-3.3-70b"
]
#
# # Keep only models that exist in df
# model_order = [m for m in model_order if m in df.index]
# norm_df = norm_df.loc[model_order]
# df = df.loc[model_order]
#
# # Plot heatmap
# plt.figure(figsize=(14, 6))
# sns.heatmap(norm_df, annot=df.round(3), fmt="", cmap="Reds", cbar=False)
# plt.title("TrustLLM Benchmark Heatmap (Red = Better)", fontsize=14)
# plt.yticks(rotation=0)
# plt.show()
# Keep only numeric columns for plotting
numeric_cols = df.select_dtypes(include='number').columns
df_numeric = df[numeric_cols]
norm_df_numeric = norm_df[numeric_cols]

# Now plot heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(norm_df_numeric, annot=df_numeric.round(3), fmt="", cmap="Reds", cbar=False)
plt.title("TrustLLM Benchmark Heatmap (Red = Better)", fontsize=14)
plt.yticks(rotation=0)
plt.show()