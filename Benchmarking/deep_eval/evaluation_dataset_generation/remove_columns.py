import os
from time import sleep

import pandas as pd
from pathlib import Path
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from config import (
    PROJECT_DIR,
)

from tqdm import tqdm

# === Load CSV ===
input_path = Path(
    f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv")  # Change this or make it dynamic
df = pd.read_csv(input_path)
print(df.columns)
# df = df.rename(
#     columns={
#         "base_model_answer": "llama-2-7b",
#         "sft_model_answer": "SciComma-2-7b",
#         "llama70B": "llama-3.3-70b",
#         "llama70B_SFT": "SciComma-3.3-70B",
#         "llama3_1_sft": "SciComma-3.1-8B"})

df.to_csv(input_path, index=False)
