import os
from time import sleep

import pandas as pd
from pathlib import Path
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from config import (
    PROJECT_DIR,
    # OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    LANGCHAIN_API_KEY)
from tqdm import tqdm
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "scillama"

# === Load CSV ===
input_path = Path(f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv")  # Change this or make it dynamic
df = pd.read_csv(input_path)

if "question" not in df.columns:
    raise ValueError("CSV must contain a 'question' column.")

questions = df["question"].tolist()

# === Set up LangChain model ===
# model_name = "o1"
model_name = "claude-3-7-sonnet-20250219"
# llm = ChatOpenAI(model=model_name)
llm = ChatAnthropic(model=model_name)
# === Create prompt template ===
prompt = ChatPromptTemplate.from_template(
    """Answer the following question succinctly in three paragraphs or less. Keep your answer short.
Question: {question}""")

# === Create a Runnable that maps prompts to model ===
chain = prompt | llm
batch_size = 5
rate_limit_duration = 60
responses = []
for i in tqdm(range(0, len(questions), batch_size)):
    responses += chain.batch([{"question": q} for q in questions[i:i+batch_size]])
    sleep(rate_limit_duration)
# === Extract responses from Message objects ===
df[model_name] = [resp.content for resp in responses]
print([resp.content for resp in responses])
# === Save to output.csv in the same directory ===
output_path = input_path.parent / "output.csv"
model_only = input_path.parent / "model_only.csv"
df.to_csv(output_path, index=False)
df[["question", model_name]].to_csv(model_only, index=False)
print(f"Output written to {output_path}")
