import os
from time import sleep

import pandas as pd
from pathlib import Path
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from evaluation.rubrics.settings import (
    ANTHROPIC_API_KEY,
    LANGCHAIN_API_KEY,
    PROJECT_ROOT,
    require_env,
)
from tqdm import tqdm


def main() -> None:
    anthropic_key = require_env("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    langchain_key = require_env("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["LANGCHAIN_PROJECT"] = "scillama"

    input_path = PROJECT_ROOT / "evaluation/model_outputs/main/all_models_joined.csv"
    df = pd.read_csv(input_path)
    df = df.rename(columns={"Question": "question"})
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")
    questions = df["question"].tolist()

    model_name = "claude-3-7-sonnet-20250219"
    llm = ChatAnthropic(model=model_name)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question succinctly in three paragraphs or less. "
        "Keep your answer short.\nQuestion: {question}"
    )
    chain = prompt | llm
    batch_size = 5
    rate_limit_duration = 60
    responses = []
    for i in tqdm(range(0, len(questions), batch_size)):
        responses += chain.batch(
            [{"question": question} for question in questions[i : i + batch_size]]
        )
        sleep(rate_limit_duration)

    df[model_name] = [response.content for response in responses]
    output_path = input_path.parent / "output.csv"
    model_only = input_path.parent / "model_only.csv"
    df.to_csv(output_path, index=False)
    df[["question", model_name]].to_csv(model_only, index=False)
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
