#!/usr/bin/env python3
"""
Generate balanced_dataset_v2 by running v2 metrics on the balanced_dataset.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the deep_eval directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Benchmarking" / "deep_eval"))

import pandas as pd
from tqdm import tqdm
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Suppress DEBUG logs from deepeval and HTTP libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("main_logger").setLevel(logging.WARNING)

from deepeval.test_case import LLMTestCase
from custom_metrics.metrics import (
    humor_metric_explicit_v2,
    metaphor_metric_explicit_v2,
    analogy_metric_explicit_v2,
    connection_to_everyday_life_metric_explicit_v2,
)

GEVAL_RETRIES = 3

# Map metric names to v2 metric objects
V2_METRICS = {
    "humor_v2": humor_metric_explicit_v2,
    "metaphor_v2": metaphor_metric_explicit_v2,
    "analogy_v2": analogy_metric_explicit_v2,
    "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
}


async def evaluate_single_answer(
    question: str,
    answer: str,
    metric_name: str,
    metric_function,
    semaphore: asyncio.Semaphore,
) -> float | None:
    """Evaluate a single answer with a single metric."""
    async with semaphore:
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )
        for i in range(GEVAL_RETRIES):
            try:
                await metric_function.a_measure(test_case)
                return metric_function.score
            except ValueError:
                print(f"  Retry {i+1}/{GEVAL_RETRIES} for {metric_name}: Invalid JSON")
                continue
            except Exception as e:
                print(f"  Error for {metric_name}: {e}")
                break
        return None


async def process_dataset(input_path: str, output_path: str):
    """Process the dataset and add v2 metric scores."""
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Dataset has {len(df)} rows")
    
    # Create new score columns
    for metric_name in V2_METRICS.keys():
        df[f"{metric_name}_score"] = None
    
    semaphore = asyncio.Semaphore(20)  # Limit concurrent API calls
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        question = row["question"]
        answer = row["answer"]
        
        # Run all v2 metrics for this answer
        tasks = []
        metric_names = []
        for metric_name, metric_function in V2_METRICS.items():
            tasks.append(
                evaluate_single_answer(
                    question, answer, metric_name, metric_function, semaphore
                )
            )
            metric_names.append(metric_name)
        
        results = await asyncio.gather(*tasks)
        
        for metric_name, score in zip(metric_names, results):
            df.at[idx, f"{metric_name}_score"] = score
    
    # Save the output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved v2 dataset to: {output_path}")
    
    return df


def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / "balanced_dataset" / "balanced_dataset.csv"
    output_dir = script_dir / "balanced_dataset_v2"
    output_path = output_dir / "balanced_dataset_v2.csv"
    
    # Also copy the labelstudio JSON to the v2 folder for running intercoder_reliability
    import shutil
    json_src = script_dir / "balanced_dataset" / "labelstudio_output.json"
    json_dst = output_dir / "labelstudio_output.json"
    
    asyncio.run(process_dataset(str(input_path), str(output_path)))
    
    # Copy the labelstudio JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_src.exists():
        shutil.copy(json_src, json_dst)
        print(f"Copied labelstudio_output.json to {output_dir}")


if __name__ == "__main__":
    main()

