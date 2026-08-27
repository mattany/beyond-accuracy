#!/usr/bin/env python3
"""
Run connection_to_everyday_life v3 metric on a formatted CSV export.

API-required prep utility: not part of the publication rerun command set.
Requires `OPENAI_API_KEY` and never mutates canonical tracked inputs; write results
to an explicit `--output` path.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
from evaluation.rubrics.settings import OPENAI_API_KEY  # noqa: E402

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

GEVAL_RETRIES = 3
CONCURRENCY_LIMIT = 10


async def evaluate_row(index, row, metric_function, scores, reasons, semaphore, LLMTestCase, pbar=None):
    """Evaluate a single row with the metric."""
    async with semaphore:
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
        )
        success = False
        for i in range(GEVAL_RETRIES):
            try:
                await metric_function.a_measure(test_case)
                success = True
                break
            except ValueError:
                print(f"Row {index}: Retry {i+1}/{GEVAL_RETRIES} - Invalid JSON")
                continue
            except Exception as e:
                print(f"Row {index}: Error - {e}")
                break

        if success:
            scores[index] = metric_function.score
            if hasattr(metric_function, "reason") and metric_function.reason:
                reasons[index] = metric_function.reason
            else:
                reasons[index] = None
        else:
            scores[index] = None
            reasons[index] = None
            print(f"Warning: Row {index} failed")

        if pbar:
            pbar.update(1)


async def run_metric(input_path: Path, output_path: Path) -> None:
    from deepeval.test_case import LLMTestCase
    from evaluation.rubrics.custom_metrics.metrics import (
        connection_to_everyday_life_metric_explicit_v3,
    )

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print("Running connection_to_everyday_life v3 metric on all rows (parallel)...")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    scores = {}
    reasons = {}
    pbar = tqdm(total=len(df), desc="Processing v3 metric")
    tasks = [
        evaluate_row(
            row["Index"],
            row,
            connection_to_everyday_life_metric_explicit_v3,
            scores,
            reasons,
            semaphore,
            LLMTestCase,
            pbar,
        )
        for _, row in df.iterrows()
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    df["connection_v3_score"] = df["Index"].map(scores)
    df["connection_v3_reason"] = df["Index"].map(reasons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated CSV with v3 columns to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "API-required prep utility for connection v3 metric scoring. "
            "Writes to --output and never updates the input file in place."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Formatted CSV to score (for example balanced_30_formatted.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV path for scored output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must differ from --input")
    asyncio.run(run_metric(args.input, args.output))


if __name__ == "__main__":
    main()
