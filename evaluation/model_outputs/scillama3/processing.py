"""Convert SciComma Llama 3.3 70B QA dumps to CSV.

The ``output.csv`` and ``base_model_output.csv`` files in this directory hold
Llama-3.3-70B SFT and base-model answers for the 91-item evaluation subset.
They are not duplicated in ``evaluation/model_outputs/main/`` (which covers the
8B SciComma variants) and are retained for exploratory 70B comparisons.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def process_file(input_file: str | Path, output_file: str | Path) -> None:
    """Parse a QA text dump and write ``row_index,index,question,answer`` CSV."""
    with open(input_file, "r", encoding="utf-8") as handle:
        data = handle.read()

    pattern = re.compile(
        r"### Index: (\d+)\n<\|begin_of_text\|>.*?### Question:\n(.*?)\n\n### Answer:\n(.*?)(?=\n### Index:|<\|eot_id\|>|$)",
        re.DOTALL,
    )
    matches = pattern.findall(data)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["row_index", "index", "question", "answer"])
        for row_index, match in enumerate(matches, start=1):
            index, question, answer = match
            writer.writerow([row_index, index.strip(), question.strip(), answer.strip()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SciComma Llama QA text dump to CSV."
    )
    parser.add_argument("input_file", type=Path, help="Source QA text dump")
    parser.add_argument("output_file", type=Path, help="Destination CSV path")
    args = parser.parse_args()
    process_file(args.input_file, args.output_file)
    print(f"Conversion completed. CSV saved to {args.output_file}")


if __name__ == "__main__":
    main()
