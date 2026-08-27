# Evaluation artifacts

This directory contains the publication evaluation code, canonical model
outputs, final score tables, and factuality analyses.

## Rubric evaluation

Run commands from the repository root. API-backed rubric scoring reads
credentials from environment variables such as `OPENAI_API_KEY`; no local
configuration file is required.

```bash
python -m evaluation.rubrics.custom_metrics.run
python -m evaluation.rubrics.custom_metrics.aggregate_v2 \
  --output-dir /tmp/rubric_aggregations_v2 \
  --bootstrap-dir /tmp/rubric_bootstrap
python -m evaluation.rubrics.custom_metrics.winrate \
  --output /tmp/winrate_llama_vs_gpt35.csv
```

The runner reads model answers from
`evaluation/model_outputs/main/all_models_joined.csv`. Main rubric scoring
(run 9) writes metric CSVs to `evaluation/results/rubric_scores/` and resumes
via `evaluation/results/rubric_scores/.checkpoints/`. Experiment B metric
scoring (run 10) writes to `evaluation/results/preference_metrics/` and resumes
via `evaluation/results/preference_metrics/.checkpoints/`.
`evaluation/rubrics/custom_metrics/aggregate_v2.py` regenerates weighted score
tables and plots.

**Safe reruns:** pass `--output-dir` and `--bootstrap-dir` to `aggregate_v2`
and `--output` to `winrate` so canonical `aggregations_v2/`, `bootstrap/`, and
`winrate__*.csv` files are not overwritten. The API runner mutates metric CSVs
in the canonical result directories; treat full rubric rescoring as a
destructive operation.

The `model_outputs/scillama3/` files are retained conservatively: they have
different model columns from the canonical generation tables and their question
set is not byte-for-byte identical.

## Factuality

Generate the TruthfulQA table and figure from its retained per-question
checkpoints:

```bash
cp -r evaluation/factuality/truthfulqa_results /tmp/truthfulqa_results
python evaluation/factuality/truthfulqa_visualization.py \
  --results-dir /tmp/truthfulqa_results \
  --output /tmp/truthfulqa_comparison.png
```

`--output` controls only the PNG. The script also writes
`truthfulqa_results_summary.csv` into `--results-dir`; use a scratch copy of
the checkpoint directory to avoid overwriting the tracked summary.

`evaluation/factuality/truthfulqa_benchmark.py` is the Colab-oriented
checkpointing runner. `evaluation/factuality/trust_llm/` is a separate,
auxiliary TrustLLM truthfulness analysis.

No tracked repository artifact was found for the paper's GPT-5.2
claim-level/atomic-claim verification. TrustLLM is not the provenance for that
analysis; the missing claim-level artifact remains a documented
reproducibility gap.
