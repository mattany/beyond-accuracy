# Evaluation artifacts

This directory contains the publication evaluation code, canonical model
outputs, final score tables, and factuality analyses.

## Rubric evaluation

Run commands from the repository root. API-backed rubric scoring reads
credentials from environment variables such as `OPENAI_API_KEY`; no local
configuration file is required.

```bash
python -m evaluation.rubrics.custom_metrics.run
python -m evaluation.rubrics.custom_metrics.aggregate_v2
python -m evaluation.rubrics.custom_metrics.winrate
```

The runner reads model answers from
`evaluation/model_outputs/main/all_models_joined.csv`. It writes the main
rubric evaluation to `evaluation/results/rubric_scores/`; Experiment B writes
to `evaluation/results/preference_metrics/`.
`evaluation/rubrics/custom_metrics/aggregate_v2.py` regenerates the canonical
weighted score tables and plots.

The hidden `.checkpoints/` directory under the preference results is retained
because `custom_metrics/run.py` loads those JSON files to resume incomplete
model/metric evaluations. The `model_outputs/scillama3/` files are also
retained conservatively: they have different model columns from the canonical
generation tables and their question set is not byte-for-byte identical.

## Factuality

Generate the TruthfulQA table and figure from its retained per-question
checkpoints:

```bash
python evaluation/factuality/truthfulqa_visualization.py \
  --results-dir evaluation/factuality/truthfulqa_results
```

`evaluation/factuality/truthfulqa_benchmark.py` is the Colab-oriented
checkpointing runner. `evaluation/factuality/trust_llm/` is a separate,
auxiliary TrustLLM truthfulness analysis.

No tracked repository artifact was found for the paper's GPT-5.2
claim-level/atomic-claim verification. TrustLLM is not the provenance for that
analysis; the missing claim-level artifact remains a documented
reproducibility gap.
