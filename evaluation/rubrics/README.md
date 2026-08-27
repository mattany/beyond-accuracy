# Science-communication rubric evaluation

This Poetry component contains the retained metric definitions, evaluator
runner, aggregation, bootstrap, and win-rate code used for the paper.

From the repository root:

```bash
poetry install --directory evaluation/rubrics
PYTHONPATH=. poetry --directory evaluation/rubrics run \
  python -m evaluation.rubrics.custom_metrics.run
python -m evaluation.rubrics.custom_metrics.aggregate_v2
python -m evaluation.rubrics.custom_metrics.winrate
```

API-backed scoring reads provider credentials from environment variables.
Canonical model answers are under `evaluation/model_outputs/`, and final score
tables are under `evaluation/results/`.
