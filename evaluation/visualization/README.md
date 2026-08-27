# Rubric visualization

Plotting utilities for science-communication rubric scores. The primary
publication figures (`fig:stacked_scores`, metric heatmaps) are produced by
`evaluation/rubrics/custom_metrics/aggregate_v2.py` when run from the repository
root:

```bash
python -m evaluation.rubrics.custom_metrics.aggregate_v2 \
  --output-dir /tmp/rubric_aggregations_v2 \
  --bootstrap-dir /tmp/rubric_bootstrap
```

`plot_scores.py` contains additional exploratory plotting helpers used during
metric development. Canonical score tables live under
`evaluation/results/rubric_scores/`.

Pass both `--output-dir` and `--bootstrap-dir` under `/tmp` when rerunning so
tracked `aggregations_v2/` plots/CSVs and `bootstrap/` confidence intervals are
not overwritten.
