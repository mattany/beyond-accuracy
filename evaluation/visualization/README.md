# Rubric visualization

Plotting utilities for science-communication rubric scores. The primary
publication figures (`fig:stacked_scores`, metric heatmaps) are produced by
`evaluation/rubrics/custom_metrics/aggregate_v2.py` when run from the repository
root:

```bash
python -m evaluation.rubrics.custom_metrics.aggregate_v2
```

`plot_scores.py` contains additional exploratory plotting helpers used during
metric development. Canonical score tables live under
`evaluation/results/rubric_scores/`.
