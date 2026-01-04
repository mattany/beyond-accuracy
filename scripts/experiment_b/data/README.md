# Logistic Regression Input Data

This zip file contains all data needed to reproduce the logistic regression analysis predicting human preference from metric scores.

## Files

### Metric Scores (from run_10)
| File | Description |
|------|-------------|
| `metaphor_v8.csv` | Metaphor usage scores (LLM-judged, 0-1) |
| `analogy_v2.csv` | Analogy usage scores (LLM-judged, 0-1) |
| `scaffolding_v2.csv` | Scaffolding scores (LLM-judged, 0-1) |
| `jargon.csv` | Jargon avoidance scores (0-1, higher = less jargon) |
| `flesch_reading_ease.csv` | Flesch Reading Ease (0-1 normalized) |
| `flesch_kincaid.csv` | Flesch-Kincaid Grade Level |
| `ari.csv` | Automated Readability Index |
| `dale_chall.csv` | Dale-Chall Readability Score |

### Human Preference Data
| File | Description |
|------|-------------|
| `experiment_b_eval_dataset.csv` | 800 pairwise comparisons with human choices |

Columns:
- `question`: The science question
- `explanation_a`: First explanation
- `explanation_b`: Second explanation
- `comparison_id`: Unique identifier (0-799)
- `human_choice`: "Explanation A" or "Explanation B"
- `cluster`: Which model comparison cluster (0-4)
- `model_a`, `model_b`: Model names

### Configuration
| File | Description |
|------|-------------|
| `normalization_ranges.json` | Min/max ranges for normalizing continuous metrics |
| `logistic_regression.py` | The regression analysis script |

## Cluster Definitions

Each cluster compares two models with 200 comparisons:

| Cluster | Model A | Model B | Description |
|---------|---------|---------|-------------|
| 0 | SFT | SynthDPO | Base SFT vs Synthetic DPO |
| 1 | SFT_p | SynthDPO_p | Prompted SFT vs Prompted Synthetic DPO |
| 2 | GPT | SynthDPO | GPT-4o vs Synthetic DPO |
| 3 | GPT_p | SynthDPO_p | Prompted GPT vs Prompted Synthetic DPO |
| 4 | SFT | GPT | Base SFT vs GPT-4o |

**Note:** `--exclude-prompted-dpo` excludes clusters 1 and 3 (prompted models vs SynthDPO_p).

## Metric Categories

### Binary LLM Metrics
Binarized at threshold 0.5 in "binarized" mode:
- `metaphor_v8`
- `analogy_v2`
- `scaffolding_v2`

### Continuous-Only Metrics
Never binarized, always normalized:
- `jargon`

### Readability Metrics
Normalized using ranges in `normalization_ranges.json`, then averaged into single `readability` predictor:
- `flesch_reading_ease`
- `flesch_kincaid` (inverted: lower grade = better)
- `ari` (inverted: lower grade = better)
- `dale_chall` (inverted: lower = better)

## Model Types

### Difference Model (default)
```
A_preferred = β₁*(metric_a - metric_b) + β₂*(metric_a - metric_b) + ...
```
Asks: Does having MORE of a quality than B predict preference for A?

### Interaction Model
```
A_preferred = β₁*metric_a + β₂*metric_a*(metric_a - metric_b) + ...
```
Asks: Does A's absolute level matter? Does the benefit of beating B depend on A's level?

## Running the Analysis

```bash
# Difference model, both binarized and continuous
python logistic_regression.py --mode both

# Interaction model
python logistic_regression.py --mode both --model-type interaction

# Compare All vs No-DPO (continuous only, single figure)
python logistic_regression.py --mode continuous --model-type interaction --compare-dpo

# Exclude readability metrics
python logistic_regression.py --mode both --no-readability

# Exclude prompted-DPO clusters
python logistic_regression.py --mode both --exclude-prompted-dpo
```

## Output

- CSV files with coefficients, standard errors, p-values, odds ratios, 95% CIs
- PNG figures with formatted regression tables
