# Experiment B: Metric-Human Preference Correlation Analysis

This experiment analyzes whether automated explanation quality metrics (analogy, metaphor, scaffolding, humor, jargon) correlate with human preferences in pairwise comparison tasks.

## Overview

Human annotators were shown pairs of scientific explanations (A vs B) and asked to choose which they preferred. This experiment tests whether automated metrics can predict those preferences.

**Key Question**: If explanation A scores higher than B on a metric (e.g., contains more analogies), do humans also prefer A?

## Data Flow

```
first_exp.csv + sec_exp.csv
         │
         ▼ (pivot_by_model.py)
experiment_b_merged.csv (2343 rows, 8 clusters)
         │
         ▼ (sample_for_metrics.py)
experiment_b_sampled.csv (100 per cluster = 800 rows)
         │
         ▼ (run_metrics_exp_b.py)
Metric scores in evaluation/results/preference_metrics/
         │
         ▼ (metric_correlation.py)
correlation_results.csv
         │
         ▼ (visualize_cluster_analysis.py)
combined_cluster_analysis.png + statistics CSVs
```

## Pipeline Scripts (Run in Order)

### 1. `pivot_by_model.py`
**Purpose**: Merge and clean the raw experiment data.

**What it does**:
- Merges `first_exp.csv` and `sec_exp.csv` (two annotation rounds)
- Cleans question text (removes appended answers)
- Cleans model names (removes duplicates like "model,model")
- Filters out control questions (missing sources) and same-model comparisons
- **Creates clusters**: Each unique model pair gets a cluster ID (0-7)

**Output**: `experiment_b_merged.csv` (2343 valid comparisons across 8 clusters)

### 2. `sample_for_metrics.py`
**Purpose**: Sample balanced data for metric evaluation.

**What it does**:
- Samples N examples per cluster (configurable, default=100)
- Preserves existing samples when increasing N (checkpoint-friendly)
- Ensures equal representation across model pairs

**Configuration**: Change `SAMPLES_PER_CLUSTER` at top of file

**Output**: `experiment_b_sampled.csv`

### 3. `run_metrics_exp_b.py`
**Purpose**: Run automated metrics on both explanations in each comparison.

**What it does**:
- Creates `experiment_b_eval_dataset.csv` with columns for metric evaluation
- Runs these metrics on both explanation_a and explanation_b:
  - **LLM-based**: analogy_v2, metaphor_v8, scaffolding_v2, humor_v5, jargon
  - **Readability**: flesch_kincaid, flesch_reading_ease, dale_chall, ari
- Uses preference-metric checkpoint directory for incremental processing

**Output**: Metric CSVs in `evaluation/results/preference_metrics/`

### 4. `metric_correlation.py`
**Purpose**: Calculate correlations between metrics and human preferences.

**What it does**:
- Loads metric scores and human preferences
- Computes metric differences (score_A - score_B) for each comparison
- Binarizes scores for presence/absence metrics (threshold=0.5)
- Runs two correlation methods (see detailed explanation below)

**Output**: `correlation_results.csv`

### 5. `per_cluster_analysis.py`
**Purpose**: Breakdown correlations by model-pair cluster.

**What it does**:
- Analyzes correlations separately for each of the 8 clusters
- Compares Synth DPO clusters (1,2,3) vs non-DPO clusters
- Useful for understanding if correlations vary by model type

### 6. `visualize_cluster_analysis.py`
**Purpose**: Generate comprehensive visualization.

**Output**: 
- `combined_cluster_analysis.png` - Multi-panel figure
- `cluster_statistics.csv` - Per-cluster breakdown
- `aggregate_statistics.csv` - Overall statistics
- `summary_statistics.csv` - Summary table

---

## Correlation Methods (Detailed)

### Method 1: Point-Biserial Correlation

Measures the strength of association between a continuous metric and a binary outcome (human chose A vs B).

#### For Binary Metrics (analogy, metaphor, scaffolding, humor)

These metrics measure presence/absence of a feature (score > 0.5 = present).

**Binarization and Tie Handling**:

1. **Binarize scores**: 
   - `bin_A = 1` if `score_A > 0.5`, else `0`
   - `bin_B = 1` if `score_B > 0.5`, else `0`

2. **Categorize each comparison**:
   | bin_A | bin_B | Category | Metric Prediction |
   |-------|-------|----------|-------------------|
   | 1 | 0 | Only A has feature | Predict A |
   | 0 | 1 | Only B has feature | Predict B |
   | 1 | 1 | TIE (both have) | **Excluded** |
   | 0 | 0 | TIE (neither has) | **Excluded** |

3. **Correlation on non-tie cases only**:
   ```python
   # metric_pred: 1 if only A has feature, 0 if only B has
   r, p = stats.pointbiserialr(human_pref, metric_pred)
   ```

**Why exclude ties?** If both explanations have (or lack) the feature, the metric provides no signal for differentiating them. Including ties would dilute the correlation.

**Agreement calculation**:
```python
# Of cases where metric predicts A or B, how often does human agree?
agreement = (metric_pred == human_pref).mean() * 100
```
- `N_valid`: Number of non-tie cases (where metric differentiates)
- `N_ties`: Number of tie cases (both have OR neither has feature)

#### For Continuous Metrics (jargon, readability)

No binarization needed. Use the raw difference:

```python
diff = score_A - score_B
r, p = stats.pointbiserialr(human_pref, diff)

# Agreement: if diff > 0, predict A; else predict B
metric_pred = (diff > 0).astype(int)
agreement = (metric_pred == human_pref).mean() * 100
```

### Method 2: Logistic Regression

Multivariate analysis to find which metrics best predict human preference when considered together.

```python
X = [analogy_diff, metaphor_diff, scaffolding_diff, humor_diff, jargon_diff, readability_diff]
y = human_pref (1 = chose A, 0 = chose B)

# Standardize features for comparable coefficients
X_scaled = StandardScaler().fit_transform(X)

# Fit logistic regression
lr = LogisticRegression()
lr.fit(X_scaled, y)

# 5-fold cross-validation for accuracy estimate
cv_scores = cross_val_score(lr, X_scaled, y, cv=5)
```

**Interpretation**:
- **β > 0**: Higher metric difference (A > B) → more likely to choose A
- **β < 0**: Higher metric difference → more likely to choose B
- **Odds Ratio** = exp(β): Multiplicative change in odds per 1 SD increase

---

## Clusters

| ID | Models | Type | N |
|----|--------|------|---|
| 0 | SFT vs GPT_p | Non-DPO | 100 |
| 1* | SFT_p vs SynthDPO_p | **Synth DPO** | 100 |
| 2* | GPT vs SynthDPO | **Synth DPO** | 100 |
| 3* | GPT_p vs SynthDPO_p | **Synth DPO** | 100 |
| 4 | SFT_p vs GPT_p | Non-DPO | 100 |
| 5 | SFT_p vs OrgSFT_p | Non-DPO | 100 |
| 6 | SFT vs Vanilla_p | Non-DPO | 100 |
| 7 | GPT_p vs Human | Non-DPO | 100 |

**Model name legend**:
- SFT = SciComma-3.1-8B_y (supervised fine-tuned)
- SFT_p = SciComma-3.1-8B_prompt (SFT with prompt)
- GPT = gpt-3.5-turbo-0125
- GPT_p = gpt-3.5-turbo-0125_cot (with chain-of-thought prompt)
- SynthDPO = scicomma-3.1-dpo (DPO trained on synthetic preferences)
- SynthDPO_p = scicomma-3.1-dpo_prompt (DPO with prompt)
- OrgSFT_p = organic_SFT_prompted
- Vanilla_p = vanilla_prompted
- Human = human_answers

---

## Output Files

### Data Files
| File | Description |
|------|-------------|
| `experiment_b_merged.csv` | All valid comparisons with cluster IDs |
| `experiment_b_sampled.csv` | Sampled subset for metric evaluation |
| `experiment_b_eval_dataset.csv` | Formatted for metric runner |
| `correlation_results.csv` | Full results with metric diffs and human prefs |
| `cluster_statistics.csv` | Per-cluster correlations and agreements |
| `aggregate_statistics.csv` | All/DPO/Non-DPO aggregate stats |
| `summary_statistics.csv` | Summary table (one row per metric) |

### Visualizations
| File | Description |
|------|-------------|
| `combined_cluster_analysis.png` | Multi-panel analysis figure |
| `30example_combined_cluster_analysis.png` | Same analysis with N=30/cluster |

---

## Quick Start

```bash
# 1. Merge and prepare data
python pivot_by_model.py

# 2. Sample for metrics (edit SAMPLES_PER_CLUSTER if needed)
python sample_for_metrics.py

# 3. Run metrics (requires OpenAI API key)
python run_metrics_exp_b.py

# 4. Analyze correlations
python metric_correlation.py

# 5. Per-cluster breakdown
python per_cluster_analysis.py

# 6. Generate visualization
python visualize_cluster_analysis.py
```

## Publication rerun commands

These scripts read canonical tracked outputs and print results without mutating
the publication CSVs when invoked as documented:

```bash
# Teacher robustness / composition tables (stdout only)
python human_study/preferences/teacher_significance.py

# Experiment B continuous regression with formality covariate
python human_study/preferences/logistic_regression.py \
  --mode continuous \
  --with-formality \
  --output /tmp/logistic_regression_continuous_with_formality.csv
```

Compare regenerated outputs against the canonical CSVs under
`human_study/preferences/data/` rather than writing back into that directory.

---

## Key Findings (100 examples/cluster)

| Metric | All r | Agreement | N_valid | Interpretation |
|--------|-------|-----------|---------|----------------|
| analogy_v2 | -0.026 | 48.6% | ~350 | No correlation |
| metaphor_v8 | **-0.146** | 42.7% | ~350 | Weak negative |
| scaffolding_v2 | 0.015 | 50.8% | ~350 | No correlation |
| humor_v5 | N/A | 0% | 0 | All ties (rare feature) |
| jargon | 0.049 | 52.0% | 800 | No correlation |

**Interpretation**: The automated metrics show weak or no correlation with human preferences. The negative metaphor correlation suggests humans may slightly prefer explanations with *fewer* metaphors, contrary to what the metric assumes is "good."

---

## Notes

- **Ties dominate**: For binary metrics, most comparisons are ties (both have or neither has the feature), reducing N_valid significantly.
- **Cluster variation**: Correlations vary by cluster. Cluster 7 (GPT_p vs Human) shows positive correlations, while cluster 3 (GPT_p vs SynthDPO_p) shows strong negative correlations.
- **Sample size matters**: Results from 30 vs 100 examples per cluster differ substantially—larger samples give more stable estimates.

