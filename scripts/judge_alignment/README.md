# Inter-coder Reliability Analysis

This script analyzes agreement between human annotators and compares human labels with LLM-generated scores.

## Usage

```bash
python intercoder_reliability.py path/to/export.json
```

Output CSVs and plots are saved to the same directory as the input JSON file.

## Output Files

### `intercoder_reliability.csv`

Measures agreement among human annotators for each metric.

| Column | Description |
|--------|-------------|
| `metric` | The metric being evaluated (Analogy, Metaphor, Humor, Connection) |
| `percent_agreement` | Proportion of items where **all** annotators gave the same label (0 or 1). Range: 0–1 |
| `kappa` | Fleiss' kappa coefficient, measuring agreement beyond chance. Range: -1 to 1 (0 = chance, 1 = perfect) |
| `ac1` | Gwet's AC1 coefficient, an alternative that handles skewed distributions better. Range: -1 to 1 |
| `P_e_kappa` | Expected agreement by chance (Fleiss' kappa formula: p₀² + p₁²) |
| `P_e_ac1` | Expected agreement by chance (Gwet's AC1 formula: 2·p₀·p₁) |

**Fleiss' Kappa vs Gwet's AC1:**

Fleiss' kappa can give misleadingly low values when label distributions are skewed (the "prevalence paradox"). For example, if annotators all correctly agree that humor is rare (~85% "no"), kappa penalizes this because high agreement on a rare category is "expected by chance."

Gwet's AC1 addresses this by using a different chance formula:
- **Kappa's P_e** = p₀² + p₁² → High when skewed (e.g., 0.85² + 0.15² = 0.74)
- **AC1's P_e** = 2·p₀·p₁ → Low when skewed (e.g., 2·0.85·0.15 = 0.26)

This means AC1 gives annotators credit for agreeing on the base rate, while kappa does not.

**Interpretation of Kappa/AC1:**
- < 0: Less than chance agreement
- 0.01–0.20: Slight agreement
- 0.21–0.40: Fair agreement
- 0.41–0.60: Moderate agreement
- 0.61–0.80: Substantial agreement
- 0.81–1.00: Almost perfect agreement

### `human_llm_corr_mean.csv`

Correlations between the **mean** of human annotator labels and LLM scores.

| Column | Description |
|--------|-------------|
| `metric` | The metric being evaluated |
| `pearson_mean` | Pearson correlation (linear relationship). Range: -1 to 1 |
| `spearman_mean` | Spearman correlation (rank-based, monotonic relationship). Range: -1 to 1 |
| `kendall_mean` | Kendall's tau (rank-based, more robust to outliers). Range: -1 to 1 |

**Logic:** For each item, the human labels (0 or 1 from each annotator) are averaged to produce a continuous value (e.g., 0, 0.33, 0.67, or 1 for 3 annotators). This mean is then correlated with the LLM's score.

### `human_llm_corr_majority.csv`

Correlations between the **majority-vote** human label and LLM scores.

| Column | Description |
|--------|-------------|
| `metric` | The metric being evaluated |
| `pearson_majority` | Pearson correlation (linear relationship). Range: -1 to 1 |
| `spearman_majority` | Spearman correlation (rank-based). Range: -1 to 1 |
| `kendall_majority` | Kendall's tau (rank-based, more robust). Range: -1 to 1 |

**Logic:** For each item, a majority-vote label is computed: if at least 2 out of 3 annotators labeled the item as 1, the majority label is 1; otherwise it's 0. This binary label is then correlated with the LLM's score.

**When to use each correlation:**
- **Pearson**: Assumes linear relationship; sensitive to outliers
- **Spearman**: Captures monotonic relationships; more robust than Pearson
- **Kendall's tau**: Most robust to outliers and ties; preferred for small samples

## Metrics Evaluated

| Metric | Description |
|--------|-------------|
| **Analogy** | Explanation uses an analogy |
| **Metaphor** | Explanation uses a metaphor |
| **Humor** | Explanation incorporates humor |
| **Connection** | Explanation connects to everyday life |

## Data Requirements

The input JSON should be a Label Studio export containing:
- `data.question_id`: Unique question identifier
- `data.model`: Model name
- `data.analogy_explicit_score`: LLM score for analogy
- `data.metaphor_explicit_score`: LLM score for metaphor
- `data.humor_explicit_score`: LLM score for humor
- `data.connection_to_everyday_life_score`: LLM score for connection
- `annotations`: List of annotator responses with choices for each metric

Only items with annotations from **all** annotators are included in the analysis.

## Output Plots

### `intercoder_reliability.png`

A grouped bar chart showing:
- **Percent Agreement** (blue): Proportion of items where all human annotators gave the same label
- **Fleiss' Kappa** (orange): Agreement beyond chance (can be misleadingly low for skewed distributions)
- **Gwet's AC1** (green): Agreement beyond chance, robust to skewed distributions

### `human_llm_correlations.png`

A two-panel grouped bar chart showing correlations between human labels and LLM scores:
- **Left panel**: Mean human label correlations
- **Right panel**: Majority-vote human label correlations

Each panel shows three correlation types:
- **Pearson** (blue): Linear correlation
- **Spearman** (green): Rank-based correlation
- **Kendall's τ** (orange): Robust rank-based correlation

