# Regression Results: Excluding Human-Answer Cluster

Cluster 7 (Human vs GPT_cot) is the only cluster where one of the two explanations is human-written rather than model-generated. We re-ran the logistic regression excluding this cluster to check whether the findings hold when restricted to model-vs-model comparisons only.

## Comparison Table

| Metric | All Data (N=800) | No Human (N=700) |
|---|---|---|
| Metaphor | -0.58**, SE 0.22 | -0.61**, SE 0.23 |
| Analogy | -0.01, SE 0.19 | -0.15, SE 0.21 |
| Scaffolding | 0.64**, SE 0.20 | 0.63**, SE 0.21 |
| Jargon | 0.85, SE 0.68 | 0.87, SE 0.73 |
| Readability | 0.86*, SE 0.37 | 1.17**, SE 0.41 |
| Answer Length | -0.35*, SE 0.14 | -0.58**, SE 0.19 |

Significance: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05

Both regressions include answer length (log-transformed word count difference) as a control variable.

## Findings

- **Scaffolding and Metaphor are stable.** Both retain the same significance level (\*\*) and similar coefficient magnitudes, confirming these effects are not driven by the human-answer cluster.
- **Readability strengthens.** The coefficient increases from 0.86 to 1.17 and gains a significance level (from \* to \*\*; p = 0.020 → 0.004). Readability differences matter more when comparing model-generated answers to each other, likely because the human cluster (where human answers tend to be less formulaic) adds noise.
- **Answer length strengthens.** The coefficient increases in magnitude from -0.35 to -0.58 and gains a significance level (from \* to \*\*; p = 0.017 → 0.002). The length penalty is driven by model-vs-model comparisons rather than the human cluster.
- **Analogy and Jargon remain non-significant** in both conditions.

## Interpretation

Excluding the human-answer cluster does not weaken any of the original findings. If anything, the model-only subset shows cleaner effects: readability and answer length both become more significant. This suggests that the human-written answers in cluster 7, which differ from model outputs in ways not fully captured by the rubric metrics, add variance that slightly dilutes the measured effects.
