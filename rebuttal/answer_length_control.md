# Regression Results: Controlling for Answer Length

## Comparison Table

| Metric | Published (Full) | New (Full) | Published (No P-DPO) | New (No P-DPO) |
|---|---|---|---|---|
| Metaphor | -0.59**, SE 0.22 | -0.58**, SE 0.22 | -0.26, SE 0.25 | -0.26, SE 0.25 |
| Analogy | -0.05, SE 0.19 | -0.01, SE 0.19 | -0.08, SE 0.21 | -0.05, SE 0.21 |
| Scaffolding | 0.59**, SE 0.19 | 0.64**, SE 0.20 | 0.53*, SE 0.23 | 0.58*, SE 0.23 |
| Jargon | 0.83, SE 0.67 | 0.85, SE 0.68 | 0.88, SE 0.75 | 0.90, SE 0.75 |
| Readability | 0.82*, SE 0.37 | 0.86*, SE 0.37 | 0.80, SE 0.41 | 0.83*, SE 0.41 |
| Answer Length | — | -0.35*, SE 0.14 | — | -0.25, SE 0.15 |

Significance: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05

## Summary of Changes

- **Metaphor, Jargon, Analogy**: Virtually unchanged in both columns. All significance levels identical.
- **Scaffolding**: Coefficient slightly increases in both columns (0.59→0.64, 0.53→0.58). Significance unchanged (\*\*, \*).
- **Readability**: Coefficient slightly increases in both columns (0.82→0.86, 0.80→0.83). Full column stays \*. No P-DPO column **gains** significance (was n.s., now p=0.043\*).
- **Answer Length**: New control variable (log-transformed word count difference). Significant in Full (p=0.017\*), marginal in No P-DPO (p=0.088). Negative coefficient means longer answers are slightly less preferred.
- **Standard errors**: Essentially unchanged across the board, confirming the new variable doesn't introduce collinearity issues.

## Interpretation

Adding answer length as a control variable does not change any of the original findings. All previously significant effects remain significant at the same level, and the readability effect in the No P-DPO subset actually strengthens. This indicates that the published results are robust to controlling for answer length.
