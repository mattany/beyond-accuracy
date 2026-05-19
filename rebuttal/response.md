Thank you for your clarification.
1. To address the bias that the length variable introduces, we added log(answer length) as a variable to our regression model.
   Recalculating the regressions shown in the paper (with and without prompted DPO) showed no negative changes in significance of the previous results. In addition, readability gained significance for the “No P-DPO” regression.

2. Indeed, the dataset that we used for the human answers removed formatting including capitalization and newline characters. In light of the valid concern you raise, in order to mitigate bias introduced by formatting, we ran another regression test on the other seven clusters - all model generated.
   The regression results show that answer length and readability gained significance, while the rest of the results remain stable.

We will add the length variable to the regression table of the final manuscript, and add a clarification about the formatting limitation of the human generated answers.
The updated results table is provided below.

| Metric | Published (Full) | New (Full) | Published (No P-DPO) | New (No P-DPO) | No Human (N=700) |
|---|---|---|---|---|---|
| Metaphor | -0.59**, SE 0.22 | -0.58**, SE 0.22 | -0.26, SE 0.25 | -0.26, SE 0.25 | -0.61**, SE 0.23 |
| Analogy | -0.05, SE 0.19 | -0.01, SE 0.19 | -0.08, SE 0.21 | -0.05, SE 0.21 | -0.15, SE 0.21   |
| Scaffolding | 0.59**, SE 0.19 | 0.64**, SE 0.20 | 0.53*, SE 0.23 | 0.58*, SE 0.23 | 0.63**, SE 0.21  |
| Jargon | 0.83, SE 0.67 | 0.85, SE 0.68 | 0.88, SE 0.75 | 0.90, SE 0.75 | 0.87, SE 0.73    |
| Readability | 0.82*, SE 0.37 | 0.86*, SE 0.37 | 0.80, SE 0.41 | 0.83*, SE 0.41 | 1.17**, SE 0.41  |
| Answer Length | — | -0.35*, SE 0.14 | — | -0.25, SE 0.15 | -0.58**, SE 0.19 |
