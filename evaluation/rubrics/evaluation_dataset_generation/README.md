# Dataset Generation Process

This document details the generation process of two datasets used for evaluating explanation quality metrics: a balanced binary metrics dataset and a side-by-side comparison dataset.

## Overview

We generated two distinct datasets from our base evaluation dataset to facilitate different aspects of metric evaluation:
1. A balanced dataset of individual model responses with binary metric scores
2. A comparative dataset of paired model responses with contrasting metric performance

## Dataset Generation Methodology

### Boolean Metrics Dataset

The first dataset was generated to ensure balanced representation of high and low-performing model responses across four key metrics: humor, metaphor usage, analogy usage, and connection to everyday life. The generation process followed these steps:

1. **Metric Binarization**: 
   - Each continuous metric score was converted to a binary value using a threshold of 0.5. Scores ≥ 0.5 were mapped to 1 (presence of the feature) & vice versa.
2. **Grade Calculation**:
   - For each model response, a composite grade was calculated as the sum of its binary metric scores, resulting in a grade range of 0-4, representing how many metrics were present in each response.
3. **Balanced Sampling**:
   - Responses were divided into two groups based on the median grade. 15 responses were randomly sampled from above the median. 15 responses were randomly sampled from below or equal to the median.

### Side-by-Side Comparison Dataset

The second dataset was designed for comparative analysis between base models and their fine-tuned counterparts. The generation process was as follows:

1. **Model Pair Definition**:
   Three specific model pairs were considered:
   - llama-2-7b vs. SciComma-2-7b
   - llama3.1-instruct vs. SciComma-3.1-8B
   - llama-3.3-70b vs. SciComma-3.3-70B

2. **Valid Pair Identification**:
   - For each question in the evaluation dataset, all model pairs were evaluated
   - A pair was considered valid if their binary metric grades (sum of binary scores) differed
   - This ensured that paired comparisons would have a clear distinction in metric performance

3. **Random Sampling**:
   - 30 question-pair combinations were randomly selected from the pool of valid pairs
   - If fewer than 30 valid pairs were found, all available pairs were used
   - The order of models within each pair was randomized to prevent position bias

4. **Metric Inclusion**:
   For each pair, both raw and binary scores were included for all four metrics:
   - Raw continuous scores (original metric values)
   - Binary scores (thresholded at 0.5)
   - Total grade (sum of binary scores)

## Dataset Properties

### Boolean Metrics Dataset
- Total samples: 30
- Balanced distribution: 15 high-grade and 15 low-grade responses
- Includes all model types
- Contains both raw and binary metric scores

### Side-by-Side Dataset
- Total pairs: 30 (or maximum available)
- Each pair contains:
  - Two different model responses (one base, one fine-tuned)
  - Different total grades between pairs
  - Randomized model order
  - Complete metric scores (raw and binary)

## Reproducibility

Both dataset generation processes use a fixed random seed (42) to ensure reproducibility. This affects:
- Random sampling of responses
- Selection of model pairs
- Order randomization within pairs
- Final dataset shuffling

The retained generation utilities now consume the canonical model-answer table
at `evaluation/model_outputs/main/all_models_joined.csv`. Pair generation writes:

- `evaluation/model_outputs/main/comparison_tasks.csv`
- `evaluation/model_outputs/main/model_pairs_metadata.csv`

Provider-backed scripts update or create model-answer CSVs beside the canonical
table. Credentials are read from environment variables through
`evaluation.rubrics.settings`.