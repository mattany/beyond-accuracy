# Paper artifact map

This manifest records the publication scope of the cleaned repository.
The retention test is whether an artifact supports a method, dataset, figure,
table, appendix result, or reproducibility step in the ACL paper. All paths
below use the final public layout. See the root `README.md` for runnable
commands and reproducibility gaps (DPO training, GPT-5.2 claim-level
factuality).

## Path relocations (pre-cleanup → final)

| Former location | Final location |
|-----------------|----------------|
| `SFT/batch_file_gen/` | `training/data_generation/` |
| `SFT/training/GPT_SFT_only.ipynb` | `training/sft/GPT_SFT_only.ipynb` |
| `DPO/preference_dataset_generation.py` | `training/dpo/preference_dataset_generation.py` |
| `Benchmarking/truthfulness/` | `training/model_variants/` |
| `Benchmarking/deep_eval/` (rubric code) | `evaluation/rubrics/` |
| `Benchmarking/eval_dataset_generation/` | `evaluation/model_generation/` |
| `scripts/experiment_b/` | `human_study/preferences/` |
| `scripts/judge_alignment/` | `human_study/judge_validation/` |
| `trust_llm/` | `evaluation/factuality/trust_llm/` |

Removed roots (`RAG/`, `rebuttal/`, `science-QA_jsonl/`, `Benchmarking/baram_tsabari/`,
obsolete judge-validation dataset versions, and IDE/OS artifacts) are not listed
here; they fail the publication retention test and are absent from the final
tree.

## QA-Pairs and teacher datasets

- `data/qa_pairs/` — human, GPT-3.5, GPT-5, and Kimi source CSVs;
  train/validation/test splits; input batches; provider outputs; `create_split.py`;
  `load_datasets.ipynb`. Supports QA-Pairs, synthetic preference pairs, Appendix
  `sec:appendix_sft_prompt`, and Appendix `sec:appendix_teacher`.
- `training/data_generation/` — active teacher batch generators (relocated from
  `SFT/batch_file_gen/`). Archive residue and batch job logs were removed.

## Model training

- `training/sft/GPT_SFT_only.ipynb` — Human/Synthetic SFT (relocated from
  `SFT/training/`). Supports `sec:model_evaluation` and Appendix
  `sec:appendix_deployment`.
- `training/dpo/preference_dataset_generation.py` — preference-pair construction
  only (relocated from `DPO/`). **Not** an end-to-end DPO trainer.
- `training/model_variants/` — non-RAG model generation and Hub upload scripts
  (relocated from `Benchmarking/truthfulness/`).
- **DPO reproducibility gap:** no tracked `DPOTrainer` / `DPOConfig`
  implementation. End-to-end Human and Synthetic DPO training is external; the
  retained generator documents preference-data construction only.
- GPT-5 and Kimi QA-Pairs outputs support `tab:teacher_robustness` and
  `tab:teacher_composition`.

## Rubric evaluation and model outputs

- `evaluation/rubrics/` — production metric definitions, runner, aggregation,
  bootstrap, win-rate code, and evaluation-dataset generation.
- `evaluation/results/rubric_scores/` — canonical final rubric metric CSVs (run 9).
- `evaluation/results/rubric_scores/.checkpoints/` — run-9 resume state for
  `custom_metrics/run.py` (per-model JSON checkpoints).
- `evaluation/results/preference_metrics/` — Experiment B metric CSVs (run 10).
- `evaluation/results/preference_metrics/.checkpoints/` — run-10 resume state for
  Experiment B metric scoring.
- `evaluation/model_outputs/main/`, `evaluation/model_outputs/dpo_variants/`,
  `evaluation/model_outputs/human_variants/` — canonical generation tables.
- `evaluation/model_outputs/scillama3/` — retained conservatively (different model
  columns; 90/91 question overlap with canonical tables).
- `evaluation/results/rubric_scores/wilcoxon_tests.py` — significance tests.
- `evaluation/results/preference_metrics/regression_metrics_merged.csv` — merged
  Experiment B regression inputs.
- Removed: runs 3–8, archived test data, consistency experiments, RAG evaluation,
  Ollama failure investigations, `consistency_check.py`, `visualize_consistency.py`,
  `aggregate_bootstrap.py`, superseded `evaluation/visualization/images/plots/run_5/`.

## Judge validation

- `human_study/judge_validation/balanced_dataset_v2_human/` — final validation
  dataset, Label Studio export, derived CSVs and plots.
- `human_study/judge_validation/tie_breaker_v2/` — third-author adjudication trail.
- `human_study/judge_validation/intercoder_reliability.py` — reliability and
  human–LLM correlation analysis (requires `--output-dir`).
- `human_study/judge_validation/labeling_interface/labelstudio_v2.xml` and
  `labelstudio_tiebreaker_v2.xml` — interfaces for retained exports.
- Label Studio JSON exports are anonymized (`annotator_N` labels); derived
  numerical CSV outputs remain publication-identical.
- API-required prep utilities outside the publication rerun set:
  `analyze_balance_options.py`, `run_connection_reasons.py`,
  `fetch_reddit_formatting.py`.
- Removed obsolete pipelines: `balanced_dataset/`, `balanced_dataset_v2/`,
  `tie_breaker_dataset/`, `add_coder_labels.py`, `boolean_metrics.py`, and
  associated one-off disagreement runners.

## Human preference study

- `human_study/preferences/data/` — raw anonymized study CSVs (`first_exp.csv`,
  `sec_exp.csv`), processed tables, formality scores, normalization metadata,
  `dpo_rubric_up_pref_down.csv`.
- `human_study/preferences/teacher_significance.py` — teacher robustness /
  composition significance (stdout only).
- `human_study/preferences/logistic_regression.py`,
  `metaphor_overoptimization.py`, and metric-prep scripts — Experiment B
  analysis.
- `evaluation/results/preference_metrics/` — metric inputs for
  `tab:exp_b_regression` and `sec:rubric_preference_alignment`.

## Factuality

- `evaluation/factuality/truthfulqa_results/` — per-model TruthfulQA
  checkpoints and `truthfulqa_summary_latest.csv` for `tab:truthfulqa_results`.
- `evaluation/factuality/truthfulqa_visualization.py` — offline table/figure
  regeneration from checkpoints.
- `evaluation/factuality/truthfulqa_benchmark.py` / `.ipynb` — Colab-oriented
  checkpoint regeneration (GPU + `HF_TOKEN`).
- `evaluation/factuality/trust_llm/` — auxiliary TrustLLM benchmark (not GPT-5.2
  claim-level provenance).
- **GPT-5.2 claim-level gap:** no tracked script or result file for Appendix
  `tab:factuality` atomic-claim analysis.
- Human-study accuracy ratings remain in Experiment B data (second factuality
  check).

## Paper-to-artifact coverage

- `tab:dataset_stats`, `sec:datasets`, `sec:appendix_sft_prompt`:
  `data/qa_pairs/`, `training/data_generation/`, `training/sft/`, and
  `training/dpo/` (preference construction only; DPO training external).
- `fig:stacked_scores`, `sec:exp_a_results`: `evaluation/model_outputs/`,
  `evaluation/rubrics/`, `evaluation/results/rubric_scores/`.
- `fig:validation_a`–`fig:validation_c`, `sec:metrics_validation`:
  `human_study/judge_validation/balanced_dataset_v2_human/`,
  `human_study/judge_validation/tie_breaker_v2/`, Label Studio interfaces,
  `intercoder_reliability.py`.
- `tab:human_preferences`, `tab:exp_b_regression`, `sec:rubric_preference_alignment`:
  `human_study/preferences/data/`, `evaluation/results/preference_metrics/`,
  final regression outputs, metaphor-overoptimization inputs.
- `tab:truthfulqa_results`: `evaluation/factuality/truthfulqa_results/`,
  `truthfulqa_visualization.py`.
- `tab:factuality`: participant-rating data present; GPT-5.2 claim-level artifact
  absent. `evaluation/factuality/trust_llm/` is auxiliary only.
- `tab:winrate`: `evaluation/results/rubric_scores/`,
  `evaluation/rubrics/custom_metrics/winrate.py`.
- `tab:teacher_robustness`, `tab:teacher_composition`: GPT-5/Kimi QA-Pairs data,
  `evaluation/results/rubric_scores/`, `human_study/preferences/teacher_significance.py`.
- `sec:appendix_deployment`: `training/sft/GPT_SFT_only.ipynb` and documented
  model artifacts on Hugging Face.
