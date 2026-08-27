# Paper artifact map

This manifest freezes the publication scope before files are moved or removed.
The retention test is whether an artifact supports a method, dataset, figure,
table, appendix result, or reproducibility step in the ACL paper. Paths below
refer to the pre-cleanup tree; Tasks 3--7 apply the stated moves.

## QA-Pairs and teacher datasets
- Keep and move `SFT/data/` to `data/qa_pairs/`.
- Keep current generators from `SFT/batch_file_gen/`.
- Remove `SFT/batch_file_gen/archive/`, batch status/job logs, and local setup residue.
- Keep the human, GPT-3.5, GPT-5, and Kimi source CSVs, train/validation/test
  splits, input batches, provider outputs, and `create_split.py`; together they
  support QA-Pairs, the synthetic preference pairs, the generation prompt in
  Appendix `sec:appendix_sft_prompt`, and the second-teacher experiment in
  Appendix `sec:appendix_teacher`.
- Keep `SFT/data/load_datasets.ipynb` as dataset inspection/provenance support.
- Treat `SFT/data/example_batch_test/` as test residue to remove after confirming
  that the active generators do not consume it.
- Remove root prompt copies (`prompts_0.csv`, `prompts_copy.csv`, and
  `prompts_original.csv`) if the Task 3 hash/content comparison confirms that
  retained `SFT/data/` sources are canonical.

## Model training
- Keep `SFT/training/GPT_SFT_only.ipynb`.
- Keep `DPO/preference_dataset_generation.py` only as the preference-dataset
  preparation step; it is not an end-to-end DPO training procedure.
- Keep the non-RAG model-variant scripts in `Benchmarking/truthfulness/` where they document paper model generation or publication.
- **DPO reproducibility gap:** a repository-wide tracked-file search found no
  `DPOTrainer` or `DPOConfig` implementation. End-to-end Human and Synthetic
  DPO training is therefore a missing or external reproducibility stage; the
  retained generator establishes preference-data construction only, and the
  repository must not imply that it reproduces DPO optimization.
- The retained SFT notebook supports the Human/Synthetic SFT methods in
  `sec:model_evaluation`; the preference generator supports construction of DPO
  inputs; the canonical outputs support the QA-Pairs win-rate analysis in
  `tab:winrate` and organic-model comparisons in `tab:human_preferences`.
- Keep the GPT-5 and Kimi training inputs and outputs identified above because
  `tab:teacher_robustness` and `tab:teacher_composition` rerun the same SFT
  procedure with alternate teachers.
- Remove model-generation code only when it is RAG-specific. Preserve notebooks,
  scripts, dependency metadata, and model-output inputs needed to document
  publication model generation.

## Rubric evaluation and model outputs
- Keep non-RAG code under `Benchmarking/deep_eval/`.
- Keep canonical final rubric outputs `Benchmarking/deep_eval/data/run_9/`.
- Keep Experiment B metric outputs `Benchmarking/deep_eval/data/run_10/`.
- Keep final generation outputs under `scripts/generations/`, `scripts/generations_2/`, and `scripts/generations_3/`.
- Remove runs 3–8, archived/test data, model-metric exploratory data, consistency experiments, RAG evaluation, and Ollama failure investigations.
- Keep the production metric definitions, prompts, aggregation and bootstrap
  code, win-rate code, evaluation-dataset generation, dependency metadata, and
  visualization code needed to regenerate `fig:stacked_scores`, the rubric
  results in `sec:exp_a_results`, and Appendix `tab:winrate`.
- Keep both legacy and final-version metric CSVs inside `run_9/` because the
  final aggregate, teacher-robustness, and win-rate analyses select named metric
  versions from that canonical run. Keep `run_9/wilcoxon_tests.py` and its
  outputs.
- Keep all metric CSVs and `regression_metrics_merged.csv` in `run_10/`; the
  Experiment B regression loads analogy, metaphor, scaffolding, jargon, and
  four readability metrics from this run for `tab:exp_b_regression`.
- **Unresolved verification item (Task 5):** conservatively retain
  `run_10/.checkpoints/` until Task 5 determines whether it is required to
  resume the retained metric run. Remove it as operational residue only after
  that dependency check is recorded.
- Remove `Benchmarking/deep_eval/custom_metrics/consistency_check.py`,
  `visualize_consistency.py`, `aggregate_bootstrap.py`, and `bootstrap.py` when
  their only live references are to deleted consistency data or runs 7--8.
  Preserve `aggregate.py`, `aggregate_v2.py`, `rerun_readability.py`, `run.py`,
  `winrate.py`, metric classes, and their package/configuration dependencies.
- Remove `Benchmarking/visualization/images/plots/run_5/` and `images.zip` as
  superseded visualization output; retain the plotting source so canonical
  `run_9/` results can be visualized after Task 5 updates its paths.
- **Unresolved verification item (Task 5):** conservatively retain
  `Benchmarking/deep_eval/scillama3/` until Task 5 compares it with canonical
  generation outputs and confirms that no retained evaluator consumes it.
  Remove it only after that verification is recorded.

## Judge validation
- Keep final validation code and the final `balanced_dataset_v2_human/` and `tie_breaker_v2/` artifacts.
- Keep Label Studio interfaces required to reproduce those exports.
- Remove superseded balanced-dataset versions, disagreement experiments, and one-off obsolete metric runners after confirming the final scripts do not consume them.
- Keep `scripts/judge_alignment/intercoder_reliability.py`; it consumes a
  caller-supplied Label Studio JSON export and writes the reliability,
  human--LLM correlation, and plots beside that export. Point the documented
  final command at `balanced_dataset_v2_human/labelstudio_output.json`.
- Keep `scripts/judge_alignment/apply_tiebreaker_to_formatted_csvs.py`,
  `tie_breaker_v2/`, and `labeling_interface/labelstudio_tiebreaker_v2.xml` as
  the adjudication trail for the paper's two-primary-annotator plus third-author
  protocol in `sec:metrics_validation`.
- Keep the complete contents of `balanced_dataset_v2_human/` and
  `tie_breaker_v2/`, including source/formatted CSVs, Label Studio exports,
  metric-bearing intermediates, and final CSV/plot outputs used by the
  validation figures and appendix annotation tables.
- Keep the Label Studio XML files needed to interpret retained exports. Task 6
  may remove interfaces for discarded dataset versions after matching each
  retained export to its interface.
- `add_coder_labels.py`, `reorder_columns.py`, `generate_v2_dataset.py`, and
  `intercoder_reliability_v2.py` explicitly consume superseded
  `balanced_dataset/` or `balanced_dataset_v2/` paths. Treat those scripts and
  paths as one obsolete pipeline and remove them together; they are not
  dependencies of the retained final `intercoder_reliability.py` command.
- Remove `boolean_metrics.py` because it consumes deleted `run_6/` and
  `test_data/` artifacts. Remove one-off disagreement and metric-development
  runners (`extract_disagreements.py`, `merge_metaphor_v8_with_v6_labels.py`,
  `run_humor_*`, `run_metaphor_metric_on_disagreements.py`, and associated
  root-level disagreement CSVs) after Task 6 confirms no retained final command
  imports them.
- Remove the superseded directories listed for Task 6:
  `balanced_30_dataset_humor_v5_conn_v4/`,
  `balanced_30_metaphor_v8_scaffolding_v2/`, `balanced_dataset/`,
  `balanced_dataset_humor_v4_conn_v3/`,
  `balanced_dataset_humor_v5_conn_v4/`, `balanced_dataset_scaf_v2/`,
  `balanced_dataset_v2/`, `balanced_dataset_v8/`, `metaphor_v6_human/`,
  `tie_breaker_dataset/`, and `unbalanced_dataset/`.

## Human preference study
- Keep `scripts/experiment_b/`, including raw anonymized study data, processed data, final regression outputs, and `teacher_significance.py`.
- Remove superseded plots/tables not referenced by the paper while retaining the inputs needed to recreate final reported outputs.
- Keep `first_exp.csv`, `sec_exp.csv`, their filtered forms, the sampled and
  merged study data, the evaluation dataset, formality scores, metric mappings,
  normalization metadata, and `run_10/` metric inputs. These support the pairwise
  preferences in `tab:human_preferences`, the regression in
  `tab:exp_b_regression`, and the metaphor over-optimization analysis in
  `sec:rubric_preference_alignment` and `sec:appendix_metaphor_examples`.
- Keep `logistic_regression.py`, `formality_analysis.py`,
  `metaphor_overoptimization.py`, filtering/sampling/metric preparation scripts,
  and the final continuous-with-formality regression CSVs. Keep
  `dpo_rubric_up_pref_down.csv` because it supplies the 141-case appendix
  analysis.
- Keep the untracked `scripts/experiment_b/teacher_significance.py` unchanged
  and uncommitted in Task 1. Task 6 will move and update it; it consumes the
  canonical `run_9/` metric CSVs for `tab:teacher_robustness` and
  `tab:teacher_composition`.
- Preserve raw and processed data before removing any plot or table. Candidate
  removals are exploratory cluster outputs, interaction/binarized regressions,
  and superseded plot variants not cited by the paper; Task 6 must first verify
  that the retained final scripts can recreate the reported outputs without
  them.

## Factuality
- Keep `scripts/truthfulqa_results/`, `scripts/truthfulqa_visualization.py`, and `trust_llm/`.
- Keep `Benchmarking/deep_eval/truthfulqa_benchmark.py` and its notebook form as
  the executable provenance for Appendix `tab:truthfulqa_results`; move them
  with the factuality pipeline in Task 5.
- Keep all per-model TruthfulQA checkpoints and summary data needed to reproduce
  the reported MC2 scores and significance/effect-size comparison. Earlier
  timestamped summaries may be removed only after confirming that
  `truthfulqa_summary_latest.csv` plus checkpoints can exactly regenerate the
  table and figure.
- Keep `trust_llm/` code, dependency locks, and evaluation JSON outputs as a
  retained auxiliary factuality benchmark. Its `heatmap.py` consumes
  TrustLLM truthfulness JSON, so it must not be described as the source of the
  paper's GPT-5.2 claim-level verification.
- The scoped tracked inventory does not expose an obvious script or result file
  for the GPT-5.2 atomic-claim analysis behind Appendix `tab:factuality`.
  Preserve any such artifact found in later full-tree review and document this
  provenance gap rather than substituting TrustLLM results.
- Human-study accuracy ratings remain with the retained Experiment B data and
  support the second of the paper's three factuality checks.

## Remove completely
- `RAG/`, `rebuttal/`, nested RAG code/data, IDE files, OS files, unrelated PDFs, root scratch notebooks/scripts, and `science-QA_jsonl/`.
- No RAG result is reported in the paper; RAG artifacts therefore fail the
  publication retention test even when nested under an otherwise retained
  evaluator.
- Remove `Benchmarking/baram_tsabari/`, obsolete `.deepeval` telemetry, `.iml`
  files, `.idea/`, root scratch notebooks/scripts and source-paper copies, and
  operational logs. None is an input or output of a retained final pipeline.

## Paper-to-artifact coverage

- `tab:dataset_stats`, `sec:datasets`, and `sec:appendix_sft_prompt`:
  `SFT/data/`, `SFT/batch_file_gen/`, and the SFT/DPO procedures.
- `fig:stacked_scores` and `sec:exp_a_results`: final model generations,
  production rubrics, and canonical `run_9/`.
- `fig:validation_a`--`fig:validation_c` and `sec:metrics_validation`:
  `balanced_dataset_v2_human/`, `tie_breaker_v2/`, Label Studio interfaces, and
  final reliability/correlation analysis.
- `tab:human_preferences`, `tab:exp_b_regression`, and
  `sec:rubric_preference_alignment`: anonymized Experiment B inputs,
  `run_10/`, final regression outputs, and metaphor-overoptimization inputs.
- `tab:truthfulqa_results`: TruthfulQA checkpoints/results and visualization;
  `tab:factuality`: participant-rating data plus the claim-level artifact to be
  located during the full-tree review. `trust_llm/` remains auxiliary
  factuality evidence, not a substitute for either table.
- `tab:winrate`: canonical `run_9/` metric inputs and win-rate outputs.
- `tab:teacher_robustness` and `tab:teacher_composition`: GPT-5/Kimi QA-Pairs
  data and outputs, canonical `run_9/`, and `teacher_significance.py`.
- `sec:appendix_deployment`: retained SFT procedure and documented model
  artifacts; the throughput figures are reported measurements, not a local
  benchmark that must be rerun during cleanup.
