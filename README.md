# Beyond Accuracy: Improving LLMs' Science Communication Skills

Code, data, and analysis for the ACL paper by Mattan Yeroushalmi,
Maxim Bragilovski, and Nir Grinberg.

- **Code:** [github.com/mattany/beyond-accuracy](https://github.com/mattany/beyond-accuracy)
- **Models and datasets:** [huggingface.co/mattany](https://huggingface.co/mattany) ([datasets collection](https://huggingface.co/mattany/datasets))

This repository is a publication-focused companion to the paper. It retains
canonical inputs, outputs, and analysis scripts needed to inspect or rerun the
reported experiments. It does **not** claim full end-to-end reproducibility of
every paper result: end-to-end DPO optimization and the GPT-5.2 claim-level
factuality analysis are documented gaps (see [Reproducing the paper](#reproducing-the-paper)
and [Expected outputs and paper mapping](#expected-outputs-and-paper-mapping)).

## Overview

The paper studies how to improve large language models' science-communication
skills beyond factual accuracy. The retained repository covers four areas:

1. **Training** — QA-Pairs curation, teacher answer generation, SFT notebooks,
   preference-pair construction, and model-variant generation scripts.
2. **Evaluation** — automated rubric scoring, aggregation, win-rate analysis,
   TruthfulQA benchmarking, and auxiliary TrustLLM evidence.
3. **Human study** — judge-validation reliability, Experiment B preference
   analysis, and teacher-robustness significance tests.
4. **Data** — shared QA-Pairs CSVs, splits, and provider outputs under
   `data/qa_pairs/`.

GPU training, paid API calls, and some Colab-oriented notebooks are documented
but not required for local verification. Offline reruns can regenerate most
tables and figures from the tracked canonical CSVs.

**Safe reruns:** Commands marked **offline** read canonical tracked artifacts.
Several scripts write new files when rerun; use explicit `--output`,
`--output-dir`, or `/tmp` destinations documented below so you do not overwrite
publication CSVs or plots. Compare regenerated outputs against canonical paths
rather than writing back into tracked directories.

## Repository map

| Path | Contents | Paper sections |
|------|----------|----------------|
| `data/qa_pairs/` | QA-Pairs CSVs, train/val/test splits, teacher batches and outputs | `sec:datasets`, `tab:dataset_stats`, Appendix `sec:appendix_sft_prompt`, `tab:teacher_robustness` |
| `training/data_generation/` | OpenAI/Moonshot batch teacher generation | QA-Pairs construction, Appendix `sec:appendix_teacher` |
| `training/sft/` | SFT Colab notebook | `sec:model_evaluation`, Appendix `sec:appendix_deployment` |
| `training/dpo/` | Preference-pair construction only (no DPO trainer) | DPO input preparation |
| `training/model_variants/` | Standalone generation CLIs and Hub upload helpers | Model output regeneration |
| `evaluation/rubrics/` | Rubric definitions, runner, aggregation, bootstrap, win-rate | `fig:stacked_scores`, `sec:exp_a_results`, `tab:winrate` |
| `evaluation/model_outputs/` | Canonical model answer tables | Rubric and preference inputs |
| `evaluation/results/rubric_scores/` | Final rubric metric CSVs and derived tables | `fig:stacked_scores`, `tab:winrate`, `tab:teacher_robustness` |
| `evaluation/results/preference_metrics/` | Experiment B metric CSVs and checkpoints | `tab:exp_b_regression`, `sec:rubric_preference_alignment` |
| `evaluation/factuality/` | TruthfulQA checkpoints/visualization; auxiliary TrustLLM | `tab:truthfulqa_results`; `tab:factuality` (partial — see gaps) |
| `human_study/judge_validation/` | Label Studio exports, intercoder reliability | `fig:validation_a`–`fig:validation_c`, `sec:metrics_validation` |
| `human_study/preferences/` | Anonymized Experiment B data and regressions | `tab:human_preferences`, `tab:exp_b_regression`, `sec:appendix_metaphor_examples` |
| `docs/artifacts.md` | Detailed paper-to-artifact manifest | Cross-reference |
| `tools/verify_repository.py` | Publication layout and secret-pattern verifier | CI / release checks |

Component READMEs under each directory document stage-specific commands.

## Data and models

### QA-Pairs (`data/qa_pairs/`)

Derived from Reddit *r/askscience* content. The paper reports use in compliance
with Reddit's permitted-use terms for non-commercial academic research; this
repository does not grant any rights beyond those of the upstream sources.
Tracked files include human, GPT-3.5, GPT-5, and Kimi teacher answers,
train/validation/test splits, batch inputs, and provider JSONL outputs.
Inspect provenance with `data/qa_pairs/load_datasets.ipynb`.

### Pref-Human and Experiment B

Human preference judgments for DPO training and Experiment B live under
`human_study/preferences/data/`. Raw study CSVs
(`human_study/preferences/data/first_exp.csv`,
`human_study/preferences/data/sec_exp.csv`) and processed tables are
anonymized for publication. Label Studio JSON exports
under `human_study/judge_validation/` use stable `annotator_N` labels instead
of email addresses.

**Residual pseudonymity caveat:** some formatted judge-validation CSV columns
still encode coder initials from the annotation workflow (for example
`mattany_connection_v2`, `nirgrn_metaphor_v2`). JSON exports are anonymized;
a broader CSV column rename would be a separate data-release decision.

### Hugging Face artifacts

Models and public dataset releases are hosted at
[huggingface.co/mattany](https://huggingface.co/mattany). Representative model
IDs referenced in this repository:

| Model role | Hugging Face ID |
|------------|-----------------|
| Synthetic SFT (SciComma) | `mattany/SciComma-3.1-8B-Instruct-lora` |
| Human SFT | `mattany/organic-sft-3.1-8B-lora` / `mattany/human-sft-3.1-8B-lora` |
| Human DPO | `mattany/organic-dpo-3.1-8B-lora` / `mattany/human-dpo-3.1-8B-lora` |

Check each model and dataset card for license terms before redistribution or
commercial use. Reddit-derived QA-Pairs content remains subject to Reddit's
[User Agreement](https://www.redditinc.com/policies/user-agreement) and the
upstream
[dhmeltzer/ask-science-qg](https://huggingface.co/datasets/dhmeltzer/ask-science-qg)
dataset terms. The MIT License applies to repository code only.

## Installation

Each component has its own dependency environment. Install only what you need
for the stage you are running.

### Data generation and training

```bash
cd training/data_generation
./setup_env.sh
source .venv/bin/activate
cd ../..   # return to repository root
```

Uses [uv](https://docs.astral.sh/uv/) with Python 3.10+. See
`training/data_generation/README.md` for provider-specific `TEACHER_PROVIDER`
settings.

SFT and GPU generation additionally require PyTorch, Transformers, PEFT, and
(optionally) Colab/Drive for adapter storage. See `training/sft/GPT_SFT_only.ipynb`
and `training/model_variants/README.md`.

### Rubric evaluation

```bash
poetry install --directory evaluation/rubrics
```

Run scripts from the repository root with `PYTHONPATH=.` (see commands below).

### Human-study analysis

Most scripts use pandas, scipy, and statsmodels available in a standard scientific
Python environment. No separate lockfile is required for the documented offline
reruns.

### Factuality analysis

TruthfulQA visualization requires matplotlib and scipy. The Colab-oriented
benchmark runner additionally needs GPU access, Transformers, and `HF_TOKEN`.
TrustLLM heatmaps use a separate Poetry environment:

```bash
cd evaluation/factuality/trust_llm
poetry install
```

## Credentials

Copy `.env.example` to `.env` and set only the providers you use. Never commit
`.env` or real key values.

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | Teacher batch generation (default), rubric LLM judges, Experiment B metrics |
| `MOONSHOT_API_KEY` | Kimi teacher generation (`TEACHER_PROVIDER=kimi`) |
| `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY` | Optional rubric judge backends |
| `HF_TOKEN` | Hugging Face model download/upload and TruthfulQA benchmark |
| `LANGCHAIN_API_KEY` | Optional LangChain tracing in rubric evaluation |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | `fetch_reddit_formatting.py` prep utility only |

Provider secrets are loaded from the environment at runtime. A local
`evaluation/rubrics/config.py` compatibility shim may exist on developer
machines but is gitignored and must not contain committed credentials.

**Git history note:** Task 7 removed credentials from the current tracked tree.
Pre-cleanup commits may still contain revoked secrets. A post-merge history
rewrite (Task 10) is planned but deferred until after the cleanup PR merges;
collaborators will need to re-clone or hard-reset once a rewritten remote is
published.

## Reproducing the paper

Stages are ordered as in the paper pipeline. Each entry notes whether it runs
**offline** from tracked artifacts, needs a **paid API**, requires **GPU**, or
is **not present** in this repository.

### 1. Prepare QA-Pairs

**Offline** — recreate train/val/test splits for alternate teacher answers.
**Warning:** without `--output-dir`, this overwrites `*_train.csv`, `*_test.csv`,
and `*_eval.csv` next to the answers file. Write to a scratch directory instead:

```bash
mkdir -p /tmp/qa_splits
python data/qa_pairs/create_split.py \
  --answers data/qa_pairs/ask_science_gpt_5_answers.csv \
  --output-dir /tmp/qa_splits

python data/qa_pairs/create_split.py \
  --answers data/qa_pairs/ask_science_kimi_answers.csv \
  --drop-truncated \
  --output-dir /tmp/qa_splits
```

Canonical splits under `data/qa_pairs/` are already tracked; rerun only when
validating split logic.

### 2. Generate teacher answers

**Paid API** — OpenAI Batch API (default GPT-5) or Moonshot Kimi batch API:

```bash
python -m training.data_generation.gen_batch
python -m training.data_generation.upload_batch_file
python -m training.data_generation.merge
```

For Kimi, prefix each command with `TEACHER_PROVIDER=kimi` and set
`MOONSHOT_API_KEY`. See `training/data_generation/README.md`.

### 3. Train SFT and DPO models

| Step | Status | Command / artifact |
|------|--------|-------------------|
| Synthetic / Human SFT | **GPU** (Colab notebook) | `training/sft/GPT_SFT_only.ipynb` — requires `HF_TOKEN`, GPU, and QA-Pairs CSVs |
| Preference-pair construction | **Offline** (given score + answer CSVs) | `python training/dpo/preference_dataset_generation.py --scores … --answers … --output …` |
| Human / Synthetic DPO training | **Not present** | No `DPOTrainer` / `DPOConfig` implementation is tracked. Published DPO weights are on Hugging Face; preference-data construction is documented in `training/dpo/README.md` only. |
| Model-variant generation | **GPU** | `python training/model_variants/organic_sft.py --adapter /path/to/adapter` (and `naive_dpo.py`) |
| Hub upload helpers | **GPU + HF_TOKEN** | `training/model_variants/upload_organic_sft.py`, `upload_organic_dpo.py` |

### 4. Evaluate science-communication rubrics

| Step | Status | Command |
|------|--------|---------|
| Score all models on all metrics | **Paid API** | `PYTHONPATH=. poetry --directory evaluation/rubrics run python -m evaluation.rubrics.custom_metrics.run` |
| Aggregate weighted scores and plots | **Offline** | `python -m evaluation.rubrics.custom_metrics.aggregate_v2 --output-dir /tmp/rubric_aggregations_v2` |
| Per-question win rates | **Offline** | `python -m evaluation.rubrics.custom_metrics.winrate --output /tmp/winrate_llama_vs_gpt35.csv` |

The runner reads `evaluation/model_outputs/main/all_models_joined.csv`, writes
main rubric metric CSVs to `evaluation/results/rubric_scores/`, and resumes via
`evaluation/results/rubric_scores/.checkpoints/` (run 9). Experiment B metric
scoring uses run 10 and resumes via
`evaluation/results/preference_metrics/.checkpoints/`. Canonical metric CSVs
are already tracked for inspection without rerunning API calls.

**Warning:** `aggregate_v2` reads metric CSVs from the canonical run directory
but regenerates plots and aggregation tables in `--output-dir` (default:
`evaluation/results/rubric_scores/aggregations_v2/`). `winrate` writes a
per-question CSV; pass `--output /tmp/...` to avoid overwriting tracked
`winrate__*.csv` files.

### 5. Validate rubric judges

**Offline** — intercoder reliability and human–LLM correlation:

```bash
python human_study/judge_validation/intercoder_reliability.py \
  human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json \
  --output-dir /tmp/judge_validation_results
```

Compare outputs against canonical CSVs under
`human_study/judge_validation/balanced_dataset_v2_human/`. Label Studio
interface XML files are in `human_study/judge_validation/labeling_interface/`.

API-required prep utilities (`analyze_balance_options.py`,
`run_connection_reasons.py`, `fetch_reddit_formatting.py`) are outside this
publication rerun set.

### 6. Analyze human preferences

**Offline** (canonical outputs already tracked):

```bash
# Teacher robustness / composition (stdout only)
python human_study/preferences/teacher_significance.py

# Experiment B continuous regression with formality covariate
python human_study/preferences/logistic_regression.py \
  --mode continuous \
  --with-formality \
  --output /tmp/logistic_regression_continuous_with_formality.csv

# Metaphor over-optimization appendix analysis
python human_study/preferences/metaphor_overoptimization.py
```

Full Experiment B metric scoring (`run_metrics_exp_b.py`) requires a **paid API**
and is documented in `human_study/preferences/README.md`.

### 7. Run factuality checks

| Check | Status | Command / artifact |
|-------|--------|-------------------|
| TruthfulQA MC2 table and figure | **Offline** from checkpoints | See command below |
| TruthfulQA checkpoint regeneration | **GPU + HF_TOKEN** (Colab-oriented) | `evaluation/factuality/truthfulqa_benchmark.py` / `.ipynb` |
| Participant accuracy ratings (Experiment B) | **Offline** | Data under `human_study/preferences/data/` |
| GPT-5.2 atomic-claim verification (`tab:factuality`) | **Not present** | No tracked script or result file reproduces this analysis |
| TrustLLM auxiliary benchmark | **Offline** heatmap | `cd evaluation/factuality/trust_llm && MPLBACKEND=Agg poetry run python heatmap.py` |

TrustLLM is auxiliary evidence and must not be substituted for the missing
GPT-5.2 claim-level provenance.

**TruthfulQA visualization (safe rerun):** `--output` controls only the PNG.
The script also writes `truthfulqa_results_summary.csv` into `--results-dir`.
Copy checkpoints to a scratch directory or pass an explicit PNG path:

```bash
cp -r evaluation/factuality/truthfulqa_results /tmp/truthfulqa_results
python evaluation/factuality/truthfulqa_visualization.py \
  --results-dir /tmp/truthfulqa_results \
  --output /tmp/truthfulqa_comparison.png
```

## Expected outputs and paper mapping

| Paper artifact | Primary location | Offline reproducible? |
|----------------|------------------|----------------------|
| `tab:dataset_stats`, `sec:datasets` | `data/qa_pairs/` | Yes (data inspection) |
| `fig:stacked_scores`, `sec:exp_a_results` | `evaluation/results/rubric_scores/aggregations_v2/`, plots from `aggregate_v2` | Yes (from tracked CSVs) |
| `tab:winrate` | `evaluation/results/rubric_scores/` + `evaluation/rubrics/custom_metrics/winrate.py` | Yes |
| `fig:validation_a`–`fig:validation_c` | `human_study/judge_validation/balanced_dataset_v2_human/` | Yes (`intercoder_reliability.py`) |
| `tab:human_preferences` | `human_study/preferences/data/` | Yes (tracked tables) |
| `tab:exp_b_regression` | `human_study/preferences/data/logistic_regression_continuous_with_formality.csv` | Yes (`logistic_regression.py --with-formality`) |
| `tab:teacher_robustness`, `tab:teacher_composition` | `human_study/preferences/teacher_significance.py` stdout vs `evaluation/results/rubric_scores/` | Yes |
| `tab:truthfulqa_results` | `evaluation/factuality/truthfulqa_results/` | Yes (`truthfulqa_visualization.py`) |
| `tab:factuality` (claim-level) | — | **No** — provenance gap |
| `sec:appendix_metaphor_examples` | `human_study/preferences/data/dpo_rubric_up_pref_down.csv`, `human_study/preferences/metaphor_overoptimization.py` | Yes |
| DPO-trained model weights | Hugging Face (`mattany/*`) | External (training not in repo) |

See `docs/artifacts.md` for the full retention manifest.

## Ethics and data provenance

- **Reddit / QA-Pairs:** Built from *r/askscience* content. The paper reports
  compliance with Reddit's permitted-use terms for non-commercial academic
  research; binding rights and restrictions come from Reddit's User Agreement
  and the upstream dataset terms, not from this repository.
- **Human annotations:** Experiment B preference data and judge-validation
  exports are anonymized for publication. Label Studio emails and draft metadata
  were replaced with stable `annotator_N` identifiers; judgment content is
  preserved.
- **Residual identifiers:** Formatted judge-validation CSVs may still contain
  pseudonymous coder initials in column names (see [Data and models](#data-and-models)).
- **Third-party models:** Llama, GPT, Kimi, and other provider models are
  subject to their respective terms of use.

Repository code is released under the MIT License. Reddit-derived QA-Pairs
content, third-party datasets, and model weights remain subject to their
source licenses and terms. See the linked dataset and model cards before
redistribution or commercial use.

## Citation

Until official ACL proceedings metadata is available, cite:

```bibtex
@inproceedings{yeroushalmi2026beyond,
  title={Beyond Accuracy: Improving LLMs' Science Communication Skills},
  author={Yeroushalmi, Mattan and Bragilovski, Maxim and Grinberg, Nir},
  year={2026}
}
```

Replace this placeholder with the official proceedings citation once published.

## License

Repository code is released under the [MIT License](LICENSE).

Reddit-derived QA-Pairs content, third-party datasets, and model weights remain
subject to their source licenses and terms. See the linked Hugging Face model
and dataset cards before redistribution or commercial use.
