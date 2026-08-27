# Publication Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `beyond-accuracy` into a focused, reproducible, credential-free public companion to “Beyond Accuracy: Improving LLMs' Science Communication Skills.”

**Architecture:** Rebuild the current research tree around four reader-facing areas: training, evaluation, human-study analysis, and data. Preserve canonical paper inputs and outputs, remove abandoned experiments, replace hard-coded credentials with environment variables, document each paper pipeline, and finally rewrite local Git history to scrub confirmed secrets.

**Tech Stack:** Python 3.10/3.11, uv/Poetry, pandas, DeepEval, Transformers/TRL/Unsloth notebooks, pytest, git-filter-repo, and a secret scanner such as Gitleaks.

## Global Constraints

- Retain only artifacts supporting an ACL paper method, dataset, figure, table, appendix result, or reproducibility step.
- Keep all tracked datasets, intermediates, and final outputs needed to rerun retained pipelines exactly.
- Remove RAG and rebuttal work completely from the current tree.
- Use MIT for repository code; document third-party dataset and model terms separately.
- Never commit credentials, replacement maps, `.env`, ignored `config.py`, IDE metadata, or OS artifacts.
- Do not commit any changes unless the user separately requests a commit.
- Do not push or force-push. A rewritten remote requires a separate confirmation immediately before that operation.
- Preserve the user's untracked `scripts/experiment_b/teacher_significance.py` by incorporating it into the teacher-robustness analysis.
- Treat GPU training and paid API evaluation as documented external stages; do not invoke them during local verification.

---

## Target File Structure

```text
README.md
LICENSE
.env.example
.gitignore
data/
  qa_pairs/
evaluation/
  factuality/
  model_generation/
  model_outputs/
  results/
    rubric_scores/
    preference_metrics/
  rubrics/
  visualization/
human_study/
  judge_validation/
  preferences/
training/
  data_generation/
  dpo/
  model_variants/
  sft/
tools/
  verify_repository.py
tests/
  test_public_repository.py
docs/
  artifacts.md
  superpowers/
```

Path ownership:

- `data/qa_pairs/` owns QA-Pairs human, GPT-3.5, GPT-5, and Kimi source/split data plus generation batches and outputs.
- `training/` owns data-generation utilities and SFT/DPO procedures.
- `evaluation/rubrics/` owns metric definitions and runners; `evaluation/results/` owns final metric outputs.
- `evaluation/factuality/` owns TruthfulQA and trust-LLM outputs.
- `human_study/judge_validation/` owns Label Studio exports, agreement analysis, and human–LLM validation.
- `human_study/preferences/` owns Experiment B input, processing, regression, and reported outputs.

---

### Task 1: Freeze the publication artifact manifest

**Files:**
- Create: `docs/artifacts.md`
- Read: `/Users/mattan.yeroushalmi/studies/acl_paper/latex/main.tex`
- Read: all tracked paths under `SFT/`, `DPO/`, `Benchmarking/`, `scripts/`, and `trust_llm/`

**Interfaces:**
- Consumes: ACL paper methods and reported claims.
- Produces: a complete keep/move/delete manifest used by Tasks 3–7.

- [ ] **Step 1: Write the paper-to-artifact map**

Create `docs/artifacts.md` with these sections and decisions:

```markdown
# Paper artifact map

## QA-Pairs and teacher datasets
- Keep and move `SFT/data/` to `data/qa_pairs/`.
- Keep current generators from `SFT/batch_file_gen/`.
- Remove `SFT/batch_file_gen/archive/`, batch status/job logs, and local setup residue.

## Model training
- Keep `SFT/training/GPT_SFT_only.ipynb`.
- Keep `DPO/preference_dataset_generation.py`.
- Keep the non-RAG model-variant scripts in `Benchmarking/truthfulness/` where they document paper model generation or publication.

## Rubric evaluation and model outputs
- Keep non-RAG code under `Benchmarking/deep_eval/`.
- Keep canonical final rubric outputs `Benchmarking/deep_eval/data/run_9/`.
- Keep Experiment B metric outputs `Benchmarking/deep_eval/data/run_10/`.
- Keep final generation outputs under `scripts/generations/`, `scripts/generations_2/`, and `scripts/generations_3/`.
- Remove runs 3–8, archived/test data, model-metric exploratory data, consistency experiments, RAG evaluation, and Ollama failure investigations.

## Judge validation
- Keep final validation code and the final `balanced_dataset_v2_human/` and `tie_breaker_v2/` artifacts.
- Keep Label Studio interfaces required to reproduce those exports.
- Remove superseded balanced-dataset versions, disagreement experiments, and one-off obsolete metric runners after confirming the final scripts do not consume them.

## Human preference study
- Keep `scripts/experiment_b/`, including raw anonymized study data, processed data, final regression outputs, and `teacher_significance.py`.
- Remove superseded plots/tables not referenced by the paper while retaining the inputs needed to recreate final reported outputs.

## Factuality
- Keep `scripts/truthfulqa_results/`, `scripts/truthfulqa_visualization.py`, and `trust_llm/`.

## Remove completely
- `RAG/`, `rebuttal/`, nested RAG code/data, IDE files, OS files, unrelated PDFs, root scratch notebooks/scripts, and `science-QA_jsonl/`.
```

- [ ] **Step 2: Validate every retained result against the paper**

Run:

```bash
git ls-files > /tmp/beyond-accuracy-tracked-before.txt
rg -n '\\label\{(fig|tab|sec):|run_[0-9]+|teacher|TruthfulQA|factual|preference|validation' \
  /Users/mattan.yeroushalmi/studies/acl_paper/latex/main.tex
```

Expected: every reported training, rubric, validation, preference, teacher-robustness, and factuality result has an entry in `docs/artifacts.md`; no RAG result is reported.

- [ ] **Step 3: Review the manifest against final-script dependencies**

Run:

```bash
rg -n 'run_[3-8]|archive|consistency_check|balanced_dataset(_v2|_v8|_scaf|_humor)?' \
  Benchmarking scripts --glob '*.py'
```

Expected: any path consumed by a retained final script is either retained or the consuming script is updated in Task 5. Record the final keep/delete resolution explicitly in `docs/artifacts.md`.

---

### Task 2: Add executable publication-layout checks

**Files:**
- Create: `tools/verify_repository.py`
- Create: `tests/test_public_repository.py`

**Interfaces:**
- Produces: `verify_repository(root: Path) -> list[str]`, returning human-readable violations.
- Consumes: target structure and forbidden-path rules from this plan.

- [ ] **Step 1: Write failing tests for the intended public tree**

Create `tests/test_public_repository.py`:

```python
from pathlib import Path

from tools.verify_repository import verify_repository


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_has_no_layout_violations():
    assert verify_repository(ROOT) == []


def test_required_public_files_exist():
    required = [
        "README.md",
        "LICENSE",
        ".env.example",
        "docs/artifacts.md",
        "training",
        "evaluation",
        "human_study",
        "data/qa_pairs",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []
```

- [ ] **Step 2: Run tests and confirm they fail before cleanup**

Run:

```bash
python -m pytest tests/test_public_repository.py -v
```

Expected: FAIL because `tools.verify_repository`, README, license, and target directories do not yet exist.

- [ ] **Step 3: Implement the repository verifier**

Create `tools/verify_repository.py`:

```python
from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_ROOTS = {
    "Benchmarking",
    "DPO",
    "RAG",
    "SFT",
    "rebuttal",
    "science-QA_jsonl",
    "scripts",
    "trust_llm",
}
FORBIDDEN_NAMES = {".DS_Store", ".idea", "__pycache__"}
FORBIDDEN_SUFFIXES = {".iml", ".pyc"}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
TEXT_SUFFIXES = {
    ".cfg", ".ini", ".ipynb", ".json", ".md", ".py", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}


def verify_repository(root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"forbidden root remains: {name}"
        for name in sorted(FORBIDDEN_ROOTS)
        if (root / name).exists()
    )
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/IDE artifact remains: {relative}")
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} in {relative}")
    return sorted(errors)


if __name__ == "__main__":
    violations = verify_repository(Path(__file__).resolve().parents[1])
    if violations:
        print("\n".join(violations), file=sys.stderr)
        raise SystemExit(1)
    print("Repository layout and tracked text are publication-clean.")
```

- [ ] **Step 4: Run the verifier and preserve the expected red state**

Run:

```bash
python tools/verify_repository.py
```

Expected: exit 1, listing old top-level areas and tracked credential findings. This verifier becomes green after Tasks 3–7.

---

### Task 3: Remove publication-irrelevant artifacts

**Files:**
- Delete: `RAG/`
- Delete: `rebuttal/`
- Delete: `Benchmarking/deep_eval/RAG/`
- Delete: `Benchmarking/deep_eval/rag_evaluation.py`
- Delete: `Benchmarking/deep_eval/data/archive/`
- Delete: `Benchmarking/deep_eval/data/consistency_check/`
- Delete: `Benchmarking/deep_eval/data/model_metric_specific/`
- Delete: `Benchmarking/deep_eval/data/run_3/` through `run_8/`
- Delete: `Benchmarking/deep_eval/data/test_data/`
- Delete: `Benchmarking/deep_eval/ollama-bad-answers/`
- Delete: `SFT/batch_file_gen/archive/`
- Delete: root scratch files listed below

**Interfaces:**
- Consumes: approved manifest from Task 1.
- Produces: a current tree containing only paper-relevant artifacts.

- [ ] **Step 1: Remove explicitly excluded experiment areas**

Run:

```bash
git rm -r RAG rebuttal \
  Benchmarking/deep_eval/RAG \
  Benchmarking/deep_eval/data/archive \
  Benchmarking/deep_eval/data/consistency_check \
  Benchmarking/deep_eval/data/model_metric_specific \
  Benchmarking/deep_eval/data/run_3 \
  Benchmarking/deep_eval/data/run_4 \
  Benchmarking/deep_eval/data/run_5 \
  Benchmarking/deep_eval/data/run_6 \
  Benchmarking/deep_eval/data/run_7 \
  Benchmarking/deep_eval/data/run_8 \
  Benchmarking/deep_eval/data/test_data \
  Benchmarking/deep_eval/ollama-bad-answers \
  SFT/batch_file_gen/archive
git rm Benchmarking/deep_eval/rag_evaluation.py
```

Expected: all listed tracked paths are staged as deletions; `run_9` and `run_10` remain.

- [ ] **Step 2: Remove unrelated root artifacts and IDE metadata**

Run:

```bash
git rm .cursorignore .cursorrules \
  baram_tsabari_An_Instrument_for_Assessing_Scientists_Written_Sk.txt \
  main.py mistral_finetuning.ipynb \
  multilingual_instruction_tuning.pdf Osborne2004Enhancing994.pdf \
  question-generation.ipynb question_generation.ipynb \
  requirements.txt science_qa.py thesis.iml
git rm -r science-QA_jsonl SFT/.idea Benchmarking/baram_tsabari
git rm Benchmarking/deep_eval/.deepeval \
  Benchmarking/deep_eval/deep_eval.iml \
  Benchmarking/eval_dataset_generation/eval_dataset_generation.iml \
  Benchmarking/visualization/visualization.iml
```

Expected: the Reddit credential in obsolete `main.py`, unrelated papers, scratch notebooks, ScienceQA data, and IDE files are removed from the current tree.

- [ ] **Step 3: Remove obsolete operational logs**

Run:

```bash
git rm SFT/batch_jobs.json SFT/batch_jobs_kimi.json \
  SFT/batch_status.txt SFT/batch_status_kimi.txt
```

Expected: resumable generation data under `SFT/data/` remains, but account/job status residue is gone.

- [ ] **Step 4: Remove root prompt duplicates after hash/content comparison**

Run:

```bash
shasum prompts_0.csv prompts_copy.csv prompts_original.csv \
  SFT/data/ask_science.csv SFT/data/ask_science_human.csv
```

Expected: inspect whether any root prompt CSV is the canonical source. If `docs/artifacts.md` maps none to the paper pipeline, run:

```bash
git rm prompts_0.csv prompts_copy.csv prompts_original.csv
```

Expected: QA-Pairs sources remain under the retained SFT data tree.

---

### Task 4: Reorganize training and data assets

**Files:**
- Move: `SFT/data/` → `data/qa_pairs/`
- Move: `SFT/batch_file_gen/` → `training/data_generation/`
- Move: `SFT/training/` → `training/sft/`
- Move: `DPO/preference_dataset_generation.py` → `training/dpo/preference_dataset_generation.py`
- Move: `Benchmarking/truthfulness/` → `training/model_variants/`

**Interfaces:**
- Produces: publication-facing training and QA dataset paths.
- Consumes: no runtime interfaces beyond filesystem paths.

- [ ] **Step 1: Move QA-Pairs data and active generators**

Run:

```bash
mkdir -p data training
git mv SFT/data data/qa_pairs
git mv SFT/batch_file_gen training/data_generation
git mv SFT/training training/sft
```

Expected: all teacher datasets and batch outputs are under `data/qa_pairs/`; active generation utilities and notebooks are under `training/`.

- [ ] **Step 2: Move DPO and model-variant procedures**

Run:

```bash
mkdir -p training/dpo training/model_variants
git mv DPO/preference_dataset_generation.py training/dpo/preference_dataset_generation.py
git mv Benchmarking/truthfulness/* training/model_variants/
git rm DPO/__init__.py SFT/__init__.py
```

Expected: no `DPO/` or `SFT/` tracked content remains.

- [ ] **Step 3: Normalize training path configuration**

Replace hard-coded `/content/drive/.../thesis/...` and local `/Users/.../thesis/...` paths in retained Python scripts with CLI arguments whose defaults are repository-relative:

```python
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "qa_pairs")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()
```

For Colab notebooks, retain explicit mount instructions but replace old repository-name paths with `/content/beyond-accuracy/...`.

- [ ] **Step 4: Check training paths**

Run:

```bash
rg -n '/Users/.*/thesis|MyDrive/thesis|My Drive/thesis|SFT/|DPO/' \
  training data --glob '*.{py,ipynb,md,toml,sh}'
```

Expected: no local user path or obsolete repository path remains; intentional prose references to SFT/DPO methods are allowed.

---

### Task 5: Reorganize evaluation and update runtime paths

**Files:**
- Move: `Benchmarking/deep_eval/` → `evaluation/rubrics/`
- Move: `Benchmarking/eval_dataset_generation/` → `evaluation/model_generation/`
- Move: `Benchmarking/visualization/` → `evaluation/visualization/`
- Move: `scripts/generations*/` → `evaluation/model_outputs/`
- Move: `scripts/truthfulqa_results/` and `scripts/truthfulqa_visualization.py` → `evaluation/factuality/`
- Move: `trust_llm/` → `evaluation/factuality/trust_llm/`
- Move: final `run_9/` and `run_10/` outputs → `evaluation/results/`

**Interfaces:**
- Consumes: `data/qa_pairs/`.
- Produces: canonical rubric and factuality commands documented by README.

- [ ] **Step 1: Move evaluation code**

Run:

```bash
printf '\n# Local provider configuration\nconfig.py\n' >> .gitignore
mkdir -p evaluation
git mv Benchmarking/deep_eval evaluation/rubrics
git mv Benchmarking/eval_dataset_generation evaluation/model_generation
git mv Benchmarking/visualization evaluation/visualization
```

Expected: retained metric definitions, evaluation generation, and visualization code live under `evaluation/`.

- [ ] **Step 2: Separate final result runs from evaluator code**

Run:

```bash
mkdir -p evaluation/results/rubric_scores evaluation/results/preference_metrics
git mv evaluation/rubrics/data/run_9/* evaluation/results/rubric_scores/
git mv evaluation/rubrics/data/run_10/* evaluation/results/preference_metrics/
```

Remove now-empty tracked directories naturally. Expected: paper score tables use named result directories rather than opaque run numbers.

- [ ] **Step 3: Move generation outputs and factuality assets**

Run:

```bash
mkdir -p evaluation/model_outputs evaluation/factuality
git mv scripts/generations evaluation/model_outputs/main
git mv scripts/generations_2 evaluation/model_outputs/dpo_variants
git mv scripts/generations_3 evaluation/model_outputs/human_variants
git mv scripts/truthfulqa_results evaluation/factuality/truthfulqa_results
git mv scripts/truthfulqa_visualization.py evaluation/factuality/truthfulqa_visualization.py
git mv trust_llm evaluation/factuality/trust_llm
```

Expected: generated model answers and all three factuality-analysis outputs are discoverable under `evaluation/`.

- [ ] **Step 4: Replace config imports with environment-backed settings**

Create `evaluation/rubrics/settings.py`:

```python
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


OPENAI_API_KEY = optional_env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = optional_env("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = optional_env("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = optional_env("GOOGLE_API_KEY")
XAI_API_KEY = optional_env("XAI_API_KEY")
MOONSHOT_API_KEY = optional_env("MOONSHOT_API_KEY")
HF_TOKEN = optional_env("HF_TOKEN")
LANGCHAIN_API_KEY = optional_env("LANGCHAIN_API_KEY")
```

Replace imports such as:

```python
from config import PROJECT_DIR, OPENAI_API_KEY
```

with:

```python
from settings import OPENAI_API_KEY, PROJECT_ROOT
```

Only assign `os.environ["OPENAI_API_KEY"]` when the value is present, and fail with a clear message immediately before an API-backed operation if it is absent.

- [ ] **Step 5: Update all result and data paths**

Apply these canonical replacements in retained Python and Markdown files:

```text
Benchmarking/deep_eval/data/run_9
→ evaluation/results/rubric_scores

Benchmarking/deep_eval/data/run_10
→ evaluation/results/preference_metrics

SFT/data
→ data/qa_pairs

scripts/generations
→ evaluation/model_outputs/main
```

Replace absolute user paths with `Path(__file__).resolve()`-based repository-relative paths.

- [ ] **Step 6: Verify evaluation path migration**

Run:

```bash
rg -n 'Benchmarking/|scripts/generations|trust_llm/|run_9|run_10|/Users/.*/thesis' \
  evaluation --glob '*.{py,ipynb,md,toml}'
```

Expected: no stale filesystem references. Model names and prose mentioning “benchmarking” remain valid.

---

### Task 6: Reorganize human-study pipelines

**Files:**
- Move: `scripts/experiment_b/` → `human_study/preferences/`
- Move: `scripts/judge_alignment/` → `human_study/judge_validation/`
- Modify: retained scripts that reference old metric and dataset paths

**Interfaces:**
- Consumes: `evaluation/results/preference_metrics/` and retained anonymized study exports.
- Produces: paper preference table, regression table, metaphor-overoptimization analysis, and judge-validation results.

- [ ] **Step 1: Move study areas**

Run:

```bash
mkdir -p human_study
git mv scripts/experiment_b human_study/preferences
git mv scripts/judge_alignment human_study/judge_validation
git mv scripts/add_model_from_csv.py evaluation/model_generation/add_model_from_csv.py
```

Expected: `teacher_significance.py` moves with the preference study; no untracked work is lost.

- [ ] **Step 2: Remove superseded judge-validation artifacts**

Using the final dependency resolution recorded in `docs/artifacts.md`, remove old versioned directories not consumed by final validation:

```text
balanced_30_dataset_humor_v5_conn_v4/
balanced_30_metaphor_v8_scaffolding_v2/
balanced_dataset/
balanced_dataset_humor_v4_conn_v3/
balanced_dataset_humor_v5_conn_v4/
balanced_dataset_scaf_v2/
balanced_dataset_v2/
balanced_dataset_v8/
metaphor_v6_human/
tie_breaker_dataset/
unbalanced_dataset/
```

Keep:

```text
balanced_dataset_v2_human/
tie_breaker_v2/
labeling_interface/
```

Before each deletion, confirm no retained final script references it:

```bash
for directory in \
  balanced_30_dataset_humor_v5_conn_v4 \
  balanced_30_metaphor_v8_scaffolding_v2 \
  balanced_dataset \
  balanced_dataset_humor_v4_conn_v3 \
  balanced_dataset_humor_v5_conn_v4 \
  balanced_dataset_scaf_v2 \
  balanced_dataset_v2 \
  balanced_dataset_v8 \
  metaphor_v6_human \
  tie_breaker_dataset \
  unbalanced_dataset
do
  rg -n "$directory" human_study/judge_validation --glob '*.py' || true
done
```

Expected: each removed directory has no retained consumer, or the consumer is also identified as obsolete in `docs/artifacts.md`.

- [ ] **Step 3: Update preference and validation paths**

In `human_study/preferences/logistic_regression.py`, point metrics to:

```python
METRICS_DIR = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "results"
    / "preference_metrics"
)
```

In `human_study/preferences/teacher_significance.py`, point teacher metrics to:

```python
RUN_RESULTS = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "results"
    / "rubric_scores"
)
```

Apply the same root-relative pattern throughout judge-validation scripts.

- [ ] **Step 4: Run offline human-study analyses**

Run:

```bash
python human_study/preferences/teacher_significance.py
python human_study/preferences/logistic_regression.py \
  --mode continuous --with-formality
python human_study/judge_validation/intercoder_reliability.py \
  human_study/judge_validation/balanced_dataset_v2_human/labelstudio_output.json
```

Expected: commands complete without paid API calls and reproduce teacher comparisons, preference regression output, and inter-coder reliability output. If a command’s actual CLI differs, update its parser and README rather than invoking hard-coded paths.

- [ ] **Step 5: Remove the empty legacy root**

Run:

```bash
git ls-files scripts Benchmarking SFT DPO trust_llm
```

Expected: no output. Remove any empty untracked legacy directories after confirming they contain no user work.

---

### Task 7: Finish credential removal in the current tree

**Files:**
- Create: `.env.example`
- Modify: `training/data_generation/README.md`
- Modify: `human_study/judge_validation/fetch_reddit_formatting.py`
- Modify: retained notebooks containing hard-coded Hugging Face tokens
- Modify locally but do not track: `evaluation/rubrics/config.py` if the ignored file was moved
- Modify: `.gitignore`

**Interfaces:**
- Consumes: environment variables.
- Produces: credential-free tracked code and safe configuration examples.

- [ ] **Step 1: Add the environment template**

Create `.env.example`:

```dotenv
# Copy to .env and set only the providers you use.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
MOONSHOT_API_KEY=
HF_TOKEN=
LANGCHAIN_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=beyond-accuracy research script
```

- [ ] **Step 2: Refactor Reddit configuration**

Replace literal Reddit credentials in `human_study/judge_validation/fetch_reddit_formatting.py` with:

```python
import os

import praw


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this script.")
    return value


reddit = praw.Reddit(
    client_id=required_env("REDDIT_CLIENT_ID"),
    client_secret=required_env("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv(
        "REDDIT_USER_AGENT",
        "beyond-accuracy research script",
    ),
)
```

- [ ] **Step 3: Refactor notebook authentication**

Remove literal `hf_...` values from retained notebooks. Use:

```python
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])
```

Do not retain old tokens in comments or cell outputs. Clear credential-bearing outputs while preserving scientific outputs.

- [ ] **Step 4: Safen ignored local configuration**

Confirm the ignored config remains untracked:

```bash
git check-ignore -v evaluation/rubrics/config.py
git ls-files --error-unmatch evaluation/rubrics/config.py
```

Expected: first command identifies an ignore rule; second exits nonzero. Replace local credential literals with `os.getenv(...)`, but never stage this file.

- [ ] **Step 5: Replace the generated `.gitignore`**

Reduce `.gitignore` to reusable patterns:

```gitignore
.DS_Store
.env
.env.*
!.env.example
.idea/
.vscode/
*.iml
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
config.py
*.log
```

Do not retain the current thousand-line RAG-specific ignore list.

- [ ] **Step 6: Scan tracked content**

Run:

```bash
python tools/verify_repository.py
git grep -nEi \
  '(sk-(proj-|ant-api[0-9]+-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,})'
```

Expected: no secret matches. Example placeholders such as `OPENAI_API_KEY=` are allowed.

---

### Task 8: Add publication documentation and licensing

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Modify: component READMEs and `docs/artifacts.md`

**Interfaces:**
- Consumes: final paths and commands from Tasks 4–7.
- Produces: public onboarding and reproducibility instructions.

- [ ] **Step 1: Add the MIT license**

Create `LICENSE` with the standard MIT License text, copyright:

```text
Copyright (c) 2026 Mattan Yeroushalmi, Maxim Bragilovski, and Nir Grinberg
```

- [ ] **Step 2: Write the root README**

Use this exact section structure:

```markdown
# Beyond Accuracy: Improving LLMs' Science Communication Skills

Code, data, and analysis for the ACL paper by Mattan Yeroushalmi,
Maxim Bragilovski, and Nir Grinberg.

## Overview
## Repository map
## Data and models
## Installation
### Data generation and training
### Rubric evaluation
### Human-study analysis
### Factuality analysis
## Credentials
## Reproducing the paper
### 1. Prepare QA-Pairs
### 2. Generate teacher answers
### 3. Train SFT and DPO models
### 4. Evaluate science-communication rubrics
### 5. Validate rubric judges
### 6. Analyze human preferences
### 7. Run factuality checks
## Expected outputs and paper mapping
## Ethics and data provenance
## Citation
## License
```

Document exact commands validated in Tasks 4–7. Link datasets and models to `https://huggingface.co/mattany` and code to `https://github.com/mattany/beyond-accuracy`.

- [ ] **Step 3: Add citation metadata to README**

Until official proceedings metadata exists, use:

```bibtex
@inproceedings{yeroushalmi2026beyond,
  title={Beyond Accuracy: Improving LLMs' Science Communication Skills},
  author={Yeroushalmi, Mattan and Bragilovski, Maxim and Grinberg, Nir},
  year={2026}
}
```

State that readers should replace it with the proceedings citation once available.

- [ ] **Step 4: Document licensing boundaries**

State explicitly:

```markdown
Repository code is released under the MIT License. Reddit-derived QA-Pairs
content, third-party datasets, and model weights remain subject to their
source licenses and terms. See the linked dataset and model cards before
redistribution or commercial use.
```

- [ ] **Step 5: Validate documentation paths**

Run:

```bash
rg -No '`[^`]+`' README.md docs/artifacts.md
python tools/verify_repository.py
```

Expected: every documented repository path exists, and the verifier reports a publication-clean tree.

---

### Task 9: Verify the reorganized repository

**Files:**
- Modify as failures require: retained Python files, notebooks, READMEs, and tests

**Interfaces:**
- Consumes: complete current-tree cleanup.
- Produces: verification evidence before history rewriting.

- [ ] **Step 1: Run repository tests**

Run:

```bash
python -m pytest tests/test_public_repository.py -v
```

Expected: PASS.

- [ ] **Step 2: Compile retained Python**

Run:

```bash
python -m compileall -q training evaluation human_study tools
```

Expected: exit 0. Exclude any intentionally Colab-only `.py` export that contains notebook shell syntax by converting it to `.ipynb` or making it valid Python.

- [ ] **Step 3: Search for stale names and paths**

Run:

```bash
rg -n '/Users/mattan|studies/thesis|MyDrive/thesis|My Drive/thesis|mattany/thesis|RAG/|rebuttal/' \
  . --glob '!docs/superpowers/**' --glob '!.git/**'
```

Expected: no stale paths or old repository URLs. Historical discussion in the design and plan is excluded.

- [ ] **Step 4: Review tracked size and status**

Run:

```bash
git status --short
git count-objects -vH
git ls-files | awk -F/ '{print $1}' | sort | uniq -c | sort -nr
```

Expected: only publication-clean top-level areas remain; the pre-existing untracked teacher script is now included under `human_study/preferences/`; ignored local config is absent from status.

- [ ] **Step 5: Run available secret scanner on current files**

If Gitleaks is installed:

```bash
gitleaks dir . --no-banner --redact
```

Otherwise install it through Homebrew and rerun:

```bash
brew install gitleaks
gitleaks dir . --no-banner --redact
```

Expected: no leaks. Investigate each finding; do not suppress a real secret.

---

### Task 10: Rewrite local history to remove confirmed credentials

**Files:**
- Create temporarily outside repository: `/tmp/beyond-accuracy-replacements.txt`
- Rewrite: all local Git refs
- Never modify remote refs by pushing in this task

**Interfaces:**
- Consumes: confirmed literal secrets from tracked history.
- Produces: locally rewritten credential-free Git history.

- [ ] **Step 1: Confirm the working tree and backup source**

Run:

```bash
git status --short
git remote -v
```

Expected: cleanup changes are present and the existing remote still points to `mattany/beyond-accuracy`. Because history rewriting requires committed input, stop here unless the user has separately authorized commits. If authorized, commit the publication cleanup before continuing.

- [ ] **Step 2: Build a non-repository replacement map**

Generate `/tmp/beyond-accuracy-replacements.txt` directly from all historical patches so no secret needs to be copied into this plan:

```bash
python - <<'PY'
import re
import subprocess
from pathlib import Path

history = subprocess.run(
    ["git", "log", "-p", "--all", "--no-ext-diff", "--text"],
    check=True,
    capture_output=True,
    text=True,
    errors="ignore",
).stdout

patterns = [
    re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"\bxai-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\blsv2_[A-Za-z0-9_:-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(
        r"(?i)(?:client_id|client_secret)\s*=\s*[\"']([A-Za-z0-9_-]{16,})[\"']"
    ),
]

secrets = set()
for pattern in patterns:
    for match in pattern.finditer(history):
        secrets.add(match.group(1) if match.lastindex else match.group(0))

destination = Path("/tmp/beyond-accuracy-replacements.txt")
destination.write_text(
    "".join(
        f"literal:{secret}==>***REMOVED***\n"
        for secret in sorted(secrets)
    ),
    encoding="utf-8",
)
print(f"Wrote {len(secrets)} candidate replacements to {destination}")
PY
```

Review the candidate count and provider types without printing values. Remove any false positives using a local editor, and ensure the set covers tracked Reddit credentials and provider/Hugging Face tokens. Then set restrictive permissions:

```bash
chmod 600 /tmp/beyond-accuracy-replacements.txt
```

- [ ] **Step 3: Install and dry-check git-filter-repo**

Run:

```bash
git filter-repo --version
```

Expected: a version string. If unavailable:

```bash
brew install git-filter-repo
```

- [ ] **Step 4: Rewrite all local refs**

Run only after cleanup is committed and the replacement map has been reviewed:

```bash
git filter-repo --sensitive-data-removal \
  --replace-text /tmp/beyond-accuracy-replacements.txt \
  --force
```

Expected: commit IDs change wherever credentials appeared; current publication files remain intact. Re-add the remote locally if git-filter-repo removes it, but do not push.

- [ ] **Step 5: Delete the replacement map**

Run:

```bash
rm /tmp/beyond-accuracy-replacements.txt
```

Expected: the plaintext replacement list no longer exists.

- [ ] **Step 6: Scan rewritten history**

Run:

```bash
gitleaks git . --no-banner --redact
python tools/verify_repository.py
python -m pytest tests/test_public_repository.py -v
```

Expected: no secrets in history, publication verifier passes, and tests pass.

- [ ] **Step 7: Review rewritten divergence without pushing**

Run:

```bash
git status
git log --oneline --decorate -10
git fsck --full
```

Expected: repository is healthy and local history differs from the remote. Report the old and new tip commit IDs without exposing secret values.

- [ ] **Step 8: Stop at the remote rewrite gate**

Explain that publishing the scrubbed history requires a force push and that collaborators must re-clone or hard-reset. Ask for immediate explicit approval before any force push. Do not include a push command in an executed batch.

---

## Final acceptance checklist

- [ ] Current tree contains only `training/`, `evaluation/`, `human_study/`, `data/`, `tools/`, `tests/`, `docs/`, and root publication files.
- [ ] RAG, rebuttal, archived runs, IDE files, unrelated PDFs, and obsolete root experiments are absent.
- [ ] QA-Pairs, Pref-Human, final run-9/run-10 metrics, teacher-robustness outputs, human-validation outputs, and factuality outputs remain.
- [ ] README commands and path mappings are accurate.
- [ ] Offline analyses and repository tests pass.
- [ ] Current-tree and rewritten-history secret scans pass.
- [ ] Ignored local config remains untracked and contains no literal credentials.
- [ ] No commit or push occurred without the user’s separate authorization.
