# Publication Repository Cleanup Design

## Goal

Turn the renamed `beyond-accuracy` repository into a focused, documented, and credential-free companion to the ACL paper. The public tree should contain only code, data, and results needed to reproduce or inspect claims made in the paper.

## Scope and retention rule

Retain an artifact only when it supports a method, dataset, figure, table, appendix result, or reproducibility step in the ACL paper. Keep all tracked datasets, intermediate outputs, and final results needed for exact reruns of those retained pipelines.

Remove:

- the complete `RAG/` and `rebuttal/` areas;
- RAG-specific code and data nested elsewhere;
- IDE metadata and operating-system artifacts;
- unrelated source PDFs and abandoned root-level notebooks/scripts;
- duplicate, superseded, archived, or exploratory experiments not represented in the paper.

Before deletion, map ambiguous artifacts to the paper. Git history remains the recovery mechanism for intentionally removed research artifacts, subject to credential scrubbing.

## Public structure

Reorganize retained artifacts into:

- `training/`: QA-Pairs preparation, generation, SFT, and DPO code and inputs;
- `evaluation/`: science-communication rubrics, model benchmarking, aggregation, and factuality checks;
- `human_study/`: judge validation, annotation-processing, and human-preference analyses;
- `data/`: shared paper datasets and intermediates that do not naturally belong to one pipeline;
- root documentation and configuration: `README.md`, `LICENSE`, `.env.example`, `.gitignore`, and component-specific dependency metadata.

Update imports, paths, commands, and documentation after moves. Keep distinct dependency environments where training and evaluation stacks are incompatible.

## README and reproducibility

The root README will include:

1. paper overview and repository status;
2. a map from repository components to paper sections, figures, tables, and appendices;
3. prerequisites and component-specific installation commands;
4. required environment variables, with no secret values;
5. locations and provenance of QA-Pairs, Pref-Human, models, and retained outputs;
6. ordered reproduction instructions:
   - data preparation,
   - SFT and DPO training,
   - rubric evaluation,
   - judge validation,
   - human-preference analysis,
   - factuality checks;
7. expected inputs and outputs for each stage;
8. citation and authorship information;
9. licensing and third-party data/model terms.

Repository code will use the MIT License. Dataset and model usage terms will be documented separately rather than implied to fall under the code license.

## Credential handling

For retained code, replace embedded credentials with environment-variable loading and list variable names in `.env.example`. Delete credential-bearing files when the complete file is outside publication scope.

`Benchmarking/deep_eval/config.py` is ignored and has never been tracked. Convert the local file to environment-variable loading for local safety, but do not include it in history-rewrite inputs.

Identify confirmed secrets in tracked files and rewrite all local Git refs to remove their literal values. The history rewrite must otherwise preserve historical repository content. Publication-irrelevant paths will be deleted from the current tree in a normal cleanup commit, so their earlier versions remain recoverable without restoring exposed credentials.

Do not force-push rewritten history without a separate, immediate user confirmation. Before that gate, retain the existing remote as the fallback. Document that collaborators must re-clone or reset after a rewritten remote is published.

## Verification

Verify the cleanup with:

- a current-tree secret scan;
- a full rewritten-history secret scan;
- a check that ignored local configuration is not tracked;
- a search for stale paths, references to deleted experiments, and old repository names;
- Python syntax checks and import checks where dependencies permit;
- representative non-GPU, non-paid-API analysis commands;
- checks that all README commands and documented inputs exist;
- `git status` review to ensure unrelated local work is not accidentally removed.

GPU training and paid API calls will not be rerun locally. Their commands, expected outputs, and prerequisites will be documented instead.

## Safety and completion criteria

The cleanup is complete when:

- every retained top-level area maps directly to the ACL paper;
- RAG, rebuttal, and unrelated experiments are absent;
- required tracked data and results remain available;
- a new reader can locate and run each paper pipeline from the README;
- no confirmed credential remains in the current tree or rewritten history;
- local verification passes, with external/GPU-only limitations explicitly documented;
- remote history remains untouched until the user approves the required force push.
