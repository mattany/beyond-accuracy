# Evaluation model generation

This Poetry component contains the retained batch-generation scripts and
provider outputs used to add model answers to the evaluation dataset.

Run commands from the repository root:

```bash
poetry install --directory evaluation/model_generation
PYTHONPATH=. poetry --directory evaluation/model_generation run \
  python -m evaluation.model_generation.gpt --help
python -m evaluation.model_generation.add_gpt3_5_cot --help
```

Defaults resolve repository-relative inputs under `data/qa_pairs/` and
`evaluation/model_outputs/main/`. Provider credentials are supplied through
the environment-backed training generation utilities.
