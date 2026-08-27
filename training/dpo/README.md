# Preference-data preparation

`preference_dataset_generation.py` converts per-answer rubric scores and model
outputs into pairwise `prompt`, `chosen`, and `rejected` records. Inputs and the
output path are explicit command-line arguments because the historical
intermediate files are not part of the retained publication artifacts:

```bash
python training/dpo/preference_dataset_generation.py \
  --scores path/to/per_question_scores.csv \
  --answers path/to/model_answers.csv \
  --output path/to/preference_pairs.csv
```

This repository does not contain an end-to-end DPO trainer or optimizer. The
script documents preference-data construction only.

Publishing is opt-in. Pass both `--push-to-hub` and an explicit
`--hub-repo ORGANIZATION/DATASET`; no personal Hub destination is embedded in
the generator.
