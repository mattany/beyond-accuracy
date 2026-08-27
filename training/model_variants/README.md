# Model variants

`naive_dpo.py` and `organic_sft.py` are retained as standalone Python command-line
programs. They preserve the model-loading, resumable generation, and source
Colab provenance without notebook-only `%pip`, `!pip`, or Drive-mount syntax.

Install `torch`, `transformers`, `peft`, and `pandas` before running either
program. In Colab, mount Drive in a separate cell when an adapter is stored
there, then pass that location explicitly:

```bash
python training/model_variants/organic_sft.py \
  --adapter "/content/drive/My Drive/models/organic_sft"
```

Both programs default to the retained QA-Pairs test set and publication
generation directories. `--input`, `--output`, and `--base-model` can override
those repository-relative defaults.
