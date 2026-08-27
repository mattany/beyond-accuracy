# TrustLLM factuality analysis

This component retains the auxiliary TrustLLM truthfulness JSON outputs and
their heatmap visualization. It is separate from the paper's TruthfulQA and
GPT-5.2 claim-level analyses.

Install and run from the component directory:

```bash
cd evaluation/factuality/trust_llm
poetry install
MPLBACKEND=Agg poetry run python heatmap.py
```

`heatmap.py` reads the bundled `eval_results/` directory relative to its own
file location. These results must not be attributed as provenance for the
missing GPT-5.2 atomic-claim analysis.
