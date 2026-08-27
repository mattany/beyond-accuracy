# In this repo

1. Setup the env:
   ```
   ./setup_env.sh
   source .venv/bin/activate
   ```

2. Add a config file to `training/data_generation/config.py` with your
   credentials:
   ```python
   # OpenAI teacher (default provider)
   OPENAI_API_KEY = "sk-proj-XXXXXXXXX"

   # Kimi / Moonshot teacher (only needed when TEACHER_PROVIDER=kimi)
   MOONSHOT_API_KEY = "sk-XXXXXXXXX"
   # Optional overrides (defaults shown):
   # MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"  # use .cn for a China account
   # MOONSHOT_MODEL = "kimi-k2.6"
   ```

3. From the repo root:
   ```
   python -m training.data_generation.gen_batch
   python -m training.data_generation.upload_batch_file
   python -m training.data_generation.merge
   ```

## Choosing the teacher (provider)

All three steps read the `TEACHER_PROVIDER` env var (`openai` by default). Both
providers speak the OpenAI Batch wire format; only the endpoint, model, and
request body differ. Each provider writes to its own files so runs never clobber
each other:

| Provider | API key in config | Model | Answers CSV | Output dir |
|----------|-------------------|-------|-------------|------------|
| `openai` (default) | `OPENAI_API_KEY` | `gpt-5-2025-08-07` | `ask_science_gpt_5_answers.csv` | `GPT5_outputs/` |
| `kimi` | `MOONSHOT_API_KEY` | `kimi-k2.6` | `ask_science_kimi_answers.csv` | `kimi_outputs/` |

Run against Kimi by prefixing each command:
```
TEACHER_PROVIDER=kimi python -m training.data_generation.gen_batch
TEACHER_PROVIDER=kimi python -m training.data_generation.upload_batch_file
TEACHER_PROVIDER=kimi python -m training.data_generation.merge
```

Note: Kimi's batch API rejects sampling params (`temperature`, `top_p`, `n`,
penalties) and uses `max_tokens`, so the generated request body drops
`reasoning_effort`/`max_completion_tokens` automatically for that provider.

- `gen_batch.py` - generates the batch files for the active provider.
- `upload_batch_file.py` - uploads/polls/downloads via the provider's Batch API.
- `merge.py` - merges questions and answers based on `custom_id`.
- `cancel_all_batches.py` - cancels in-flight batches for the active provider.
