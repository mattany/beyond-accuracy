# In this repo

1. Setup the env:
   ```
   ./setup_env.sh
   source .venv/bin/activate
   ```

2. Add a config file to `batch_file_gen/config.py` with your OpenAI credentials:
   ```
   OPENAI_API_KEY = "sk-proj-XXXXXXXXX"
   PROJECT_DIR = "/path/to/thesis"
   ```

3. From the repo root:
   ```
   python -m SFT.batch_file_gen.gen_batch
   python -m SFT.batch_file_gen.upload_batch_file
   python -m SFT.batch_file_gen.merge
   ```

- `gen_batch.py` - generates the batch files to be sent to OpenAI.
- `upload_batch_file.py` - uploads your file to OpenAI using their API.
- `merge.py` - merges questions and answers based on `custom_id`.
