# In this repo

1. To use, add a config file to `batch_file_gen/config.py` with you opeai credentials: 
    ```
   OPENAI_PROJECT_ID = "proj_XXXXXXXXX"
   OPENAI_ORG_ID = "org-XXXXXXXXX"
   OPENAI_API_KEY = "sk-proj-XXXXXXXXX"
   ```

2. `gen_batch.py` - generates the batch files to be sent to open ai (e.g. `batch_file_2.jsonl`). 
   - Supports configurable batch size `requests_per_file` and configurable request size (`mini_batch_size`).
3. `upload_batch_file.py` - uploads your file to openai using their api.
5. `merge.py` - used to merge between the questions and the answers based on the index supplied to the `custom_id` field in the request to open AI, and the row in the csv.