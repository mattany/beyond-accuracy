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
4. `parse_output.py` - used for converting the openai api output e.g. `output_3.jsonl` back to a python dict e.g. `output_dict_3.py` (I used the openAI UI to get the outputs, not the cli like with the upload).
5. `merge.py` - used to merge between the questions and the answers based on index. E.g. uses `outputs/output_dict_3.py` to create `outputs/numbered_questions_with_3rd_batch_answers` and then merges all into `final.csv`
