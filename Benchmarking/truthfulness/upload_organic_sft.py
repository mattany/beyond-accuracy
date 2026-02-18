# -*- coding: utf-8 -*-
"""Upload Organic SFT Model to HuggingFace

Run this in Google Colab to upload the organic_sft LoRA adapter to HuggingFace.
"""

# %%
# Install Dependencies
# !pip install torch transformers peft huggingface_hub

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/thesis/huggingface_cache'

# %%
# Login to HuggingFace
from huggingface_hub import notebook_login, HfApi
notebook_login()

# %%
# Configuration
# Local path to the LoRA adapter
LOCAL_ADAPTER_PATH = "/content/drive/My Drive/models/organic_sft"

# HuggingFace repository name (change 'mattany' to your username if needed)
HF_REPO_NAME = "mattany/human-sft-3.1-8B-lora"

# Base model used for training
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# %%
# Verify the adapter exists
import os

if os.path.exists(LOCAL_ADAPTER_PATH):
    print(f"✓ Found adapter at: {LOCAL_ADAPTER_PATH}")
    print("\nContents:")
    for f in os.listdir(LOCAL_ADAPTER_PATH):
        size = os.path.getsize(os.path.join(LOCAL_ADAPTER_PATH, f))
        print(f"  - {f} ({size / 1024 / 1024:.2f} MB)")
else:
    raise FileNotFoundError(f"Adapter not found at {LOCAL_ADAPTER_PATH}")

# %%
# Load and verify the adapter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LOCAL_ADAPTER_PATH)
print("✓ Model loaded successfully!")

# Quick test
model.eval()
test_prompt = "What is photosynthesis?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(f"\nTest generation:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# %%
# Upload to HuggingFace
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create the repository if it doesn't exist
try:
    create_repo(HF_REPO_NAME, repo_type="model", exist_ok=True)
    print(f"✓ Repository created/verified: {HF_REPO_NAME}")
except Exception as e:
    print(f"Repository status: {e}")

# Upload the adapter files
print(f"\nUploading adapter to {HF_REPO_NAME}...")
api.upload_folder(
    folder_path=LOCAL_ADAPTER_PATH,
    repo_id=HF_REPO_NAME,
    repo_type="model",
)
print(f"✓ Upload complete!")

# %%
# Create a model card
MODEL_CARD = f"""---
tags:
- llama
- peft
- lora
- science-communication
base_model: {BASE_MODEL}
license: llama3.1
---

# Organic SFT (Human-SFT) - LLaMA 3.1 8B LoRA

This is a LoRA adapter for science communication, trained on **human-written** scientific explanations from Reddit r/AskScience.

## Model Description

- **Base Model:** `{BASE_MODEL}`
- **Training Data:** Human-written answers from r/AskScience (organic/real data)
- **Training Method:** Supervised Fine-Tuning (SFT) with LoRA
- **Purpose:** Generate high-quality scientific explanations

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "{BASE_MODEL}"
adapter = "{HF_REPO_NAME}"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)

# Generate
prompt = "What is photosynthesis?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

This model was trained as part of a research project comparing synthetic (GPT-generated) vs organic (human-written) training data for science communication.
"""

# Save and upload model card
with open("/tmp/README.md", "w") as f:
    f.write(MODEL_CARD)

api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id=HF_REPO_NAME,
    repo_type="model",
)
print(f"✓ Model card uploaded!")

print(f"\n{'='*60}")
print(f"✓ DONE! Model available at: https://huggingface.co/{HF_REPO_NAME}")
print(f"{'='*60}")

