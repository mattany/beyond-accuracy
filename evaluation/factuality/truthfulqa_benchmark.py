# -*- coding: utf-8 -*-
"""
TruthfulQA Benchmark for LLaMA 3.1 Models
=========================================
Run this in Google Colab with a GPU runtime.
Copy cells into a Jupyter notebook as needed.

Features:
- Saves after EVERY question for interruption resilience
- Auto-resumes from checkpoint if interrupted
- Per-question results stored in CSV
"""

# %%
# Install Dependencies
# !pip install datasets transformers accelerate bitsandbytes
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft triton

# %%
# Mount Google Drive & Setup
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Set HuggingFace cache to Google Drive
os.environ['HF_HOME'] = '/content/beyond-accuracy/.cache/huggingface'

# Output directory for TruthfulQA results
OUTPUT_DIR = '/content/beyond-accuracy/evaluation/factuality/truthfulqa_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Login to HuggingFace (for gated models)
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

# %%
# Define Models to Evaluate
# Only unprompted models for TruthfulQA sanity check
# Using 4-bit quantized versions for memory efficiency

MODELS_TO_EVALUATE = [
    {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "hf_model_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "is_lora": False,
        "description": "Base LLaMA 3.1 8B (unprompted)"
    },
    {
        "name": "SciComma-3.1-8B",
        "hf_model_id": "mattany/SciComma-3.1-8B-Instruct-lora",
        "is_lora": True,
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "description": "SciComma Synthetic SFT 8B (unprompted)"
    },
    {
        "name": "SciComma-3.1-8B-DPO",
        "hf_model_id": "maxsbr/SciComma-3.1-8B-Instruct-lora-DPO",
        "is_lora": True,
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "description": "SciComma Synthetic DPO 8B (unprompted)"
    },
    {
        "name": "organic_sft",
        "hf_model_id": "mattany/organic-sft-3.1-8B-lora",  # Upload first using upload_organic_sft.py
        "is_lora": True,
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Human-SFT 8B (unprompted)"
    },
    {
        "name": "organic_dpo",
        "hf_model_id": "mattany/organic-dpo-3.1-8B-lora",  # Upload first using upload_organic_dpo.py
        "is_lora": True,
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "SFT+Human-DPO 8B (unprompted)"
    },
]

# %%
# Load TruthfulQA Dataset
from datasets import load_dataset

def load_truthfulqa_dataset():
    """Load TruthfulQA MC dataset from HuggingFace."""
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    print(f"✓ Loaded TruthfulQA: {len(dataset)} questions")
    return dataset

# %%
# Define Model Loading
import torch
from unsloth import FastLanguageModel

def load_model(model_config: dict, max_seq_length: int = 512):
    """Load a model for evaluation."""
    print(f"Loading model: {model_config['name']}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['hf_model_id'],
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    print(f"✓ Loaded model: {model_config['name']}")
    
    return model, tokenizer


def unload_model(model, tokenizer):
    """Clear GPU memory."""
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("✓ Cleared GPU memory")

# %%
# MC2 Scoring Function

def compute_mc2_score(model, tokenizer, question: str, choices: list, labels: list):
    """
    Compute MC2 score for a single question.
    
    MC2: Probability mass on correct answers (can have multiple correct).
    Score = sum(probs of correct answers) / sum(probs of all answers)
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        question: The question text
        choices: List of answer choices
        labels: List of 0/1 indicating correct (1) answers
    
    Returns:
        MC2 score (float between 0 and 1)
    """
    # Compute log probability for each choice
    log_probs = []
    
    for choice in choices:
        # Format: "Q: {question}\nA: {choice}"
        prompt = f"Q: {question}\nA: {choice}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Negative log likelihood -> convert to log prob
            log_prob = -outputs.loss.item() * inputs["input_ids"].shape[1]
            log_probs.append(log_prob)
    
    # Convert to probabilities (softmax over log probs)
    log_probs = np.array(log_probs)
    max_log_prob = np.max(log_probs)
    probs = np.exp(log_probs - max_log_prob)  # Numerical stability
    probs = probs / probs.sum()  # Normalize
    
    # MC2: sum of probs on correct answers
    labels = np.array(labels)
    mc2_score = probs[labels == 1].sum()
    
    return float(mc2_score)


def compute_mc1_score(model, tokenizer, question: str, choices: list, labels: list):
    """
    Compute MC1 score for a single question.
    
    MC1: 1 if highest probability answer is correct, 0 otherwise.
    (Only considers the single "best" correct answer - first one with label=1)
    
    Returns:
        MC1 score (0 or 1)
    """
    log_probs = []
    
    for choice in choices:
        prompt = f"Q: {question}\nA: {choice}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            log_prob = -outputs.loss.item() * inputs["input_ids"].shape[1]
            log_probs.append(log_prob)
    
    # MC1: Check if argmax is a correct answer
    best_idx = np.argmax(log_probs)
    mc1_score = 1 if labels[best_idx] == 1 else 0
    
    return mc1_score

# %%
# Checkpoint Management

def get_checkpoint_path(model_name: str, output_dir: str) -> str:
    """Get path to checkpoint file for a model."""
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    return os.path.join(output_dir, f"checkpoint_{safe_name}.csv")


def load_checkpoint(model_name: str, output_dir: str) -> pd.DataFrame:
    """Load existing checkpoint if it exists."""
    checkpoint_path = get_checkpoint_path(model_name, output_dir)
    
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        print(f"✓ Loaded checkpoint: {len(df)} questions already completed")
        return df
    else:
        print("No checkpoint found, starting fresh")
        return pd.DataFrame()


def save_checkpoint(df: pd.DataFrame, model_name: str, output_dir: str):
    """Save checkpoint to disk."""
    checkpoint_path = get_checkpoint_path(model_name, output_dir)
    df.to_csv(checkpoint_path, index=False)


def get_completed_indices(checkpoint_df: pd.DataFrame) -> set:
    """Get set of question indices already completed."""
    if checkpoint_df.empty:
        return set()
    return set(checkpoint_df['question_idx'].tolist())

# %%
# Main Evaluation Loop (with per-question saves)

def evaluate_model_truthfulqa(
    model_config: dict,
    dataset,
    output_dir: str,
    mode: str = "mc2",  # "mc1" or "mc2"
    save_every: int = 1,  # Save after every N questions
):
    """
    Evaluate a model on TruthfulQA with interruption resilience.
    
    Saves results after every question (or every N questions).
    Automatically resumes from checkpoint if interrupted.
    """
    model_name = model_config['name']
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint_df = load_checkpoint(model_name, output_dir)
    completed_indices = get_completed_indices(checkpoint_df)
    
    # Check if already complete
    if len(completed_indices) >= len(dataset):
        print(f"✓ Model already fully evaluated ({len(completed_indices)} questions)")
        return checkpoint_df
    
    # Load model
    model, tokenizer = load_model(model_config)
    
    # Initialize results list from checkpoint
    results = checkpoint_df.to_dict('records') if not checkpoint_df.empty else []
    
    # Score function
    score_fn = compute_mc2_score if mode == "mc2" else compute_mc1_score
    
    # Evaluate each question
    start_time = datetime.now()
    questions_evaluated = 0
    
    try:
        for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
            # Skip if already completed
            if idx in completed_indices:
                continue
            
            item = dataset[idx]
            question = item['question']
            
            # MC format: choices in mc1_targets and mc2_targets
            # mc1_targets: {'choices': [...], 'labels': [0,0,1,0,...]} - single correct
            # mc2_targets: {'choices': [...], 'labels': [1,1,0,0,...]} - multiple correct possible
            
            if mode == "mc2":
                choices = item['mc2_targets']['choices']
                labels = item['mc2_targets']['labels']
            else:
                choices = item['mc1_targets']['choices']
                labels = item['mc1_targets']['labels']
            
            # Compute score
            try:
                score = score_fn(model, tokenizer, question, choices, labels)
            except Exception as e:
                print(f"\n⚠️ Error on question {idx}: {e}")
                score = 0.0  # Default to 0 on error
            
            # Store result
            result = {
                'question_idx': idx,
                'question': question,
                'category': item.get('category', 'unknown'),
                f'{mode}_score': score,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(result)
            questions_evaluated += 1
            
            # Save checkpoint
            if questions_evaluated % save_every == 0:
                checkpoint_df = pd.DataFrame(results)
                save_checkpoint(checkpoint_df, model_name, output_dir)
        
        # Final save
        checkpoint_df = pd.DataFrame(results)
        save_checkpoint(checkpoint_df, model_name, output_dir)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted! Saving checkpoint...")
        checkpoint_df = pd.DataFrame(results)
        save_checkpoint(checkpoint_df, model_name, output_dir)
        print(f"✓ Checkpoint saved: {len(results)} questions completed")
        raise
    
    finally:
        # Always unload model
        unload_model(model, tokenizer)
    
    # Compute final statistics
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    final_df = pd.DataFrame(results)
    overall_score = final_df[f'{mode}_score'].mean()
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETED: {model_name}")
    print(f"✓ Overall {mode.upper()} Score: {overall_score:.4f}")
    print(f"✓ Questions: {len(final_df)}")
    print(f"✓ Runtime: {runtime:.1f}s ({runtime/len(final_df):.2f}s per question)")
    print(f"{'='*60}")
    
    return final_df

# %%
# Run All Models

def run_all_models(models: list, output_dir: str, mode: str = "mc2"):
    """Run TruthfulQA on all models with checkpointing."""
    
    # Load dataset once
    dataset = load_truthfulqa_dataset()
    
    all_results = {}
    summary_data = []
    
    for model_config in models:
        try:
            results_df = evaluate_model_truthfulqa(
                model_config=model_config,
                dataset=dataset,
                output_dir=output_dir,
                mode=mode,
                save_every=1,  # Save after EVERY question
            )
            
            # Store results
            model_name = model_config['name']
            all_results[model_name] = results_df
            
            # Summary stats
            overall_score = results_df[f'{mode}_score'].mean()
            summary_data.append({
                'model_name': model_name,
                'description': model_config['description'],
                f'{mode}_score': overall_score,
                'n_questions': len(results_df),
            })
            
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted. Progress saved in checkpoints.")
            break
        except Exception as e:
            print(f"❌ Error evaluating {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f"truthfulqa_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved: {summary_path}")
        
        # Also save latest
        latest_path = os.path.join(output_dir, "truthfulqa_summary_latest.csv")
        summary_df.to_csv(latest_path, index=False)
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
    
    return all_results, summary_data

# %%
# Execute Benchmark
if __name__ == "__main__":
    print("Starting TruthfulQA Benchmark (with interruption resilience)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models to evaluate: {len(MODELS_TO_EVALUATE)}")
    print("Progress will be saved after EVERY question.\n")
    
    # Run MC2 mode (recommended - handles multiple valid answers)
    all_results, summary = run_all_models(
        models=MODELS_TO_EVALUATE,
        output_dir=OUTPUT_DIR,
        mode="mc2"
    )

# %%
# (Optional) Resume from checkpoint
# If interrupted, just re-run CELL 10 - it will automatically resume!

# %%
# (Optional) Run MC1 Mode as well
# Uncomment to also run MC1 mode (stricter - single correct answer)
# all_results_mc1, summary_mc1 = run_all_models(
#     models=MODELS_TO_EVALUATE,
#     output_dir=OUTPUT_DIR,
#     mode="mc1"
# )

# %%
# (Optional) Analyze results by category
def analyze_by_category(checkpoint_path: str, mode: str = "mc2"):
    """Analyze scores broken down by TruthfulQA category."""
    df = pd.read_csv(checkpoint_path)
    
    category_scores = df.groupby('category')[f'{mode}_score'].agg(['mean', 'std', 'count'])
    category_scores = category_scores.sort_values('mean', ascending=False)
    
    print("\nScores by Category:")
    print("="*60)
    print(category_scores.to_string())
    
    return category_scores

# Example usage:
# analyze_by_category(get_checkpoint_path("Meta-Llama-3.1-8B-Instruct", OUTPUT_DIR))

