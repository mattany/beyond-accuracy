import os
import asyncio
from tqdm import tqdm
from config import PROJECT_DIR, OPENAI_API_KEY
import logging
import json
from pathlib import Path
import argparse
from functools import partial

from custom_metrics.aggregate_v2 import RUN_NUMBER
from deepeval.metrics import GEval

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def create_geval_factory(metric_instance):
    """
    Create a factory function that produces fresh GEval instances
    with the same configuration as the given instance.
    
    This is needed to avoid race conditions when running metrics in parallel.
    """
    if not isinstance(metric_instance, GEval):
        # Non-GEval metrics (e.g., ReadabilityMetric) don't have the race condition
        return lambda: metric_instance
    
    # Extract configuration from existing instance
    return partial(
        GEval,
        name=metric_instance.name,
        evaluation_params=metric_instance.evaluation_params,
        criteria=getattr(metric_instance, 'criteria', None),
        evaluation_steps=getattr(metric_instance, 'evaluation_steps', None),
        model=getattr(metric_instance, 'model_name', 'gpt-4o'),  # fallback to default
        threshold=metric_instance.threshold,
        async_mode=getattr(metric_instance, 'async_mode', True),
        strict_mode=getattr(metric_instance, 'strict_mode', False),
        verbose_mode=getattr(metric_instance, 'verbose_mode', False),
    )


GEVAL_RETRIES = 3
GEVAL_TIMEOUT = 10  # seconds
import pandas as pd
from deepeval.test_case import LLMTestCase
from custom_metrics.metrics import (
    # baram tsabari metrics
    # explanation_type_metric_explicit_v2,
    # connection_to_everyday_life_metric_explicit_v2,
    scaffolding_metric_v2,
    metaphor_metric_explicit_v8,
    # content_units_metric_explicit,
    # content_units_metric_explicit_v2,
    humor_metric_explicit_v5,
    # analogy_metric_explicit,
    analogy_metric_explicit_v2,
    # correctness_metric_explicit,
    jargon_metric,
    # zemla metrics
    # internal_coherence_metric_explicit,
    # completeness_metric_explicit,
    # alternatives_metric_explicit,
    # articulation_metric_explicit,
    # perceived_truth_metric_explicit,
    # readability metrics
    flesch_kincaid,
    flesch_reading_ease,
    dale_chall,
    ari,
)


# Suppress DEBUG logs from deepeval
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("main_logger").setLevel(logging.INFO)

async def get_metric_scores_for_model(
    eval_df,
    model_name,
    metric_factory,
    semaphore,
    reference_column=None,
    pbar=None,
    position=None,
    checkpoint_dir=None,
    metric_name=None,
):
    """
    Get scores for a model with checkpoint support.
    
    Args:
        checkpoint_dir: Directory to save intermediate results (e.g., /tmp/run_9/metaphor_v8/)
        metric_name: Name of the metric being evaluated (for checkpoint file naming)
    """
    # Try to load existing checkpoint
    checkpoint_path = None
    if checkpoint_dir and metric_name:
        checkpoint_path = Path(checkpoint_dir) / f"{model_name}.json"
        if checkpoint_path.exists():
            print(f"Loading checkpoint for {model_name} from {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                scores = checkpoint_data.get("scores", {})
                # Convert string keys back to int
                scores = {int(k): v for k, v in scores.items()}
                reasons = checkpoint_data.get("reasons", {})
                reasons = {int(k): v for k, v in reasons.items()}
                print(f"Resumed {len(scores)}/{len(eval_df)} completed for {model_name}")
        else:
            scores = {}
            reasons = {}
    else:
        scores = {}
        reasons = {}
    
    # Identify rows that still need processing
    remaining_rows = [(index, row) for index, row in eval_df.iterrows() if index not in scores]
    
    if not remaining_rows:
        print(f"All rows already completed for {model_name}")
        return {"model_name": model_name, "scores": scores, "reasons": reasons}
    
    # Each row gets its own fresh metric instance via the factory
    # This allows full parallelism with no race conditions on metric.reason
    tasks = [
        evaluate_row(
            index,
            metric_factory,  # Pass factory, each row creates its own instance
            model_name,
            reasons,
            reference_column,
            row,
            scores,
            semaphore,
            pbar,
            checkpoint_path,
        )
        for index, row in remaining_rows
    ]
    await asyncio.gather(*tasks)
    
    # Save final checkpoint
    if checkpoint_path:
        save_checkpoint(checkpoint_path, scores, reasons)
    
    return {"model_name": model_name, "scores": scores, "reasons": reasons}


def save_checkpoint(checkpoint_path, scores, reasons):
    """Save intermediate results to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump({
            "scores": {str(k): v for k, v in scores.items()},
            "reasons": {str(k): v for k, v in reasons.items()}
        }, f, indent=2)


async def evaluate_row(
    index,
    metric_factory,
    model_name,
    reasons,
    reference_column,
    row,
    scores,
    semaphore,
    pbar=None,
    checkpoint_path=None,
):
    async with semaphore:
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row[model_name],
            expected_output=row[reference_column] if reference_column else None,
        )
        
        # Create a fresh metric instance for this row - avoids race conditions
        metric_instance = metric_factory() if callable(metric_factory) else metric_factory
        
        success = False
        score_result = None
        reason_result = None
        
        for i in range(GEVAL_RETRIES):
            try:
                score_result = await asyncio.wait_for(
                    metric_instance.a_measure(test_case),
                    timeout=GEVAL_TIMEOUT
                )
                # Capture reason from our private instance - no race condition possible
                reason_result = getattr(metric_instance, "reason", None)
                success = True
                break
            except asyncio.TimeoutError:
                print(
                    f"Question index: {index}. Try #{i + 1}. Timeout after {GEVAL_TIMEOUT}s. Retrying..."
                )
                continue
            except ValueError:
                print(
                    f"Question index: {index}. Try #{i + 1}. Encountered invalid JSON. Retrying..."
                )
                continue
            except Exception as e:
                print(
                    f"Question index: {index}. Try #{i + 1}. Unexpected error: {type(e).__name__}: {e}. Retrying..."
                )
                continue
        
        if success:
            scores[index] = score_result if score_result is not None else metric_instance.score
            if reason_result:
                print(f"Row {index} reason: {reason_result[:50]}...")
                reasons[index] = reason_result
            
            # Save checkpoint after every successful evaluation
            if checkpoint_path:
                save_checkpoint(checkpoint_path, scores, reasons)
        else:
            print(f"Warning: row #{index} failed for model {model_name} after {GEVAL_RETRIES} retries")
        if pbar:
            pbar.set_description(f"added result for model {model_name}")
            pbar.update(1)


# async def generate_metric_report_with_reference_models(
#     metrics,
#     evaluation_dataset,
#     models,
#     reference_models=None,
#     models_to_evaluate=None,
#     run_number=0,
# ):
#     if not models_to_evaluate:
#         models_to_evaluate = models
#     eval_df = pd.read_csv(evaluation_dataset)
#     for metric, metric_function in metrics.items():
#         for model in models_to_evaluate:
#             for reference_model in reference_models:
#                 if reference_model == model:
#                     continue
#                 print(
#                     f"Evaluating {metric} for model with reference model {reference_model}"
#                 )
#                 await update_or_insert_score_column(
#                     eval_df,
#                     output_path=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/{metric}_reference-{reference_model}_evaluation_scores.csv",
#                     model_name=model,
#                     metric_function=metric_function,
#                     semaphore=False,
#                     reference_column=models[reference_model],
#                 )


async def generate_metric_report(
    metrics,
    evaluation_dataset,
    models_to_evaluate,
    run_number=0,
    overwrite_metrics=None,
):
    """
    Generate metric evaluation report with smart skipping and checkpoint recovery.
    
    Args:
        metrics: Dict of metric_name -> metric_function
        evaluation_dataset: Path to evaluation CSV
        models_to_evaluate: List of model names to evaluate
        run_number: Run number for output directory
        overwrite_metrics: Set of metric names to force re-run (ignore existing results)
    """
    eval_df = pd.read_csv(evaluation_dataset)
    expected_rows = len(eval_df)
    overwrite_metrics = overwrite_metrics or set()
    
    for metric, metric_function in metrics.items():
        print(f"\n{'='*60}")
        print(f"Processing metric: {metric}")
        print(f"{'='*60}\n")
        
        # Check if this metric should be force-overwritten
        force_overwrite = metric in overwrite_metrics
        if force_overwrite:
            print(f"⚠️  FORCE OVERWRITE enabled for {metric} - ignoring existing results\n")
        
        # Check which models already have complete results in the final CSV
        output_path = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/{metric}.csv"
        models_to_process = []
        
        if os.path.exists(output_path) and not force_overwrite:
            existing_df = pd.read_csv(output_path)
            # Check if dataset size changed - existing rows keep their indices, new rows are appended
            if len(existing_df) != expected_rows:
                print(f"ℹ️  Dataset size changed ({len(existing_df)} -> {expected_rows}). Will process new rows.")
                models_to_process = models_to_evaluate.copy()
            else:
                for model in models_to_evaluate:
                    score_col = f"{model}__score"
                    if score_col in existing_df.columns:
                        # Check if the column is complete (no NaN values)
                        non_null_count = existing_df[score_col].notna().sum()
                        if non_null_count == expected_rows:
                            print(f"✓ Skipping {model} - already complete ({non_null_count}/{expected_rows} rows)")
                            continue
                        else:
                            print(f"○ Re-running {model} - incomplete ({non_null_count}/{expected_rows} rows)")
                            models_to_process.append(model)
                    else:
                        print(f"○ Running {model} - not found in existing results")
                        models_to_process.append(model)
        else:
            if force_overwrite and os.path.exists(output_path):
                print(f"○ Force overwrite: re-running all models despite existing results")
            else:
                print(f"○ No existing results found, running all models")
            models_to_process = models_to_evaluate.copy()
        
        if not models_to_process:
            print(f"\n✓ All models already complete for {metric}, skipping metric entirely\n")
            continue
        
        print(f"\nProcessing {len(models_to_process)}/{len(models_to_evaluate)} models for {metric}\n")
        
        # Setup checkpoint directory for this metric
        checkpoint_dir = Path(f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/.checkpoints/{metric}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(40)
        # Create a factory to produce fresh metric instances per model
        # This avoids race conditions on metric.reason when models run in parallel
        metric_factory = create_geval_factory(metric_function)
        pbar = tqdm(total=len(models_to_process) * len(eval_df))
        tasks = [
            get_metric_scores_for_model(
                eval_df,
                model_name=model,
                metric_factory=metric_factory,
                semaphore=semaphore,
                pbar=pbar,
                position=i + 1,
                checkpoint_dir=checkpoint_dir,
                metric_name=metric,
            )
            for i, model in enumerate(models_to_process)
        ]
        results = await asyncio.gather(*tasks)
        
        # Load or create output dataframe
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            # Check if the existing output matches the current eval dataset size
            if len(output_df) != len(eval_df):
                print(f"⚠️  Dataset size changed ({len(output_df)} -> {len(eval_df)}). Starting fresh for {metric}.")
                output_df = pd.DataFrame()
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df = pd.DataFrame()
        
        # Always include question column for alignment verification
        if 'question' not in output_df.columns and 'question' in eval_df.columns:
            output_df.insert(0, 'question', eval_df['question'].values)
        
        # Update with new results (only for processed models)
        for result in results:
            model_name, scores_dict, reasons_dict = (
                result["model_name"],
                result["scores"],
                result["reasons"],
            )
            scores = [scores_dict.get(i, None) for i in range(len(eval_df))]
            reasons = [reasons_dict.get(i, None) for i in range(len(eval_df))]
            output_df[f"{model_name}__score"] = scores
            if any(reasons):
                output_df[f"{model_name}__reason"] = reasons
        
        output_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved final results to {output_path}")
        print(f"  Checkpoints saved in {checkpoint_dir}")
        print(f"  (You can delete checkpoints after successful completion)\n")


# def check_reasons(model_map, model, metric, run_number=0):
#     scores_df = pd.read_csv(
#         f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/{metric}_evaluation_scores_run.csv"
#     )
#     qa_df = pd.read_csv(
#         f"{PROJECT_DIR}/Benchmarking/deep_eval/data/evaluation_dataset.csv"
#     )
#     score_col = f"{metric}_score__{model}"
#     reason_col = f"{metric}_reason__{model}"
#     output_df = scores_df[[score_col, reason_col]].join(
#         qa_df.rename(columns={model_map[model]: model})[["question", model]]
#     )
#     output_df.to_csv(
#         f"{PROJECT_DIR}/Benchmarking/deep_eval/data/model_metric_specific/run_{run_number}/{model}_{metric}_evaluation_scores_with_reasons.csv",
#         index=False,
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run metric evaluation with checkpoint recovery')
    parser.add_argument(
        '--overwrite-metrics',
        type=str,
        default='',
        help='Comma-delimited list of metric names to force overwrite (e.g., "metaphor_v8,humor_v5")'
    )
    args = parser.parse_args()
    
    # Parse overwrite metrics into a set
    overwrite_metrics = set()
    if args.overwrite_metrics:
        overwrite_metrics = set(m.strip() for m in args.overwrite_metrics.split(',') if m.strip())
        print(f"\n⚠️  Force overwrite enabled for metrics: {', '.join(overwrite_metrics)}\n")
    
    # RAG
    # generate_metric_report(
    #     geval={
    #         'correctness': correctness_metric
    #     },
    #     evaluation_dataset=f'{PROJECT_DIR}/Benchmarking/deep_eval/RAG/data/joined_answers.csv',
    #     MODEL_MAP={
    #         'gpt_4_turbo': 'gpt_4_turbo',
    #         'gpt_4o': 'gpt_4o_1',
    #         'gpt_4o_validation': 'gpt_4o_2',
    #         'llama_2_base': 'llama_2',
    #         'llama_2_sft': 'llama_2_sft',
    #     },
    #     reference_models=['llama_2_base'],
    #     models_to_evaluate=['llama_2_sft', 'gpt_4o_validation', 'llama_2_base', 'gpt_4o', 'gpt_4_turbo']
    # )

    # TEST SET
    models = [
        # "human"
        # "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        # "Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt",
        # "gpt-3.5-turbo-0125",
        # "gpt-3.5-turbo-0125_cot",
        # "SciComma-3.1-8B_y",
        # "SciComma-3.1-8B_prompt",
        # "scicomma-3.1-dpo",
        # "scicomma-3.1-dpo_prompt",
        # "organic_dpo",
        # "organic_dpo_prompt",
        # "organic_sft",
        # "organic_sft_prompt",
        "gpt-5",
        "kimi-k2-thinking",
        "grok-4",
        # "gpt-4o",
        # "llama3.1-instruct",
        # "gpt-4",
        # "llama-3.3-70b",
        # "SciComma-3.3-70B",
        # "o1",
        # "claude-3-7-sonnet-20250219",
        # "llama-2-7b",
        # "SciComma-2-7b",
        # "scicomma-3.1-dpo_real_256",
        # "scicomma-3.1-dpo_real_512",
        # "scicomma-3.1-dpo_real_512_short",
        # "scicomma-3.1-dpo_full"
    ]
    asyncio.run(
        generate_metric_report(
            metrics={
                # ## BARAM TSABARI METRICS
                "jargon": jargon_metric,
                "metaphor_v8": metaphor_metric_explicit_v8,
                "humor_v5": humor_metric_explicit_v5,
                "analogy_v2": analogy_metric_explicit_v2,
                # "explanation_type_v2": explanation_type_metric_explicit_v2,
                # "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
                #
                # # ## READING EASE
                "scaffolding_v2": scaffolding_metric_v2,
                "flesch_kincaid": flesch_kincaid,
                "flesch_reading_ease": flesch_reading_ease,
                "dale_chall": dale_chall,
                "ari": ari,

                ### DEPRECATED
                ## CORRECTNESS METRICS
                # "correctness_explicit": correctness_metric_explicit,
                # ## ZEMLA METRICS
                # "internal_coherence_v2": internal_coherence_metric_explicit_v2,
                # "completeness_v2": completeness_metric_explicit_v2,
                # "alternatives_explicit": alternatives_metric_explicit,  # unreliable for LLM measurement
                # "articulation_v2": articulation_metric_explicit_v2,
                # "perceived_truth_explicit": perceived_truth_metric_explicit,  # unreliable for LLM measurement
            },
            evaluation_dataset=f"{PROJECT_DIR}/Benchmarking/deep_eval/data/test_data/corrected_evaluation_dataset.csv",
            models_to_evaluate=models,
            run_number=RUN_NUMBER,
            overwrite_metrics=overwrite_metrics,
        )
    )

    # check_reasons(
    #     model_map=MODEL_MAP,
    #     model='llama_2_sft',
    #     metric='completeness_explicit',
    #     run_number=0
    # )
