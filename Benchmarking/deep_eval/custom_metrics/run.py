import os
import asyncio
from readability.exceptions import ReadabilityException
from tqdm import tqdm
from config import PROJECT_DIR, OPENAI_API_KEY
import logging

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
GEVAL_RETRIES = 3
import pandas as pd
from deepeval.test_case import LLMTestCase
from custom_metrics.metrics import (
    # baram tsabari metrics
    # explanation_type_metric_explicit,
    explanation_type_metric_explicit_v2,
    # connection_to_everyday_life_metric_explicit,
    connection_to_everyday_life_metric_explicit_v2,
    # metaphor_metric_explicit,
    metaphor_metric_explicit_v2,
    # content_units_metric_explicit,
    # content_units_metric_explicit_v2,
    # humor_metric_explicit,
    humor_metric_explicit_v2,
    # analogy_metric_explicit,
    analogy_metric_explicit_v2,
    # correctness_metric_explicit,
    jargon_metric,
    # NEW baram tsabari metrics (2012 paper)
    scaffolding_metric,
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
    metric_function,
    semaphore,
    reference_column=None,
    pbar=None,
    position=None,
):
    scores = {}
    reasons = {}
    tasks = [
        evaluate_row(
            index,
            metric_function,
            model_name,
            reasons,
            reference_column,
            row,
            scores,
            semaphore,
            pbar,
        )
        for index, row in eval_df.iterrows()
    ]
    await asyncio.gather(*tasks)
    return {"model_name": model_name, "scores": scores, "reasons": reasons}


async def evaluate_row(
    index,
    metric_function,
    model_name,
    reasons,
    reference_column,
    row,
    scores,
    semaphore,
    pbar=None,
):
    async with semaphore:
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row[model_name],
            expected_output=row[reference_column] if reference_column else None,
        )
        success = False
        for i in range(GEVAL_RETRIES):
            try:
                await metric_function.a_measure(test_case)
                success = True
                break
            except ValueError:
                print(
                    f"Question index: {index}. Try #{i + 1}. Encountered invalid JSON. Retrying..."
                )
                continue
            except ReadabilityException as e:
                print(
                    f"Question index: {index}. Try #{i + 1}. Ran into readability exception: {e}. Continuing"
                )
                break
        if success:
            scores[index] = metric_function.score
            if getattr(metric_function, "reason"):
                print("reason:", metric_function.reason)
                reasons[index] = metric_function.reason
        else:
            print(f"Warning: row #{index} failed for model {model_name}")
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
):
    eval_df = pd.read_csv(evaluation_dataset)
    for metric, metric_function in metrics.items():
        semaphore = asyncio.Semaphore(40)
        pbar = tqdm(total=len(models_to_evaluate) * len(eval_df))
        tasks = [
            get_metric_scores_for_model(
                eval_df,
                model_name=model,
                metric_function=metric_function,
                semaphore=semaphore,
                pbar=pbar,
                position=i + 1,
            )
            for i, model in enumerate(models_to_evaluate)
        ]
        results = await asyncio.gather(*tasks)
        output_path = (
            f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}/{metric}.csv"
        )
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df = pd.DataFrame()
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
        # "llama-2-7b",
        # "SciComma-2-7b",
        # "gpt-3.5-turbo-0125",
        # "gpt-4o",
        # "llama3.1-instruct",
        # "gpt-3.5-turbo-0125_cot",
        # "gpt-4",
        # "llama-3.3-70b",
        # "SciComma-3.3-70B",
        # "SciComma-3.1-8B",
        # "o1",
        # "claude-3-7-sonnet-20250219",
        "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "Meta-Llama-3.1-8B-Instruct-bnb-4bit_prompt",
        "SciComma-3.1-8B_y",
        "SciComma-3.1-8B_prompt",
        "human"
        # "scicomma-3.1-dpo",
        # "scicomma-3.1-dpo_prompt"
        # "scicomma-3.1-dpo_real_256",
        # "scicomma-3.1-dpo_real_512",
        # "scicomma-3.1-dpo_real_512_short",
        # "scicomma-3.1-dpo_full"
    ]
    asyncio.run(
        generate_metric_report(
            metrics={
                # ## BARAM TSABARI METRICS
                # "jargon": jargon_metric,
                # "metaphor_v2": metaphor_metric_explicit_v2,
                "humor_v2": humor_metric_explicit_v2,
                # "analogy_v2": analogy_metric_explicit_v2,
                # "explanation_type_v2": explanation_type_metric_explicit_v2,
                # "connection_to_everyday_life_v2": connection_to_everyday_life_metric_explicit_v2,
                #
                # # ## READING EASE
                # "scaffolding": scaffolding_metric,
                # "flesch_kincaid": flesch_kincaid,
                # "flesch_reading_ease": flesch_reading_ease,
                # "dale_chall": dale_chall,
                # "ari": ari,

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
            run_number=9,
        )
    )

    # check_reasons(
    #     model_map=MODEL_MAP,
    #     model='llama_2_sft',
    #     metric='completeness_explicit',
    #     run_number=0
    # )
