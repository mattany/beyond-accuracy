import os
from tqdm import tqdm
from Jargon.jargon_metric import JargonMetric
from config import PROJECT_DIR

OPENAI_API_KEY = "sk-proj-4reyI857Dx1FXwMAjCtCT3BlbkFJVrdDBRixPZCAHIfntrKN"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import pandas as pd
from deepeval.test_case import LLMTestCase
from g_eval.zemla_metrics import internal_coherence_metric, completeness_metric, alternatives_metric, \
    articulation_metric, perceived_truth_metric
# from g_eval.zemla_metrics import completeness_metric
from g_eval.explanation_quality import explanation_type_metric, correctness_metric, metaphor_metric, \
    content_units_metric, connection_to_everyday_life_metric, humor_metric, analogy_metric, metaphor_metric_explicit


def update_or_insert_score_column(eval_df, output_path, answer_column, model_name, metric_function, metric_name, reference_column=None):
    scores = []
    reasons = []
    if reference_column:
        metric_name = f"{metric_name}_with_reference_{reference_column}"
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
        if reference_column:
            test_case = LLMTestCase(
                input=row['question'],
                actual_output=row[answer_column],
                expected_output=row[reference_column]
            )
        else:
            test_case = LLMTestCase(
                input=row['question'],
                actual_output=row[answer_column],
            )
        metric_function.measure(test_case)
        print(f"EVAL STEPS: {metric_function.evaluation_steps}")

        print("Question:", row['question'])
        print(f"{answer_column}:", row[answer_column])
        print("result:", metric_function.score)
        scores.append(metric_function.score)

        if getattr(metric_function, 'reason'):
            print("reason:", metric_function.reason)
            reasons.append(metric_function.reason)
    if os.path.exists(output_path):
        output_df = pd.read_csv(output_path)
    else:
        output_df = pd.DataFrame()
    output_df[f"{metric_name}_score__{model_name}"] = scores
    if reasons:
        output_df[f"{metric_name}_reason__{model_name}"] = reasons
    output_df.to_csv(output_path, index=False)
    return eval_df
    # print(row['question'])
    # print(row['answer'])
    # print("\n")


def generate_metric_report(metrics, evaluation_dataset, model_map, reference_models=None, models_to_evaluate=None, run_number=0):
    if not models_to_evaluate:
        models_to_evaluate = model_map.keys()
    eval_df = pd.read_csv(evaluation_dataset)
    for metric, metric_function in metrics.items():
        for model in models_to_evaluate:
            answer_column = model_map[model]
            if reference_models:
                for reference_model in reference_models:
                    if reference_model == model:
                        continue
                    print("Evaluating", metric, "for", model, "with reference model", reference_model)
                    update_or_insert_score_column(
                        eval_df,
                        output_path=f"/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric}_reference:{reference_model}_evaluation_scores_run_{run_number}.csv",
                        answer_column=answer_column,
                        model_name=model,
                        metric_function=metric_function,
                        metric_name=metric,
                        reference_column=model_map[reference_model]
                    )
            else:
                print("Evaluating", metric, "for", model)
                update_or_insert_score_column(
                    eval_df,
                    output_path=f"/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric}_evaluation_scores_run_{run_number}.csv",
                    answer_column=answer_column,
                    model_name=model,
                    metric_function=metric_function,
                    metric_name=metric
                )


if __name__ == "__main__":
    # RAG
    # generate_metric_report(
    #     metrics={
    #         'correctness': correctness_metric
    #     },
    #     evaluation_dataset=f'{PROJECT_DIR}/Benchmarking/deep_eval/RAG/data/joined_answers.csv',
    #     model_map={
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
    generate_metric_report(
        metrics={
            ## BARAM TSABARI METRICS
            # 'jargon': JargonMetric(),
            # 'explanation_type': explanation_type_metric,
            # 'metaphor': metaphor_metric,
            'metaphor_explicit': metaphor_metric_explicit,
            # 'content_units': content_units_metric,
            # 'connection_to_everyday_life': connection_to_everyday_life_metric,
            # 'humor': humor_metric,
            # 'analogy': analogy_metric
            ## ZEMLA METRICS
            # 'internal_coherence': internal_coherence_metric,
            # 'completeness': completeness_metric,
            # 'alternatives': alternatives_metric,
            # 'articulation': articulation_metric,
            # 'perceived_truth': perceived_truth_metric
            ## CORRECTNESS METRICS
            # 'correctness': correctness_metric
        },
        evaluation_dataset="~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv",
        model_map={
            'llama_2_sft': 'sft_model_answer',
            'llama_2_base': 'base_model_answer',
            'llama_3_1': 'llama3_1_instruct_answer',
            'gpt_3.5_turbo': 'gpt_3_5_outputs',
            'gpt_4o': 'gpt_4o_outputs',
            'gpt_3_5_cot': 'gpt_3_5_cot',
            'gpt_4': 'gpt_4_outputs'
        },
        run_number=0
    )
