import os

from tqdm import tqdm

from Jargon.jargon_metric import JargonMetric

OPENAI_API_KEY = "sk-proj-4reyI857Dx1FXwMAjCtCT3BlbkFJVrdDBRixPZCAHIfntrKN"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
import pandas as pd
from deepeval.test_case import LLMTestCase
# from g_eval.zemla_metrics import completeness_metric
from g_eval.explanation_quality import explanation_type_metric, correctness_metric, metaphor_metric, \
    content_units_metric, connection_to_everyday_life_metric, humor_metric, analogy_metric


#"~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv"
# def get_or_create_eval_dataset():
#     try:
#         df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv")
#         return df
#     except FileNotFoundError:
#         pass
#     llama_df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/data/llama7b_base_vs_sft.csv")
#     print("LLAMA BEFORE")
#     print(llama_df.dtypes)
#
#     llama_df['question'] = llama_df['question'].astype(str)
#     print("LLAMA AFTER")
#
#     print(llama_df.dtypes)
#
#     df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv")
#     print("DF BEFORE")
#
#     print(df.dtypes)
#     df['question'] = df['question'].astype(str)
#     print("DF AFTER")
#     print(df.dtypes)
#
#
#     eval_df = pd.merge(df, llama_df, on='question', how='inner')
#     eval_df.to_csv("~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv", index=False)
#     return eval_df
# ### eval df has the answer from llama 3.1, llama 2 7b, and the fine tuned llama 2 7b


def update_or_insert_score_column(eval_df, output_path, answer_column, model_name, metric_function, metric_name):
    scores = []
    reasons = []
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
        test_case = LLMTestCase(
            input=row['question'],
            actual_output=row[answer_column],
        )
        metric_function.measure(test_case)
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


def generate_metric_report(metrics, evaluation_dataset):
    model_map = {
        'llama_2_sft': 'sft_model_answer',
        'llama_2_base': 'base_model_answer',
        'llama_3_1': 'llama3_1_instruct_answer',
        'gpt_3.5_turbo': 'gpt_3_5_outputs',
        'gpt_4o': 'gpt_4o_outputs',
        'gpt_3_5_cot': 'gpt_3_5_cot',
    }
    eval_df = pd.read_csv(evaluation_dataset)
    # for i in range(3):
    for metric, metric_function in metrics.items():
        for model, answer_column in model_map.items():
            print("Evaluating", metric, "for", model)
            update_or_insert_score_column(
                eval_df,
                output_path=f"/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/data/{metric}_evaluation_scores_run_0.csv",
                answer_column=answer_column,
                model_name=model,
                metric_function=metric_function,
                metric_name=metric,
            )


if __name__ == "__main__":
    metrics = {
        # 'jargon': JargonMetric(),
        # 'metaphor': metaphor_metric,
        # 'explanation_type': explanation_type_metric,
        'content_units': content_units_metric,
        # 'connection_to_everyday_life': connection_to_everyday_life_metric,
        # 'humor': humor_metric,
        # 'analogy': analogy_metric
    }
    generate_metric_report(metrics, evaluation_dataset="~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv")
