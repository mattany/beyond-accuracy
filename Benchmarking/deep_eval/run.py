import os
OPENAI_API_KEY = "sk-proj-4reyI857Dx1FXwMAjCtCT3BlbkFJVrdDBRixPZCAHIfntrKN"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
import pandas as pd
from deepeval.test_case import LLMTestCase
# from g_eval.zemla_metrics import completeness_metric
from g_eval.explanation_quality import explanation_type_metric, correctness_metric, metaphor_metric
#"~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv"
def get_or_create_eval_dataset():
    try:
        df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv")
        return df
    except FileNotFoundError:
        pass
    llama_df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/data/llama7b_base_vs_sft.csv")
    print("LLAMA BEFORE")
    print(llama_df.dtypes)

    llama_df['question'] = llama_df['question'].astype(str)
    print("LLAMA AFTER")

    print(llama_df.dtypes)

    df = pd.read_csv("~/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv")
    print("DF BEFORE")

    print(df.dtypes)
    df['question'] = df['question'].astype(str)
    print("DF AFTER")
    print(df.dtypes)


    eval_df = pd.merge(df, llama_df, on='question', how='inner')
    eval_df.to_csv("~/studies/thesis/Benchmarking/deep_eval/data/evaluation_dataset.csv", index=False)
    return eval_df
### eval df has the answer from llama 3.1, llama 2 7b, and the fine tuned llama 2 7b


def get_or_create_score_column(eval_df, answer_column, output_column_suffix, metric_function, metric_name):
    output_path = "~/studies/thesis/Benchmarking/deep_eval/data/eval_dataset_graded.csv"
    try:
        return pd.read_csv(output_path)
    except FileNotFoundError:
        pass
    scores = []
    reasons = []
    for index, row in eval_df.iterrows():
        test_case = LLMTestCase(
            input=row['question'],
            actual_output=row[answer_column]
        )
        metric_function.measure(test_case)
        print("result:", metric_function.score)
        print("reason:", metric_function.reason)

        scores.append(metric_function.score)
        reasons.append(metric_function.reason)
    eval_df[f"{metric_name}_{output_column_suffix}"] = scores
    eval_df.to_csv(output_path, index=False)
    return eval_df
    # print(row['question'])
    # print(row['answer'])
    # print("\n")

if __name__ == "__main__":
    eval_df = get_or_create_eval_dataset()
    metrics = {
        # 'correctness': correctness_metric,
        'explanation_type': explanation_type_metric,
        # 'metaphor': metaphor_metric
    }
    model_map = {
        'llama_2_base': 'base_model_answer',
        'llama_2_sft': 'sft_model_answer',
        'llama_3_1_base': 'answer'
    }
    for metric, metric_function in metrics.items():
        for model, answer_column in model_map.items():
            base_model_graded_df = get_or_create_score_column(
                eval_df,
                answer_column='base_model_answer',
                output_column_suffix='base_model',
                metric_function=)
            sft_model_graded_df = get_or_create_score_column(base_model_graded_df, answer_column='sft_model_answer', output_column_suffix='sft_model')
            llama_3_graded_df = get_or_create_score_column(base_model_graded_df, answer_column='answer', output_column_suffix='llama3_1')

    print(sft_model_graded_df.head())
