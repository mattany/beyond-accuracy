import pandas as pd
from config import PROJECT_DIR
data_dir = f'{PROJECT_DIR}/Benchmarking/deep_eval/RAG/data/'
gpt_4_turbo_df = pd.read_csv(f'{data_dir}/gpt_4_turbo_answers.csv').rename(columns={'answer': 'gpt_4_turbo'})
gpt_4o_df = pd.read_csv(f'{data_dir}/gpt_4o_answers.csv').rename(columns={'answer': 'gpt_4o_1'})
gpt_4o_df2 = pd.read_csv(f'{data_dir}/gpt_4o_validation_run.csv').rename(columns={'answer': 'gpt_4o_2'})
llama2_df = pd.read_csv(f'{data_dir}/llama2_basemodel_answers.csv').rename(columns={'answer': 'llama_2'})
llama2_sft_df = pd.read_csv(f'{data_dir}/llama2_sft_answers.csv').rename(columns={'answer': 'llama_2_sft'})

joined_df = (gpt_4_turbo_df
             .merge(gpt_4o_df, on="Index")
             .merge(gpt_4o_df2, on="Index")
             .merge(llama2_df, on="Index")
             .merge(llama2_sft_df, on="Index")
             .drop(columns=['Index'])
             )

joined_df.to_csv(f'{data_dir}/joined_answers.csv')