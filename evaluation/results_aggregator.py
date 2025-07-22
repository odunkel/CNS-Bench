import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
import pandas as pd
import numpy as np
import yaml


class ResultsAggregator:

    def __init__(self, args = dict()):
        self.args = args

    
    @staticmethod
    def load_evaluation_results(exp_dir_eval,exp_ids):
        
        dfs = []
        df_exps = []
        for exp_id in exp_ids:

            exp_sliding_dir_i = f"{exp_dir_eval}/../id_{exp_id:04d}/"
            if os.path.exists(exp_sliding_dir_i+'args.yaml'):
                with open(exp_sliding_dir_i+'args.yaml', 'r') as file:
                    args = yaml.safe_load(file)
                f = args['slider_weight'].split("/")[-1]
                shift = f[:f.find("class")-1]
            else:
                shift = 'shift'
            

            dfs_exp = []
            exps_dir_start = f"{exp_dir_eval}/id_{exp_id:04d}/"
            c_names = os.listdir(exps_dir_start)

            print('c_names',c_names)
            for c_name in c_names:
                
                exp_dir = f"{exps_dir_start}{c_name}/"
                dfs_i = ResultsAggregator.load_results(exp_dir)
                if len(dfs_i) == 0:
                    print(f"Continue {exp_id} {c_name} "); continue
                df_i = ResultsAggregator.combine_dfs(dfs_i)
                
                df_i['classifier'] = c_name
                df_i['exp_id'] = exp_id

                if f"{c_name}_i_class_hat" in df_i.columns: 
                    df_i[f"{c_name}"] = df_i[f"{c_name}_i_class_hat"] == df_i[f"i_class"]

                dfs.append(df_i)
                dfs_exp.append(df_i)
            df_exp = ResultsAggregator.merge_exp_dfs(dfs_exp)
            df_exp['shift'] = shift

            df_exps.append(df_exp)

        
        df_res = ResultsAggregator.combine_dfs(df_exps)


        df_res.rename(columns=lambda x: x.replace('-', '_'), inplace=True)
        
        return df_res
    
    @staticmethod
    def load_results(exp_dir: str):
        dfs = []
        files = os.listdir(exp_dir); files.sort()
        for file in files:
            if file.endswith(".csv"):
                df_eval = pd.read_csv(f"{exp_dir}/{file}")
                df_eval = df_eval.dropna(axis=1)
                N_na = len(df_eval[df_eval.isna().any(axis=1)])
                if (N_na > 0): print(f"Dropping {N_na} elements for {exp_dir} and {file}.")
                df_eval = df_eval.dropna()
                df_eval['i_class_dataset'] = int(file.split("_")[-1][:-4])
                dfs.append(df_eval)
        return dfs
    
    @staticmethod
    def combine_dfs(dfs: list):
        df_eval = pd.concat(dfs, ignore_index=True)
        if 'Unnamed: 0' in df_eval.columns:
            del df_eval['Unnamed: 0'] 
        return df_eval
    
    @staticmethod
    def merge_exp_dfs(dfs: list):
        df_res = dfs[0]
        for i in range(1,len(dfs)):
            cols_to_use = dfs[i].columns.difference(df_res.columns)
            df_res = pd.merge(df_res, dfs[i][cols_to_use], left_index=True, right_index=True)
        return df_res
    
    @staticmethod
    def filter_df(df_eval: pd.DataFrame, filter_quantity: str, filter_val: float):
        df_eval_filtered = df_eval[df_eval[filter_quantity]>filter_val]
        return df_eval_filtered
    
    @staticmethod
    def compute_delta_quantity_to_ref(df_res: pd.DataFrame, 
                    quantity_to_delta: str, new_col_name: str, ref_col_name: str,
                    groupby: List[str] = ['i_class', 'seed', 'classifier'], negate: bool = True):
        def compute_delta_clip_class(df):
            mask = (df[ref_col_name] == 0)
            delta_class_0 = df[mask][quantity_to_delta].values[0] if mask.sum() > 0 else None
            df[new_col_name] = df[quantity_to_delta] - delta_class_0
            if negate: df[new_col_name] *= -1
            return df
        df_with_delta = df_res.groupby(groupby,group_keys=False).apply(compute_delta_clip_class)
        return df_with_delta