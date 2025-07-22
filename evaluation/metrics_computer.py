import numpy as np
import pandas as pd


class EvalFilter:
    def __init__():
        pass

    @staticmethod
    def filter_k_out_of_n(df_e,quantile_ths=None):
        filter_quants = ['CLIP_class', "CLIP_ref", "CLIP_class_shift", "DINO_ref_no_head"]
        if not all([f in quantile_ths for f in filter_quants]):
            raise ValueError(f'Provide quantile thresholds for the following quantities: {filter_quants}')
        for filter_quant in quantile_ths.keys():
            df_e.loc[:,f'{filter_quant}_ic'] = df_e[filter_quant] > quantile_ths[filter_quant]
        m_filt_c = df_e.iloc[:,0] == df_e.iloc[:,0]
        m_filt_c = m_filt_c.astype(float)
        m_filt_c += df_e['CLIP_class_ic'] 
        m_filt_c += df_e['CLIP_class_shift_ic']
        m_filt_c +=  df_e['CLIP_ref_ic']
        m_filt_c +=  df_e['DINO_ref_no_head_ic']
        m_filt = m_filt_c > 2
        return m_filt

    @staticmethod
    def compute_failure_points(c_names_sel,df_e,max_shift=3):
        
        df_fp = []
        fp_collector = dict()
        for c_name in c_names_sel:
            df_false = df_e[df_e[c_name] == False]
            first_false_indices = df_false.groupby(['gt_class', 'seed']).apply(lambda x: x.first_valid_index())
            
            df_p = df_e.loc[first_false_indices]

            var1_values = df_p['scale']

            fp_max = df_p.groupby('scale').size().reset_index().max()['scale']
            
            fp_collector[c_name] = df_p.groupby('scale').size()

            fp_mean = var1_values.mean()
            fp_argmax = var1_values.argmax()
            fp_std = var1_values.std()
            N_failed = len(var1_values)

            correct_trajs = df_e.groupby(['gt_class', 'seed']).filter(lambda x: x[c_name].all())[['gt_class', 'seed']].drop_duplicates()
            correct_trajs_var1 = pd.Series(max_shift, index=correct_trajs.index, name='scale')
            var1_values = pd.concat([var1_values, correct_trajs_var1])

            l_fp = (max_shift - var1_values)
            margin_fp_mean = l_fp.mean()

            df_fp.append((c_name,fp_mean,fp_max,fp_std,margin_fp_mean,N_failed))

        df_fp = pd.DataFrame(df_fp,columns=['c_name','fp_avg','fp_max','fp_std','margin_fp_avg','N_failed'])
        fp_collector = pd.DataFrame(fp_collector)
        
        return df_fp, fp_collector