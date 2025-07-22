import os
import pandas as pd

dir_of_results = 'DIRECTORY OF RESULT GENERATED IN EASYROBUST PIPELINE'

models = os.listdir(dir_of_results)
dfs_m = []
for model in models:
    shifts = os.listdir(f'{dir_of_results}/{model}')
    for shift in shifts:
        files = os.listdir(f'{dir_of_results}/{model}/{shift}')
        
        # Load file and save it.
        csv_file = [f for f in files if f.endswith('.csv')]; 
        csv_file = csv_file[0]
        
        df = pd.read_csv(f'{dir_of_results}/{model}/{shift}/{csv_file}',index_col=0)
        
        df[['shift', 'class_str', 'img_name']] = df['img_dir'].str.split('/', expand=True).iloc[:,-3:]
        df['class_name'] = df['class_str'].str.split('_').str[-1]
        df['seed'] = df['img_name'].str.split('_').str[1]
        df['scale'] = df['img_name'].str.split('_').str[-1].str[:-5].astype(int)/1000
        df[model] = df['pred_class']==df['gt_class']
        del df['img_dir']
        dfs_m.append(df)
        df.to_csv(f'results/{model}__{shift}.csv')