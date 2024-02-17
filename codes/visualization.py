import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import os
from scipy.stats import wilcoxon

#----------------------------------------
def AggregateCsvFiles(time_stamp:str, array_keys:str, array_len:int):
    """
    A function aggregating the dataframes by different choice of settings, e.g. task, dimension, the number of data points, etc. The structure of this function is related to "bash.sh" file.
    
    Input:
    - time_stamp (str), array_keys(str): To specify the job runs in the bash file. They are used to locate the folder where the result dataframes are stored.
    - array_len (str): The number of jobs that parallelized by the bash file.
    
    Output:
    - data (pd.DataFrame): The aggregated (or concatenated) data. When "non_existing_file" is an empty list, it concatenate all result dataframe.
    - non_existing_file (list): The job number of the failed jobs run by the bash file.
    """
    non_existing_file = []
    
    # Read all csv files
    start = time.time()
    data_path  = "_".join(["output", time_stamp, array_keys])
    df_concat_list = []
    
    for i in range(1, array_len + 1):
        file_name = f"{data_path}/{i}_RepPer.csv"
        
        if os.path.exists(file_name): 
            retrieve = pd.read_csv(file_name)
            df_concat_list.append(retrieve)
        else:
            non_existing_file.append(i)
        
        if i%25 == 0: # To check the progress
            t = round(time.time()-start, 3)
            print(f"Read {i}/{array_len} files. Time = {t} sec")
    
    t = round(time.time()-start, 3)
    print(f"Total {t} sec")
    data = pd.concat(df_concat_list)
    print(f"non_existing_file: {non_existing_file}_RepPer.csv")
    
    return data, non_existing_file


#----------------------------------------
def CheckNaValue(data:pd.DataFrame):
    """
    Check whether the NA value exists in the dataframe.
    """
    na_num = data.isna().astype(int).sum(axis = 0) #
    
    if na_num.sum() != 0:
        print(f"There are NA values!\n{'-'*30}")
        print(na_num)
    else:
        print("No NA values.")



#----------------------------------------
def AggregateFolders(folder_list:list):
    """
    When the dataframes are stored in multiple folders, by calling "AggregateCsvFiles" multiple times, aggregate all dataframes. While aggregating, check the NA values in the dataframe per each folder.
    
    Input:
    - folder_list (list): Includes all the information about the folders. 
    """
    assert folder_list is not None
    
    df_list = []
    
    for i in range(len(folder_list)):
        print(f"<<<{i}>>>")
        current = folder_list[i]
        time_stamp, array_keys, array_len = current['TIME_STAMP'], current['ARRAY_KEYS'], current['ARRAY_LEN']

        curr_data, non_existing_file = AggregateCsvFiles(time_stamp, array_keys, array_len)
        
        CheckNaValue(curr_data) # check NA values
        
        df_list.append(curr_data)

    data = pd.concat(df_list)
    del df_list # To save memory
    
    return data

#----------------------------------------
RIVAL_LIST = ['oracle', 'kmmd_1', 'kmmd_2', 'kmmd_3', 'kmmd_g', 'energy']


#----------------------------------------
class Visualize():
    """
    This class contains all methods to visualize the plots in the main paper and the appendix.
    """
    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.data_num = self.data.shape[0]
        
        # Get the number of repetition of solving the optimization function. 
        # The paper and the appendix use 3 repetition.
        rep_optim = self.data['rep_optim'].unique()
        assert len(rep_optim) == 1
        self.rep_optim = rep_optim[0]
    
    #----------------------------------------
    def RefinedData(self, data:pd.DataFrame = None, setting:dict = None):
        """
        Given the data and some conditions, select the rows that meet those conditions only.
        
        Input:
        - data (pd.DataFrame): The original data.
        - setting (dict): Includes the conditions.
        
        Output:
        - txt (str): Information about the conditions.
        - mask (pd.Series): Boolean type. Indicates whether each row is selected or not.
        - output (pd.DataFrame): The dataframe with selected rows only.
        """
        
        assert setting is not None
            
        if data is None: data = self.data
        
        txt, mask = None, None
        for col in setting.keys():
            val = setting[col]
            new_txt = f"{col}: {val}"
            
            if mask is None: mask = (self.data[col] == val)
            else:            mask = mask & (self.data[col] == val)
                
            if txt is None: txt = new_txt
            else:           txt = " / ".join([txt, new_txt])
                
        output = data[mask].copy()
        
        return txt, mask, output
    
    #----------------------------------------
    def SubsampleDataByVarying(self, refined_data:pd.DataFrame, varying_cols:list = None):
        """
        Separate the given data by the values of certain columns. Here the separated data is called the 'subsampled data'.
        
        Input:
        - refined_data (pd.DataFrame): Mostly the given data is the output of "RefinedData" method).
        - varying_cols (list): The names of columns. The separation is based on these columns.
        
        Output:
        - all_varying_config (pd.DataFrame): Contains all possible configurations for all possible values of the columns in 'varying_cols'.
        - subsample (list): Contains the subsampled data and the according configuration for each subsampled data.
        """
        if varying_cols is None:
            return None, {'fixed setting': refined_data}
        
        ### Column names and values that separate 'refined_data'
        varying_dict = {col: self.data[col].unique().tolist() for col in varying_cols}
        
        ### Get configurations for all possible "varying" values
        all_varying_config = pd.DataFrame(columns = [key for key in varying_cols])
        comb_list = list(itertools.product(*[varying_dict[key] for key in varying_cols]))
        
        for idx, comb in enumerate(comb_list):
            one_config = {}
        
            for j, col in enumerate(varying_cols):
                val = comb[j]
                one_config[col] = val
            
            # concat to output
            one_config = pd.DataFrame(one_config, index = [idx+1])
            all_varying_config = pd.concat([all_varying_config, one_config])
            
        ### Get subsampled data based on value of each configuration
        subsample = {}
        for i in range(len(all_varying_config)): # for loop: for all configurations
            config = all_varying_config.iloc[i].copy()
            mask = None    
            sub_key = None
            for col in varying_cols:
                val = config[col]
                if mask is None:
                    mask = (refined_data[col] == val)
                else:
                    mask = mask & (refined_data[col] == val)
            
                if sub_key is None:
                    sub_key = f"{col}={val}"
                else:
                    sub_key = " / ".join([sub_key, f"{col}={val}"])

            # Save each subsampled data
            selected = refined_data[mask].copy()
            subsample[sub_key] = selected
            
        return all_varying_config, subsample
            
    #----------------------------------------
    def CalculateEcdf(self, alt_ipm:np.array, null_ipm:np.array, cuts = 1000):
        """
        Calculate the empirical cdf of the given MMD data.
        
        Input:
        - alt_ipm, null_ipm (np.array): The MMD data.
        - cuts (int): The number of the checkpoint to get the ecdf.
        
        Output:
        - alt_ecdf, null_ecdf (np.array): The ecdf of each given data.
        """
        ymin, ymax = np.min([alt_ipm, null_ipm]), np.max([alt_ipm, null_ipm])
        yval = np.linspace(ymin, ymax, cuts)
        
        alt_ecdf, null_ecdf = np.linspace(0, 1, cuts), np.linspace(0, 1, cuts)
        
        for (i, t) in enumerate(yval):
            alt_ecdf[i] = np.mean(alt_ipm <= t)
            null_ecdf[i] = np.mean(null_ipm <= t)
        
        return alt_ecdf, null_ecdf
    
    #----------------------------------------
    def PaperROC(self, setting_per_task:dict, no_large_k = False, cuts= 1000):
        """
        Plot the main figure in the main paper. The RKS tests and other non-parametric tests are plotted.
        
        Input:
        - setting_per_task (dict): Contains the information for each 'task'. (See the table in the main paper for more about the 'tasks'.)
        - no_large_k (bool): Limit the value of 'k' plotted at most 3 for the readability of the plot.
        - cuts (int): The number of knots to plot a single ROC curve.
        """
        ### Basic setup
        task_list = list(setting_per_task.keys())
        task_num  = len(task_list)
        d_list    = np.sort(self.data['d'].unique().tolist()) # 2, 4, 8, 16
        d_num     = len(d_list) 
        
        ### Initialize plot
        fig, axes = plt.subplots(task_num, d_num, figsize=(4 * d_num, 4 * task_num))
        xval = np.linspace(0, 1, cuts)
        
        ### Draw each subplot
        row_idx = -1 # rows represent different 'tasks'
        for task in task_list:
            task_setting = setting_per_task[task]
            task = task.split(":")[0]
            task_name_dict = {'kmmd-iso': "pancake-shift", "mean-shift": "ball-shift", "t-one":"t-coord",
                              "var-one": "var-one", "var-all" : "var-all"}
            task_name = task_name_dict[task]
            
            row_idx += 1
            col_idx = -1
            
            for d in d_list: # columns represent different 'd (dimensions)' 
                col_idx += 1
                print(row_idx, col_idx, task, d, end =" / ")
                
                # 'ax' represents each subplot
                ax = axes[row_idx, col_idx]
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.plot(xval, xval, color = 'k')
                ax.set_xlabel("False positive rate", fontsize=12)
                ax.set_ylabel("True positive rate", fontsize=12)
                ax.set_title(f"{task_name}, d={d}", fontsize = 16)
                
                # Get subsampled data for RKS test; get data for other tests
                varying_cols = ['k']
                subset_setting = { 'task': task,
                                   'd'   : d,
                                   'v'   : task_setting['v'],
                                   'nP'  : task_setting['nP'],
                                   'nQ'  : task_setting['nQ']
                }

                _, mask, refined_data = self.RefinedData(setting = subset_setting)
                
                all_varying_config, subsample_dict = self.SubsampleDataByVarying(refined_data, varying_cols)
                
                # while plotting RKS, data for rival tests is saved in this dictionary
                rival_dict = {rival: {'alt': {}, 'null':{}} for rival in RIVAL_LIST}
                
                ### --- Plot ROC curves of RKS with different k values: Start --- ###
                for sub_key in subsample_dict.keys():
                    if no_large_k:
                        if int(sub_key.split("=")[-1]) > 3:
                            continue
                            
                    # (0) Get data for sub_key
                    sub_data  = subsample_dict[sub_key]
                    sub_data  = sub_data[sub_data['best_or_last'] == 'best'] # choose 'best'
                    alt_data  = sub_data[sub_data['hypo'] == 'alt_hypo']
                    null_data = sub_data[sub_data['hypo'] == 'null_hypo']

                    # (1) Get alt/null values (the log function is used to reduce the range of values)
                    alt_ipm   = np.log(alt_data['ipm_max'].values)  # np.array
                    null_ipm  = np.log(null_data['ipm_max'].values) # np.array
                    
                    # (2) Get data for other tests:
                    #     Since seeds are fixed in certain ways during MMD calculation,
                    #     we can choose the (non-empty) data of any sub_key
                    for rival in RIVAL_LIST:
                        rival_dict[rival]['alt'][sub_key]  = alt_data[rival].values # np.array
                        rival_dict[rival]['null'][sub_key] = null_data[rival].values # np.array

                    # (3) Calculate ecdf
                    assert len(alt_ipm) == len(null_ipm)
                    if len(alt_ipm) == 0: continue
                    alt_ecdf, null_ecdf = self.CalculateEcdf(alt_ipm, null_ipm, cuts)
                    
                    # (4) Plot ROC curves
                    true_positive  = 1 - alt_ecdf
                    false_positive = 1 - null_ecdf
                    
                    color_RKS = {'k=0': 'purple', 'k=1': 'MediumBlue', 'k=2': 'green',
                                 'k=3': 'orangered' , 'k=4': 'purple', 'k=5': 'olive' }
                     
                    ax.plot(false_positive, true_positive, color = color_RKS[sub_key], label = f"RKS ({sub_key})")
                ### --- Plot ROC curves of RKS with different k values: End --- ###

                ### --- Plot ROC curves of other tests: Start --- ###
                for rival in rival_dict.keys():
                    # (0) Get data for sub_key
                    for sub_key in subsample_dict.keys():
                        # (1) Get alt/null values
                        alt_rival  = rival_dict[rival]['alt'][sub_key]
                        null_rival = rival_dict[rival]['null'][sub_key]
                        if rival != 'oracle':
                            alt_rival, null_rival = np.log(alt_rival), np.log(null_rival)
                        if len(alt_rival) > 0:
                            break

                    # (2) Calculate ecdf
                    alt_ecdf, null_ecdf = self.CalculateEcdf(alt_rival, null_rival, cuts)
                    
                    # (3) Plot ROC curves
                    true_positive  = 1 - alt_ecdf
                    false_positive = 1 - null_ecdf

                    color_rival = {'oracle': 'black',
                                   'kmmd_1': 'MediumBlue', 'kmmd_2': 'green', 'kmmd_3': 'orangered', 'kmmd_g': 'lightseagreen',
                                   'energy': 'grey'}
                    
                    label_rival = {'oracle': 'Oracle',
                                   'kmmd_1': 'KMMD (linear)', 'kmmd_2': 'KMMD (quadratic)', 'kmmd_3': 'KMMD (cubic)' , 'kmmd_g': 'KMMD (Gaussian)',
                                   'energy': 'Energy distance'}
                    
                    clr, lbl = color_rival[rival], label_rival[rival]
                    
                    if rival == 'oracle':
                        ax.plot(false_positive, true_positive, linestyle = 'dotted', lw = 1, c = clr, label = lbl)
                    elif rival.split('_')[0] == 'kmmd':
                        ax.plot(false_positive, true_positive, linestyle = 'dashed', lw =1, c = clr, label = lbl)
                    else:
                        ax.plot(false_positive, true_positive, linestyle = 'dashed', lw =1, c = clr, label = lbl)
                ### --- Plot ROC curves of other tests: End --- ###
                        
            ##### End of "for d in d_list" #####
        ##### End of "for task in task_list" #####

        handles, labels = axes[2,0].get_legend_handles_labels()
        ncol = 5 if no_large_k else 6
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02), ncol = ncol, fontsize = 16)
        fig.tight_layout()
        
        # Save
        fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)

    #----------------------------------------
    def AppendixROC(self, attribute:dict = None, no_large_k = False, cuts= 3000, optimizer_rep = True, all_or_one = 'all'):
        """
        Plot the most of ROC curves presented in Appendix. Unlike the "PaperROC" method, here only the RKS tests are plotted.
        
        Input:
        - attribute (dict): Contains all the informations about the plot generated by this method -- the row and column, fixed values, and the value varying in each subplot.
        - no_large_k (bool): Limit the value of 'k' plotted at most 3 for the readability of the plot.
        - cuts (int): The number of knots to plot a single ROC curve.
        - optimizer_rep (bool): Use the best MMD value among the results of the differently initalized optimizer solver, or use the first one instead of the best one.
        - all_or_one (bool): Use the MMD value calculated by all neurons, or use the MMD value calculated by the single best neuron among all neurons.
        """
        
        ### Basic setup
        assert attribute is not None
        row_rep, col_rep, varying_rep = attribute['row'], attribute['col'], attribute['varying']
        fixed_rep_keys   = [attribute['fixed'][i][0] for i in range(len(attribute['fixed']))]
        fixed_rep_values = [attribute['fixed'][i][1] for i in range(len(attribute['fixed']))]
        
        print(fixed_rep_keys, '\n', fixed_rep_values)
                
        row_list = np.sort(self.data[row_rep].unique().tolist()).tolist()
        col_list = np.sort(self.data[col_rep].unique().tolist()).tolist()
        
        if row_rep == 'k': row_list.remove(0)
        if col_rep == 'k': col_list.remove(0)
            
        # cf. To limit the value of rows and columns manually, a few lines of codes here, such as:
        #
        #     if row_rep == 'd': row_list = [2, 4, 8, 16]
        #     if col_rep == 'd': col_list = [2, 4, 8, 16]
            
        row_num, col_num = len(row_list), len(col_list)

        ### Initialize plot
        fig, axes = plt.subplots(row_num, col_num, figsize=(4 * col_num, 4 * row_num))
        xval = np.linspace(0, 1, cuts)
        
        ### Draw each subplot
        row_idx = -1
        for row_val in row_list:
            row_idx += 1
            col_idx = -1

            for col_val in col_list:
                col_idx += 1
                print(f"{row_idx}, {col_idx}, {row_rep}: {row_val}, {col_rep}: {col_val}", end =" / ")
                
                # 'ax' represents each subplot
                if row_num > 1: ax = axes[row_idx, col_idx]
                else:           ax = axes[col_idx]
                    
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.plot(xval, xval, color = 'k')
                ax.set_xlabel("False positive rate", fontsize=12)
                ax.set_ylabel("True positive rate", fontsize=12)
                
                if row_rep == 'log_nolog':
                    row_val_ = 'no-log' if row_val == 'nolog' else row_val
                    ax.set_title(f"{row_val_}, {col_rep}={col_val}", fontsize = 16)
                else:
                    ax.set_title(f"{row_rep}={row_val}, {col_rep}={col_val}", fontsize = 16)
                
                # Get subsampled data for RKS test
                varying_list = [varying_rep]
                subset_setting = {row_rep     : row_val,
                                  col_rep     : col_val}

                for i in range(len(fixed_rep_keys)):
                    subset_setting[fixed_rep_keys[i]] = fixed_rep_values[i]
                    
                _, _, refined_data = self.RefinedData(setting = subset_setting)
                
                all_varying_config, subsample_dict = self.SubsampleDataByVarying(refined_data, varying_list)
                
                ### --- Plot ROC curves of RKS with different k values: Start --- ###
                if varying_rep in ['lr', 'eff_tI', 'N']:
                    varying_vals = list(self.data[varying_rep].unique())
                    varying_vals.sort()
                    sub_key_list = [f"{varying_rep}={va}" for va in varying_vals]
                else:
                    sub_key_list = list(subsample_dict.keys())
                    sub_key_list.sort()

                for sub_key in sub_key_list:
                    if no_large_k:
                        if int(sub_key.split("=")[-1]) > 3:
                            continue
                            
                    # (0) Get data for sub_key
                    sub_data  = subsample_dict[sub_key]
                                        
                    # (1) Get alt/null values:
                    #     The log function is used to reduce the range of values.
                    #     1e-32 is used as a correction of 0.
                    alt_data  = sub_data[sub_data['hypo'] == 'alt_hypo']
                    null_data = sub_data[sub_data['hypo'] == 'null_hypo']

                    if all_or_one == 'all':
                        if optimizer_rep:
                            alt_ipm   = np.log(alt_data['ipm_max'].values + 1e-32)
                            null_ipm  = np.log(null_data['ipm_max'].values + 1e-32)
                        else:
                            alt_ipm   = np.log(alt_data['ipm_optim:1'].values +1e-32)
                            null_ipm  = np.log(null_data['ipm_optim:1'].values +1e-32) 
                    else:
                        if optimizer_rep:
                            alt_ipm   = np.log(alt_data['ipm_max_one'].values + 1e-32) 
                            null_ipm  = np.log(null_data['ipm_max_one'].values + 1e-32)
                        else:
                            alt_ipm   = np.log(alt_data['ipm_one:1'].values +1e-32) 
                            null_ipm  = np.log(null_data['ipm_one:1'].values +1e-32)

                    # (2) Calculate ecdf
                    assert len(alt_ipm) == len(null_ipm)
                    if len(alt_ipm) == 0: continue

                    alt_ecdf, null_ecdf = self.CalculateEcdf(alt_ipm, null_ipm, cuts)
                    
                    # (3) Plot ROC curves
                    true_positive  = 1 - alt_ecdf
                    false_positive = 1 - null_ecdf
                    
                    if varying_rep == 'k':
                        color_RKS = {'k=0': 'purple', 'k=1': 'MediumBlue', 'k=2': 'green',
                                     'k=3': 'orangered' , 'k=4': 'brown', 'k=5': 'grey' }
                        
                        ax.plot(false_positive, true_positive, color = color_RKS[sub_key], label = f"{sub_key}")
                        
                    elif varying_rep == 'N':
                        color_RKS = {'N=1': 'purple', 'N=2': 'MediumBlue', 'N=5': 'green',
                                     'N=10': 'red' , 'N=20': 'grey'}

                        ax.plot(false_positive, true_positive, color = color_RKS[sub_key], label = f"{sub_key}")
                    
                    elif varying_rep == 'log_nolog':
                        color_RKS = {'log_nolog=log': 'red', 'log_nolog=nolog': 'MediumBlue'}
                        line_RKS = {'log_nolog=log': 'solid', 'log_nolog=nolog': 'solid'}
                        sub_key_ = sub_key.split("=")[-1]
                        sub_key_ = 'no-log' if sub_key_ == 'nolog' else sub_key_

                        ax.plot(false_positive, true_positive, color = color_RKS[sub_key], linestyle = line_RKS[sub_key], label = f"{sub_key_}")
                    
                    elif varying_rep == 'lr':
                        color_RKS = {'lr=0.01': 'cyan', 'lr=0.05':'black',  'lr=0.1': 'green', 
                                     'lr=0.2': 'MediumBlue', 'lr=0.5': 'red', 'lr=1.0': 'grey' ,
                                     'lr=2.0': 'purple', 'lr=5.0': 'darkorange', 'lr=10.0': 'brown'}

                        ax.plot(false_positive, true_positive, color = color_RKS[sub_key], label = f"{sub_key}")
                        
                    else:
                        if 'eff_tI' not in sub_key:
                            ax.plot(false_positive, true_positive, label = f"{sub_key}")
                        else:
                            tI = int(sub_key.split("=")[-1])*100
                            sub_key = f"iter={tI}"
                            ax.plot(false_positive, true_positive, label = f"{sub_key}")
                ### --- Plot ROC curves of RKS with different k values: End --- ###
                        
            ##### End of "for d in d_list" #####
        ##### End of "for task in task_list" #####

        ### Ajustment for plot
        # (1) Get labels 
        if row_num > 1: handles, labels = axes[1,0].get_legend_handles_labels()
        else:           handles, labels = axes[0].get_legend_handles_labels()
        
        # (2) Plot location
        if row_num == 1:            top_adjust, label_adjust = 0.87, -0.04
        elif row_num == 2:          top_adjust, label_adjust = 0.92, -0.03
        elif row_num in range(3,9): top_adjust, label_adjust = 0.96, -0.02
        else:                       top_adjust, label_adjust = 0.98, -0.01
            
        # (3) Legend
        ncol = 5
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, label_adjust), ncol = ncol, fontsize = 16)
        fig.tight_layout()
        
        # (4) Title
        title = []
        for i in range(len(fixed_rep_keys)):
            title.append(f"<{fixed_rep_keys[i]}>={fixed_rep_values[i]}")

        fig.subplots_adjust(top = top_adjust)
        
        # (5) Save
        fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)
        
    #----------------------------------------
    def AppendixROC_optimrep(self, attribute:dict = None, no_large_k = False, cuts= 1000, all_or_one = 'all'):
        """
        Plot the ROC curves comparing the effect of repeating the optimization process with different initialized parameters.
        This is very similar to "AppendixROC" method.
        
        Input:
        - attribute (dict): Contains all the informations about the plot generated by this method -- the row and column, fixed values, and the value varying in each subplot.
        - no_large_k (bool): Limit the value of 'k' plotted at most 3 for the readability of the plot.
        - cuts (int): The number of knots to plot a single ROC curve.
        - all_or_one (bool): Use the MMD value calculated by all neurons, or use the MMD value calculated by the single best neuron among all neurons.
        """
        
        ### Basic setup
        assert attribute is not None
        row_rep, col_rep, varying_rep = attribute['row'], attribute['col'], attribute['varying']
        fixed_rep_keys   = [attribute['fixed'][i][0] for i in range(len(attribute['fixed']))]
        fixed_rep_values = [attribute['fixed'][i][1] for i in range(len(attribute['fixed']))]
        
        print(fixed_rep_keys, '\n', fixed_rep_values)
        
        row_list = np.sort(self.data[row_rep].unique().tolist()).tolist()
        col_list = np.sort(self.data[col_rep].unique().tolist()).tolist()

        # cf. To limit the value of rows and columns manually, a few lines of codes here, such as:
        #
        #     if row_rep == 'd': row_list = [2, 4, 8, 16]
        #     if col_rep == 'd': col_list = [2, 4, 8, 16]
        
        row_num, col_num = len(row_list), len(col_list)
        
        ### Initialize plot
        fig, axes = plt.subplots(row_num, col_num, figsize=(4 * col_num, 4 * row_num))
        xval = np.linspace(0, 1, cuts)
        
        ### Draw each subplot
        row_idx = -1
        for row_val in row_list:
            row_idx += 1
            col_idx = -1
            
            for col_val in col_list:
                col_idx += 1
                print(f"{row_idx}, {col_idx}, {row_rep}: {row_val}, {col_rep}: {col_val}", end =" / ")
                
                # 'ax' represents each subplot
                if row_num > 1: ax = axes[row_idx, col_idx]
                else:           ax = axes[col_idx]
                    
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.plot(xval, xval, color = 'k')
                ax.set_xlabel("False positive rate", fontsize=12)
                ax.set_ylabel("True positive rate", fontsize=12)
                ax.set_title(f"{row_rep}={row_val}, {col_rep}={col_val}", fontsize = 16)
                
                # Get subsampled data for RKS test
                varying_list = [varying_rep]
                subset_setting = {row_rep     : row_val,
                                  col_rep     : col_val}
                                  # 'tI'        : 3}
                for i in range(len(fixed_rep_keys)):
                    subset_setting[fixed_rep_keys[i]] = fixed_rep_values[i]

                _, _, refined_data = self.RefinedData(setting = subset_setting)
                all_varying_config, subsample_dict = self.SubsampleDataByVarying(refined_data, varying_list)

                ### --- Plot ROC curves of RKS with different k values: Start --- ###
                for sub_key in subsample_dict.keys():
                    if no_large_k:
                        if int(sub_key.split("=")[-1]) > 3: continue
                            
                    # Skip k=0 since it does not solve optimization problem multiple times
                    if sub_key == 'k=0': continue
                            
                    # (0) Get data for sub_key
                    sub_data  = subsample_dict[sub_key]

                    alt_data  = sub_data[sub_data['hypo'] == 'alt_hypo']
                    null_data = sub_data[sub_data['hypo'] == 'null_hypo']

                    # (1) Get alt/null values:
                    #     The log function is used to reduce the range of values.
                    if all_or_one == 'all':
                        alt_ipm_rep     = np.log(alt_data['ipm_max'].values) 
                        null_ipm_rep    = np.log(null_data['ipm_max'].values)
                        alt_ipm_norep   = np.log(alt_data['ipm_optim:1'].values)
                        null_ipm_norep  = np.log(null_data['ipm_optim:1'].values)
                    else:
                        alt_ipm_rep     = np.log(alt_data['ipm_max_one'].values) 
                        null_ipm_rep    = np.log(null_data['ipm_max_one'].values)
                        alt_ipm_norep   = np.log(alt_data['ipm_one:1'].values) 
                        null_ipm_norep  = np.log(null_data['ipm_one:1'].values)

                    # (2) Calculate ecdf
                    assert len(alt_ipm_rep) == len(null_ipm_rep)
                    assert len(alt_ipm_norep) == len(null_ipm_norep)
                    if len(alt_ipm_rep) == 0: continue
                    if len(alt_ipm_norep) == 0: continue
                        
                    alt_ecdf_rep,   null_ecdf_rep   = self.CalculateEcdf(alt_ipm_rep, null_ipm_rep, cuts)
                    alt_ecdf_norep, null_ecdf_norep = self.CalculateEcdf(alt_ipm_norep, null_ipm_norep, cuts)
                    
                    # (3) Plot ROC curves
                    true_positive_rep  = 1 - alt_ecdf_rep
                    false_positive_rep = 1 - null_ecdf_rep
                    
                    true_positive_norep  = 1 - alt_ecdf_norep
                    false_positive_norep = 1 - null_ecdf_norep
                    
                    if varying_rep == 'k':
                        color_RKS = {'k=0': 'purple', 'k=1': 'MediumBlue', 'k=2': 'green',
                                     'k=3': 'orangered' , 'k=4': 'brown', 'k=5': 'grey' }
                        
                        if sub_key in ['k=4', 'k=5']:
                            continue

                        ax.plot(false_positive_rep, true_positive_rep, color = color_RKS[sub_key], label = f"{sub_key}, rep.")
                        
                        ax.plot(false_positive_norep, true_positive_norep, color = color_RKS[sub_key], linestyle = 'dashed', label = f"{sub_key}, no rep.")
                ### --- Plot ROC curves of RKS with different k values: End --- ###
                        
            ##### End of "for d in d_list" #####
        ##### End of "for task in task_list" #####

        ### Ajustment for plot
        # (1) Get labels 
        if row_num > 1: handles, labels = axes[1,0].get_legend_handles_labels()
        else:           handles, labels = axes[0].get_legend_handles_labels()
            
        # (2) Plot location
        if row_num == 1:            top_adjust, label_adjust = 0.87, -0.04
        elif row_num == 2:          top_adjust, label_adjust = 0.92, -0.03
        elif row_num in range(3,9): top_adjust, label_adjust = 0.96, -0.02
        else:                       top_adjust, label_adjust = 0.98, -0.01
          
        # (3) Legend
        ncol = 3
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, label_adjust), ncol = ncol, fontsize = 16)
        fig.tight_layout()
        
        # (4) Title
        title = []
        for i in range(len(fixed_rep_keys)):
            title.append(f"<{fixed_rep_keys[i]}>={fixed_rep_values[i]}")

        fig.suptitle(", ".join(title), fontsize = 18, y=1)
        fig.subplots_adjust(top=top_adjust) 
        
        # (5) Save
        fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)
    
    #----------------------------------------
    def AppendixMMDCurve(self, attribute:dict = None, optimizer_rep = True, sample_rep = True, mmdtype = None, speed_only = True, both_all_or_one = False, same_y_axes = False):
        """
        Plot the convergence curve (or the trajectory of the MMD value calculated) during the whole optimization process.
        
        Input:
        - attribute (dict): Contains all the informations about the plot generated by this method -- the row and column, fixed values, and the value varying in each subplot.
        - optimizer_rep (bool): Use the best MMD value among the results of the differently initalized optimizer solver, or use the first one instead of the best one.
        - sample_rep (bool): When it is True, plot the median value across multiple samples and the colored range between 25% and 75% quantile. When it is False, plot the convergence curve of one sample.
        - speed_only (bool): When it is True, the data for each sample is normalized by the maximum value of that data. When it is False, the raw data is used.
        - both_all_or_one (bool): When it is True, plot the MMD trajectory calculated by both all neurons and the best neuron among them. When it is False, plot the MMD trajectory calculated by the best neuron only; and 'all_or_one' value must he specified in 'attribute' parameter.
        - same_y_axes (bool): Whether all subplots share the same y-axes or not.
        """

        ### Basic setup
        assert attribute is not None
        
        if mmdtype is None: mmdtype = "best:1200"
            
        smpl_rep_total = max(self.data['rep_no:smpl'].unique())
        
        row_rep, col_rep, varying_rep = attribute['row'], attribute['col'], attribute['varying']
        fixed_rep_keys   = [attribute['fixed'][i][0] for i in range(len(attribute['fixed']))]
        fixed_rep_values = [attribute['fixed'][i][1] for i in range(len(attribute['fixed']))]
        
        if both_all_or_one == True: assert "all_or_one" not in fixed_rep_keys
        else:                       assert "all_or_one" in fixed_rep_keys
        
        print(fixed_rep_keys, '\n', fixed_rep_values)
        
        row_list = np.sort(self.data[row_rep].unique().tolist()).tolist()
        col_list = np.sort(self.data[col_rep].unique().tolist()).tolist()
        
        ### Several manual adjustment for rows and columns representation: for readability of plot
        if row_rep == 'log_nolog': row_list = ['log']
            
        if row_rep == 'lr': row_list = [0.01, 0.1, 0.5, 1, 5]
            
        if varying_rep == 'N':
            if row_rep == 'd': row_list = [16]
            if col_rep == 'lr': col_list = [0.01, 0.5, 1, 5]
            
        row_num, col_num = len(row_list), len(col_list)
        
        ### Initialize plot
        fig, axes = plt.subplots(row_num, col_num, figsize=(4 * col_num, 4 * row_num))
        ymin, ymax = float("inf"), float("-inf")
        
        row_idx = -1
        for row_val in row_list:
            row_idx += 1
            col_idx = -1
            
            for col_val in col_list:
                col_idx += 1
                
                # 'ax' represents subplot
                if row_num > 1: ax = axes[row_idx, col_idx]
                else:           ax = axes[col_idx]
                
                ax.set_xlabel("Steps", fontsize=12)
                ax.set_ylabel("MMD values", fontsize=12)
                
                if row_rep == 'log_nolog':
                    row_val_ = 'no-log' if row_val == 'nolog' else row_val
                    ax.set_title(f"{row_val_}, {col_rep}={col_val}", fontsize = 16)
                else:
                    ax.set_title(f"{row_rep}={row_val}, {col_rep}={col_val}", fontsize = 16)
                
                ### Set varying values
                varying_list = [varying_rep]
                subset_setting = {row_rep     : row_val,
                                  col_rep     : col_val}
                
                for i in range(len(fixed_rep_keys)):
                    subset_setting[fixed_rep_keys[i]] = fixed_rep_values[i]

                _, _, refined_data = self.RefinedData(setting = subset_setting)
                all_varying_config, subsample_dict = self.SubsampleDataByVarying(refined_data, varying_list)
                
                ### --- Plot convergence curves: Start --- ###
                mmdcurve_cols = [f"mmdcurve:{i}" for i in range(1200)]
                mmdbest_record, mmdcurve_record = {}, {}
                ax_max = float("-inf")
                
                for sub_key in subsample_dict.keys():
                    # --- Some manual adjustment can happen here too, such as:
                    if sub_key in ['N=1', 'N=5', 'N=20']: continue
                                
                    ## (0) Get data for sub_key
                    mmdcurve_record[sub_key] = {'alt':None, 'null':None}
                    mmdbest_record[sub_key]  = {'alt':None, 'null':None}
                                                
                    sub_data  = subsample_dict[sub_key]
                                    
                    # (1) Get alt/null values, considering 'optmizer_rep'
                    alt_data  = sub_data[sub_data['hypo'] == 'alt_hypo']
                    null_data = sub_data[sub_data['hypo'] == 'null_hypo']
                                                
                    if optimizer_rep == True:
                        # Get 'mmdbest'
                        if ':' in mmdtype:      mmdbest    = "mmd" + mmdtype
                        elif mmdtype == "last": mmdbest = 'mmdfinal'
                        
                        # Check both_all_or_one
                        if both_all_or_one == False:
                            alt_data = alt_data.loc[alt_data.groupby('rep_no:smpl')[mmdbest].idxmax()]
                            null_data = null_data.loc[null_data.groupby('rep_no:smpl')[mmdbest].idxmax()]
                        else:
                            alt_data = alt_data.loc[alt_data.groupby(['rep_no:smpl', 'all_or_one'])[mmdbest].idxmax()]
                            null_data = null_data.loc[null_data.groupby(['rep_no:smpl', 'all_or_one'])[mmdbest].idxmax()]
                    
                    else:
                        alt_data = alt_data[alt_data['rep_no:optim'] == 1]
                        null_data = null_data[null_data['rep_no:optim'] == 1]
                        
                    mmdbest_record[sub_key]['alt']  = alt_data[mmdbest].values
                    mmdbest_record[sub_key]['null'] = null_data[mmdbest].values
                        
                    # (2-1) When 'sample_rep' is True
                    if sample_rep == True:
                        mmdcurve_record[sub_key]['alt']  = alt_data
                        mmdcurve_record[sub_key]['null'] = null_data
                        
                        # Check 'speed_only'
                        if speed_only == True:
                            alt_data.loc[:, mmdcurve_cols]  = alt_data.loc[:, mmdcurve_cols] / alt_data.loc[:, 'mmdbest:1200'].values[:, None]
                            null_data.loc[:, mmdcurve_cols] = null_data.loc[:, mmdcurve_cols] / null_data.loc[:, 'mmdbest:1200'].values[:, None]
                        
                        else:
                            alt_data.loc[:, mmdcurve_cols]  = alt_data.loc[:, mmdcurve_cols] 
                            null_data.loc[:, mmdcurve_cols] = null_data.loc[:, mmdcurve_cols]

                        alt_quantile  = alt_data[mmdcurve_cols].quantile([0.25, 0.5, 0.75])
                        null_quantile = null_data[mmdcurve_cols].quantile([0.25, 0.5, 0.75])
                     
                    # (2-2) When 'sample_rep' is False
                    else:
                        alt_data  = alt_data[alt_data['rep_no:smpl'] == 1]
                        null_data = null_data[null_data['rep_no:smpl'] == 1]
                        
                        alt_grid  = alt_data['gridsearch'].values[0]
                        null_grid = null_data['gridsearch'].values[0]

                        # Check 'both_all_or_one'
                        if both_all_or_one == False:
                            alt_data  = alt_data[mmdcurve_cols].values.reshape(-1)
                            null_data = null_data[mmdcurve_cols].values.reshape(-1)
                            
                            temp = np.concatenate((alt_data, null_data))
                            curr_ymin, curr_ymax = np.amin(temp), np.amax(temp)
                            
                        else:
                            alt_data_all  = alt_data[alt_data['all_or_one'] == 'all'][mmdcurve_cols].values.reshape(-1)
                            alt_data_one  = alt_data[alt_data['all_or_one'] == 'one'][mmdcurve_cols].values.reshape(-1)         
                            null_data_all  = null_data[null_data['all_or_one'] == 'all'][mmdcurve_cols].values.reshape(-1)
                            null_data_one  = null_data[null_data['all_or_one'] == 'one'][mmdcurve_cols].values.reshape(-1)

                            temp = np.concatenate((alt_data_all, null_data_all, alt_data_one, null_data_one))
                            curr_ymin, curr_ymax = np.amin(temp), np.amax(temp)
                    
                        # update ymin and ymax
                        if curr_ymin < ymin: ymin = curr_ymin
                        if curr_ymax > ymax: ymax = curr_ymax
                            
                    
                    # (3) Plot data
                    if varying_rep == 'log_nolog':
                            color_RKS = {'log_nolog=log/alt' : 'red', 'log_nolog=nolog/alt' : 'mediumblue',
                                         'log_nolog=log/null': 'darkorange', 'log_nolog=nolog/null': 'green'}
                    if varying_rep == 'N':
                            color_RKS = {'N=10/alt' : 'red', 'N=2/alt' : 'mediumblue',
                                         'N=10/null': 'darkorange', 'N=2/null': 'green'}
                        
                    log_nolog = sub_key.split("=")[-1]
                    log_nolog = 'no-log' if log_nolog == 'nolog' else log_nolog
                    if varying_rep == 'N':
                        log_nolog = f"N={log_nolog}"
                        
                    alpha = 0.8
                    if sample_rep == False:
                        if both_all_or_one == False:
                            ax.plot(range(1200), alt_data, color = color_RKS["/".join([sub_key, 'alt'])], alpha = alpha, label = f"{log_nolog}, alt", linewidth = 2)
                            ax.plot(range(1200), null_data, color = color_RKS["/".join([sub_key, 'null'])], alpha = alpha,  label = f"{log_nolog}, null", linewidth = 2)    
                            
                        else:
                            ax.plot(range(1200), alt_data_all, color = color_RKS["/".join([sub_key, 'alt'])], alpha = alpha, label = f"{log_nolog}, alt, all", linewidth = 2)
                            ax.plot(range(1200), null_data_all, color = color_RKS["/".join([sub_key, 'null'])], alpha = alpha,  label = f"{log_nolog}, null, all", linewidth = 2)    
                            ax.plot(range(1200), alt_data_one, color = color_RKS["/".join([sub_key, 'alt'])], alpha = alpha, label = f"{log_nolog}, alt, one", linestyle = 'dashed', linewidth = 2)
                            ax.plot(range(1200), null_data_one, color = color_RKS["/".join([sub_key, 'null'])], alpha = alpha,  label = f"{log_nolog}, null, one", linestyle = 'dashed', linewidth = 2)

                        if subset_setting['d'] == 2:
                            ax.axhline(y=alt_grid, color = 'k') #, label = f"{sub_key}, alt. grid")
                            ax.axhline(y=null_grid, color = 'k') #, label = f"{sub_key}, null. grid")
                            
                    elif sample_rep == True:
#                         ax.set_ylim(0, 1.01)
                        for altnull in ['alt', 'null']:
                            df = alt_quantile if altnull == 'alt' else null_quantile
                            q25, q50, q75 = df.loc[0.25], df.loc[0.5], df.loc[0.75]
                            
                            ax.plot(range(1200), q50, color = color_RKS["/".join([sub_key, altnull])], alpha = 0.8, label = f"{log_nolog}, {altnull}", linewidth = 3)
                            ax.fill_between(range(1200), q25, q75, color = color_RKS["/".join([sub_key, altnull])], alpha=0.3)
                ### --- Plot convergence curves: Start --- ###
            ### --- End of "for col_val in col_list" --- ###        
        
        ### Ajustment for plot
        # (0) Check same_y_axes
        for row_idx in range(row_num):
            for col_idx in range(col_num):
                if row_num > 1: ax = axes[row_idx, col_idx]
                else:           ax = axes[col_idx]

                if same_y_axes:
                    ax.set_ylim(0.98 * ymin, 1.02 * ymax)
        
        # (1) Get labels 
        if row_num > 1: handles, labels = axes[1,0].get_legend_handles_labels()
        else:           handles, labels = axes[0].get_legend_handles_labels()
            
        # (2) Legend
        ncol = 4
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02), ncol = ncol, fontsize = 16)
        fig.tight_layout()
        
        # (3) Title
        title = [mmdbest]
        for i in range(len(fixed_rep_keys)):
            title.append(f"<{fixed_rep_keys[i]}>={fixed_rep_values[i]}")

        fig.suptitle(", ".join(title), fontsize = 18, y=1)
        
        if row_num ==1: fig.subplots_adjust(top=0.86) 
        else:           fig.subplots_adjust(top=0.97) 
            
        # (4) Save
        fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)
              
    #----------------------------------------
    def AppendixMMD_compare(self, vistype = None, datatype = None, attribute:dict = None, optimizer_rep = True, sample_rep = True, mmdtype = None):
        """
        Compare the optimization results of two optimization settings, by drawing the scatter plot or the table.
        
        Input:
        - vistype (str): The type of the result -- scatter plot, or table.
        - datatype (str): The type of data used. This indicates whether data includes all history of the optimization (which is used in drawing the convergence curve), or only includes the result of certain checkpoints throughout the optimization.
        - attribute (dict): Contains all the informations about the plot generated by this method -- the row and column, fixed values, and the value varying in each subplot.
        - optimizer_rep (bool): Use the best MMD value among the results of the differently initalized optimizer solver, or use the first one instead of the best one.
        - sample_rep (bool): When it is True, plot the median value across multiple samples and the colored range between 25% and 75% quantile. When it is False, plot the convergence curve of one sample.
        - mmdtype (str): Specify the checkpoint to compare two optimization settings.
        """
        
        # Basic setting
        assert vistype in ['table', 'scatter']
        assert datatype in ['all_steps', 'only_checkpoints']
        assert attribute is not None
        assert mmdtype is not None
        
        if datatype == 'all_steps': smpl_rep_total = max(self.data['rep_no:smpl'].unique()) 
        else:                       smpl_rep_total = max(self.data['rep_sample'].unique()) 
                    
        lr_list, col_rep, c_list = attribute['lr_list'], attribute['col'], attribute['c_list']
        
        fixed_rep_keys   = [attribute['fixed'][i][0] for i in range(len(attribute['fixed']))]
        fixed_rep_values = [attribute['fixed'][i][1] for i in range(len(attribute['fixed']))]
        
        if datatype == 'only_checkpoints':
            idx = fixed_rep_keys.index('all_or_one')
            fixed_rep_keys.pop(idx)
            all_or_one = fixed_rep_values.pop(idx)
            
        print(fixed_rep_keys, fixed_rep_values)
        
        ### Initialize plot
        title = [mmdtype]
        for i in range(len(fixed_rep_keys)):
            title.append(f"<{fixed_rep_keys[i]}>={fixed_rep_values[i]}")
        if len(lr_list) == 1:
            title.append(f"<lr>={lr_list[0]}")
        
        ### Sanity check for certain choices of two optimization settting
        compare_rep, compare_vals = attribute['compare'][0], attribute['compare'][1]
        
        assert compare_rep in self.data.columns or compare_rep == 'opt_rep'
        
        if compare_rep == 'log_nolog':
            assert 'N' in fixed_rep_keys
            assert optimizer_rep is not None
        elif compare_rep == 'N':
            assert 'log_nolog' in fixed_rep_keys
            assert optimizer_rep is not None
        elif compare_rep == 'opt_rep':
            assert 'N' in fixed_rep_keys
            assert 'log_nolog' in fixed_rep_keys
            assert optimizer_rep is None
        elif compare_rep =='lr':
            assert 'N' in fixed_rep_keys
            assert 'log_nolog' in fixed_rep_keys
            assert len(lr_list) == 0
            
        assert len(compare_vals) == 2
        comp1, comp2  =  compare_vals[0], compare_vals[1]
        
        if vistype == 'table': row_list = [[comp1, 'alt'], [comp2, 'alt'], [comp1, 'null'], [comp2, 'null']]    
        elif vistype in ['scatter', 'wilcoxon']: row_list = [[comp1, 'alt'], [comp1, 'null']]    
        
        col_list = np.sort(self.data[col_rep].unique().tolist()).tolist()   
        row_num, col_num = len(row_list), len(col_list)
        
        if len(lr_list) == 2:
            fig, axes = plt.subplots(row_num, col_num, figsize=(len(c_list) * col_num, 1.5 * row_num))   
        elif len(lr_list) == 5:
            fig, axes = plt.subplots(row_num, col_num, figsize=(len(c_list) * col_num, 2.4 * row_num))
        else:
            fig, axes = plt.subplots(row_num, col_num, figsize=(len(c_list) * col_num, 4 * row_num))

        if vistype == 'scatter':
            ax_max = float("-inf")
            count_dict = {}

        ### Draw each subplot
        row_idx = -1
        for row_val in row_list:
            print()
            row_idx += 1
            col_idx = -1
            
            comp     = row_val[0]
            comp_alt = comp1 if comp == comp2 else comp2
            
            for col_val in col_list:
                col_idx += 1
                
                # 'ax' represents subplot
                ax = axes[row_idx, col_idx]
                if vistype == 'table': ax.axis('off')

                if compare_rep == 'opt_rep': ax_title = f"({row_idx+1}, {col_idx+1}) {comp} rep ≥ c x {comp_alt} rep: {row_val[1]}, {col_rep}={col_val}"
                elif compare_rep == 'N': ax_title = f"({row_idx+1}, {col_idx+1}) (N={comp}) ≥ c x (N={comp_alt}): {row_val[1]}, {col_rep}={col_val}"
                else: ax_title = f"({row_idx+1}, {col_idx+1}) {comp} ≥ c x {comp_alt}: {row_val[1]}, {col_rep}={col_val}"

                ax_cellText  = []         
                print(f"{row_idx}, {col_idx}, {ax_title}", end =" / ")
                    
                # Get subsampled data for RKS test
                if compare_rep == 'opt_rep': varying_list = ['lr']
                elif compare_rep == 'lr':    varying_list = [compare_rep]
                else:                        varying_list = [compare_rep, 'lr']
                    
                subset_setting = {'hypo': row_val[1]+"_hypo", col_rep: col_val}
                
                for i in range(len(fixed_rep_keys)): subset_setting[fixed_rep_keys[i]] = fixed_rep_values[i]

                _, _, refined_data = self.RefinedData(setting = subset_setting)
                all_varying_config, subsample_dict = self.SubsampleDataByVarying(refined_data, varying_list)

                ### --- Plot comparison scatter plot or table: Start --- ###
                mmdbest_record = {}
                for sub_key in subsample_dict.keys():
                    # (0) Get data for sub_key
                    sub_data  = subsample_dict[sub_key]
                    
                    # (1) Save subset data into 'mmdbest_record' (depending on 'datatype')
                    if datatype == 'all_steps':
                        if optimizer_rep == True:
                            mmdbest  = "mmd" + mmdtype
                            sub_data = sub_data.loc[sub_data.groupby('rep_no:smpl')[mmdbest].idxmax()]
                            mmdbest_record[sub_key] = sub_data[mmdbest].values

                        else:
                            sub_data = sub_data[sub_data['rep_no:smpl'] == 1]
                        
                    elif datatype == 'only_checkpoints':
                        if mmdtype == "last":
                            raise NotImplementedError()
                        elif ':' in mmdtype:
                            cp = int(mmdtype.split(":")[-1])
                            eff_tI = cp // 100

                        sub_data = sub_data[sub_data['eff_tI'] == eff_tI]

                        # For 'only_checkpoints' datatype, repetition of optimization can be compared
                        if compare_rep == 'opt_rep':
                            if all_or_one == 'all':
                                mmdbest_record[' / '.join(['opt_rep=3', sub_key])] = sub_data['ipm_max'].values
                                mmdbest_record[' / '.join(['opt_rep=1', sub_key])] = sub_data['ipm_optim:1'].values
                            elif all_or_one == 'one':
                                mmdbest_record[' / '.join(['opt_rep=3', sub_key])] = sub_data['ipm_max_one'].values
                                mmdbest_record[' / '.join(['opt_rep=1', sub_key])] = sub_data['ipm_one:1'].values
                            else:
                                raise NotImplementedError()
                            
                        elif optimizer_rep == True:
                            if all_or_one == 'all':   mmdbest_record[sub_key] = sub_data['ipm_max'].values
                            elif all_or_one == 'one': mmdbest_record[sub_key] = sub_data['ipm_max_one'].values
                            else:                     raise NotImplementedError()
                        else:
                            if all_or_one == 'all':   mmdbest_record[sub_key] = sub_data['ipm_optim:1'].values
                            elif all_or_one == 'one': mmdbest_record[sub_key] = sub_data['ipm_one:1'].values
                            else:                     raise NotImplementedError()
                            
                ### (2-1) Generate table
                if vistype == 'table':
                    for lr in lr_list:
                        win_count = []
                        larger, smaller = f"{compare_rep}={comp} / lr={lr}", f"{compare_rep}={comp_alt} / lr={lr}"

                        for c in c_list:
                            compare = mmdbest_record[larger] >= c * mmdbest_record[smaller]
                            win_count.append(sum(compare))

                        ax_cellText.append(win_count)

                    ax.table(cellText=ax_cellText, rowLabels = lr_list, colLabels = [f"c=\n{c}" for c in c_list],
                             cellLoc='center', loc='center', edges='BR', fontsize = 14)
                    
                    if len(lr_list) == 2:   ax.text(0.4, 0.82, ax_title, ha='center', va='center', fontsize = 13)
                    elif len(lr_list) == 5: ax.text(0.4, 0.87, ax_title, ha='center', va='center', fontsize = 13)
                    else:                   ax.text(0.4, 0.85, ax_title, ha='center', va='center', fontsize = 13)
                        
                    ax.text(-0.18, 0.5, 'lr', va='center')
                
                ### (2-2) Generate scatter plot
                elif vistype == 'scatter':
                    
                    if compare_rep == 'log_nolog':
                        comp_ = 'no-log' if comp == 'nolog' else comp
                        comp_alt_ = 'no-log' if comp_alt == 'nolog' else comp_alt
                        ax.set_xlabel(f"{comp_}")
                        ax.set_ylabel(f"{comp_alt_}")
                    else:
                        ax.set_xlabel(f"{compare_rep}={comp}")
                        ax.set_ylabel(f"{compare_rep}={comp_alt}")
                    
                    if compare_rep != 'lr':
                        assert len(lr_list) == 1
                        lr = lr_list[0]
                        x_value, y_value = mmdbest_record[f"{compare_rep}={comp} / lr={lr}"], mmdbest_record[f"{compare_rep}={comp_alt} / lr={lr}"]
                    else:
                        x_value, y_value = mmdbest_record[f"{compare_rep}={comp}"], mmdbest_record[f"{compare_rep}={comp_alt}"]
                    
                    # Different color based on location of points in scatter plot
                    ratio = y_value / x_value
                    color_list, count= [], {i:0 for i in range(7)}
                    
                    for r in ratio:
                        if (0.9 <= r) and (r <= 1/0.9):
                            color_list.append('green')
                            count[0] += 1
                        elif r > 1/0.7:
                            color_list.append('black')
                            count[1] += 1
                        elif ((1/0.8 < r) and (r <= 1/0.7)):
                            color_list.append('red')
                            count[2] += 1
                        elif ((1/0.9 < r) and (r <= 1/0.8)):
                            color_list.append('orange')
                            count[3] += 1
                        elif ((0.8 <= r) and (r < 0.9)):
                            color_list.append('orange')
                            count[4] += 1
                        elif ((0.7 <= r) and (r < 0.8)):
                            color_list.append('red')
                            count[5] += 1
                        elif r < 0.7:
                            color_list.append('black')
                            count[6] += 1

                    print(len(x_value), len(y_value), end=" / ")
                    count_dict[str(row_idx)+str(col_idx)] = count
                    
                    curr_max = np.amax(np.concatenate((x_value, y_value)))
                    if curr_max > ax_max:
                        ax_max = curr_max

                    ax.scatter(x_value, y_value, color = color_list, s=24)
                    ax.set_title(f"{row_val[1]} hypo, d={col_val}", fontsize = 13)
                   
                    # May change xlim and ylim a little bit, such as:
                    #     ax.set_xlim(0, 1.02 * curr_max) 
                    #     ax.set_ylim(0, 1.02 * curr_max)
                    # or:
                    #     ax.axis('equal')
                    
                ### --- Plot comparison scatter plot or table: End --- ###
                
            ### End of "for col_val in col_list" ###
        ### End of "for row_val in row_list" ###
        
        ### Attribute of scatter plot or table ###
        if vistype == 'table':
            if len(lr_list) == 2:
                fig.subplots_adjust(wspace=0.3, hspace=-0.25)
                fig.subplots_adjust(top = 0.97)
            elif len(lr_list) == 5:
                fig.subplots_adjust(wspace=0.3, hspace=-0.2)
                fig.subplots_adjust(top = 0.98)
            else:
                fig.subplots_adjust(wspace=0.3, hspace=-0.25)
                fig.subplots_adjust(top = 0.99)
        
        if vistype == 'scatter':
            for row_idx in range(row_num):
                for col_idx in range(col_num):
                    
                    if row_num > 1: ax = axes[row_idx, col_idx]
                    else:           ax = axes[col_idx]

                    ax.set_xlim(0, 1.02 * ax_max)
                    ax.set_ylim(0, 1.02 * ax_max)

                    xcut = np.linspace(0, 1.02 * ax_max, 100)
                    ax.plot(xcut, xcut, color = 'k')

                    for c in [0.9, 0.8, 0.7]:
                        ax.plot(xcut, c * xcut, color = 'k', alpha = 0.3)
                        ax.plot(xcut, (1/c) * xcut, color = 'k', alpha = 0.3)
    
                    count = count_dict[str(row_idx)+str(col_idx)]
        
                    ax.text(ax_max * 0.93, ax_max * 0.98, str(count[0]), ha='center', va='center', fontsize = 11)
            
                    ax.text(ax_max * 0.60, ax_max * 0.98, str(count[1]), ha='center', va='center', fontsize = 11)
                    ax.text(ax_max * 0.71, ax_max * 0.98, str(count[2]), ha='center', va='center', fontsize = 11)
                    ax.text(ax_max * 0.82, ax_max * 0.98, str(count[3]), ha='center', va='center', fontsize = 11)
                    
                    ax.text(ax_max * 0.98, ax_max * 0.83, str(count[4]), ha='center', va='center', fontsize = 11)
                    ax.text(ax_max * 0.98, ax_max * 0.73, str(count[5]), ha='center', va='center', fontsize = 11)
                    ax.text(ax_max * 0.98, ax_max * 0.63, str(count[6]), ha='center', va='center', fontsize = 11)
                    
            fig.subplots_adjust(wspace=0.25, hspace=0.3)
            fig.subplots_adjust(top = 0.92)
            
        ### Save
        fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)