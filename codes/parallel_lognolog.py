import os
import sys
import pickle
import pandas as pd

import sampling

"""
The main experiment is run in Line 83. If you want to run different experiment, change this line.
"""

################### Arguments from 'sys' ###################
"""
This is due to "--array" option in the bash file.
"""
ARRAY_IDX = int(sys.argv[1]) # refers '$SLURM_ARRAY_TASK_ID '
TIME_STAMP = sys.argv[2] # refers '$time_stamp'

########## Get certain configuration ##########
"""
'config_this_time' only matters in the later part of the code.
This depends on ARRAY_IDX.
"""
file_name = "array_dict.csv"
array_csv = pd.read_csv(file_name)

file_name = "config_all.csv"
config_all = pd.read_csv(file_name)
    
array_this_time = array_csv.loc[[ARRAY_IDX-1]].copy()
array_keys = array_this_time.columns.tolist()
ARRAY_KEYS = "_".join(array_keys) # will be used at the end (saving file)

config_this_time = config_all
for key in array_keys:
    mask = (config_this_time[key] == array_this_time[key].item())
    config_this_time = config_this_time.loc[mask]
    
################# Preparation #################
"""
Will be used in the definition of SETTING and the output.
"""
setting_keys = {'sample': ['sample_key', 'task', 'v', 'd', 'k', 'n'], # 6
                'optim':  ['optim_key', 'optim', 'lr', 'N', 'tB', 'tI', # 6
                           'lamb', 'optimizer', 'device', 'smpl_type'] # 4
               }

############## Run by each config (i.e. 'config_this_time') #############
"""
Actual experiements are conducted here.
"""
for i in range(len(config_this_time)):
    cf = config_this_time.iloc[i].copy()
    df_cf = config_this_time.iloc[[i]].copy() # will be used at the end (saving file)
    
    ##### Setup #####
    SETTING = {'sample': {}, 'optim':{}}
    
    for t in ['sample', 'optim']:
        for key in setting_keys[t]:
            if key == 'n': # This only happenes when t=='sample'
                SETTING[t][key] = {}
                SETTING[t][key]['P'] = cf['nP']
                SETTING[t][key]['Q'] = cf['nQ']
            else:
                SETTING[t][key] = cf[key]
    
    ##### Extra setup #####
    add_seed = SETTING['sample']['d'] + SETTING['sample']['n']['P'] + SETTING['sample']['n']['Q'] + int(100 * SETTING['sample']['v'])
    S_SEED       = int(cf['s_seed'] + add_seed) 
    O_SEED       = int(cf['s_seed'] + add_seed) 
    SAMPLE_KEY   = SETTING['sample']['sample_key']
    OPTIM_KEY    = SETTING['optim']['optim_key']
    REP_SAMPLE   = int(cf['rep_sample'])
    REP_OPTIM    = int(cf['rep_optim'])
    
    ##### Run #####
    # 1. Initialization
    rep_sample = sampling.MySample(sdict = SETTING['sample'], s_seed = S_SEED, verbose = False)
    rep_sample.OptInit(odict = SETTING['optim'], o_seed = O_SEED)
    
    # 2. Key part -- actual experiment (sampling, optimization, repetition, etc) happens
    # You may select 'lognolog' option and non-empty list, based on your goal.
    df_rep_permute = rep_sample.AltNullRepeat(REP_SAMPLE, REP_OPTIM, SAMPLE_KEY, OPTIM_KEY, 'logonly', []) 
    
    # 3. Save to 'df_output'
    df_cf['array_idx'] = ARRAY_IDX
    df_cf['array_key'] = ARRAY_KEYS

    df_output_one_config = df_cf.join(df_rep_permute, how = 'cross') 
    
    if i == 0: df_output = df_output_one_config
    else:      df_output = pd.concat([df_output, df_output_one_config], axis=0)

############## Save to file ##############
folder = f"output_{TIME_STAMP}_{ARRAY_KEYS}"
if not os.path.exists(folder):
    os.mkdir(folder)

file_name = f"{folder}/{ARRAY_IDX}_RepPer.csv"
df_output.to_csv(file_name, index = False)