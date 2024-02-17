#!/bin/bash
#SBATCH --mail-type     ALL
#SBATCH --mail-user     Your_Email_Address
#SBATCH --job-name      Job_Name
#SBATCH --array         Specify_The_Indices_Of_Array_You_Want_To_Run_e.g._1-100
#SBATCH --partition     Partition_Name
#SBATCH --output        out/out_%j_%a.out (: You_Must_Make_The_Folder_'out'_Before_Running_This_Bash_File)

time_stamp="Anything_You_Want_That_can_Specify_The_Time_Or_Job_(cf.Regarding_Virtual_Environment_Below_Check_'mmd-env.yml'_File)"

source activate ~/venv/mmd-env
which python
pip freeze

python parallel_mmdcurve.py $SLURM_ARRAY_TASK_ID $time_stamp #(: Or, python parallel_lognolog.py $SLURM_ARRAY_TASK_ID $time_stamp)