#!/bin/bash
#PBS -l nodes=s001-n002:ppn=2,walltime=12:00:00
cd $PBS_O_WORKDIR
MYCONDA=/glob/development-tools/versions/oneapi/2022.1.2/oneapi/intelpython/latest/etc/profile.d/conda.sh
MOB=/home/u149902/micromamba/envs/pro
# source /opt/intel/oneapi/setvars.sh
source $MYCONDA
conda activate $MOB
echo "Success"
python /home/u149902/sdpiit/notebooks/xgboost_optuna.py
# python -m tbb /home/u149902/tps_june/notebooks/svm1.py
# Remember to have an empty line at the end of the file; otherwise the last command will not run

