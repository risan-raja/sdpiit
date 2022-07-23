#!/bin/bash
#PBS -l nodes=s042-n002:ppn=2,walltime=2:00:00
cd $PBS_O_WORKDIR
# MYCONDA=/glob/development-tools/versions/oneapi/2022.1.2/oneapi/intelpython/latest/etc/profile.d/conda.sh
MOB=/home/u149902/bin/micromamba
source /opt/intel/oneapi/setvars.sh
source $MYCONDA
conda activate dnn
python -m tbb /home/u149902/tps_june/notebooks/svm1.py
# Remember to have an empty line at the end of the file; otherwise the last command will not run