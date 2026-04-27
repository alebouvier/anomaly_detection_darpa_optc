#!/bin/bash

#OAR -p nodeset NOT IN (abacus27) and gpudevice=0
#OAR -l host=1,walltime=2:00:00
#OAR -O job_%jobid%.out
#OAR -E job_%jobid%.err

log() {
  echo "[$(date)] $1"
}

log "Start Job"
cd ~/optc_ad_project

log "Activation environement"
source ~/miniconda3/etc/profile.d/conda.sh

conda activate optc-gpu


echo "Conda:"
which conda

echo "Python:"
which python
python --version

export DATA_BASE="./data"


log "Start validation link prediction"
python src/optc_ad.py -t validate_link_prediction --model_name DyGFormer

log "Start test anomaly detection"
python src/optc_ad.py -t test_anomaly_detection --model_name DyGFormer

log "End Job"