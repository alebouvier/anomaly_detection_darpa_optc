#!/bin/bash

#OAR -p nodeset NOT IN (abacus27) and gpudevice=0
#OAR -l host=1,walltime=1:00:00
#OAR -O job_%jobid%.out
#OAR -E job_%jobid%.err

log() {
  echo "[$(date)] $1"
}

log "Start Job"
cd ~/optc_ad_project

log "Activation environement"
source .venv/bin/activate


log "Start test anomaly detection"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py -t test_anomaly_detection --model_name TGAT

log "End Job"