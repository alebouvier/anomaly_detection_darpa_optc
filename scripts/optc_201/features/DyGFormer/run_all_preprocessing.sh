#!/bin/bash

#OAR -p nodeset NOT IN (abacus27) and gpudevice=0
#OAR -l host=1,walltime=3:00:00
#OAR -O job_%jobid%.out
#OAR -E job_%jobid%.err

log() {
  echo "[$(date)] $1"
}

log "Start Job"
cd ~/optc_ad_project

log "Activation environement"
source .venv/bin/activate

log "Start preprocessing log data to graph"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t graph --with_features --model_name DyGFormer

log "Start training model word2vec"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t w2v --with_features --model_name DyGFormer

log "Start extracting features"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t feature --with_features --model_name DyGFormer

log "Start preprocessing graph to csv"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t preprocessing --with_features --model_name DyGFormer