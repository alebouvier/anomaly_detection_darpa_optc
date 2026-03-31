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
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t graph

log "Start training model word2vec"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t w2v

log "Start extracting features"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t feature 


log "Start preprocessing graph to csv"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t preprocessing


log "Start training link prediction"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t train_link_prediction

log "Start validation link prediction"
~/.pyenv/versions/3.11.15/bin/python src/optc_ad.py --clients 201 -t validate_link_prediction

log "End Job"