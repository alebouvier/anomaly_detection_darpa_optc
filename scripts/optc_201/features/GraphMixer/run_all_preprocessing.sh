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
source ~/miniconda3/etc/profile.d/conda.sh

conda activate optc-gpu


echo "Conda:"
which conda

echo "Python:"
which python
python --version

export DATA_BASE="./data"

log "Start preprocessing log data to graph"
python src/optc_ad.py --clients 201 -t graph --with_features

log "Start training model word2vec"
python src/optc_ad.py --clients 201 -t w2v --with_features

log "Start extracting features"
python src/optc_ad.py --clients 201 -t feature --with_features


log "Start preprocessing graph to csv"
python src/optc_ad.py --clients 201 -t preprocessing --with_features