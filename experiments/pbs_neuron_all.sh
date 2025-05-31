#!/bin/bash

#PBS -N neurons
#PBS -lselect=1:ncpus=8
#PBS -q highcpu
#PBS -o pbs_logs/
#PBS -e pbs_logs/

source ~/.bashrc
conda activate ml-pws

cd $PBS_O_WORKDIR

OUTPUT_FILE="full_out.json"

python neuron_analysis.py \
    ../data/barmovie0113extended.data \
    --neurons $(seq 0 49) \
    -o experiment2/$OUTPUT_FILE \
    --threads 8
