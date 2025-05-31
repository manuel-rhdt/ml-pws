#!/bin/bash

#PBS -N neurons
#PBS -lselect=1:ncpus=1
#PBS -q highcpu
#PBS -o pbs_logs/
#PBS -e pbs_logs/
#PBS -J 0-49

source ~/.bashrc
conda activate ml-pws

cd $PBS_O_WORKDIR

# Get the current neuron index from the job array
NEURON_INDEX=${PBS_ARRAY_INDEX}

# Define output filename based on neuron index
OUTPUT_FILE="out_${NEURON_INDEX}.json"
# OUTPUT_FILE="full_out.json"

python neuron_analysis.py \
    ../data/barmovie0113extended.data \
    --neurons $NEURON_INDEX \
    -o experiment2/$OUTPUT_FILE \
    --threads 1
