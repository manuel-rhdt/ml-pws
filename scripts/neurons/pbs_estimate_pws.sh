#!/bin/bash
#PBS -lselect=50:ncpus=1
#PBS -lplace=free
#PBS -q highcpu
#PBS -N pws_neurons

set -e

export PATH=$PBS_O_PATH
export PYTHON=$(which python)
export CONFIG_PATH="$PBS_O_WORKDIR/scripts/neurons/pws_config.json"
export SCRIPT_PATH="$PBS_O_WORKDIR/scripts/neurons/estimate_pws.py"

cd $PBS_O_WORKDIR

echo "Starting PWS estimation for all neurons"
echo "Config: $CONFIG_PATH"
echo "Script: $SCRIPT_PATH"
echo "Working directory: $PBS_O_WORKDIR"
echo ""

mpiexec $PYTHON $SCRIPT_PATH $CONFIG_PATH

echo ""
echo "PWS estimation completed!"
