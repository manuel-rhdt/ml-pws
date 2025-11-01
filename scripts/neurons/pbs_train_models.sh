#PBS -lselect=16:ncpus=1
#PBS -lplace=free
#PBS -q highcpu
#PBS -N neurons

set -e

export PATH=$PBS_O_PATH
export PYTHON=$(which python)
export CONFIG_PATH="$PBS_O_WORKDIR/scripts/neurons/config.json"
export SCRIPT_PATH="$PBS_O_WORKDIR/scripts/neurons/train_models.py"

echo mpiexec $PYTHON $SCRIPT_PATH $CONFIG_PATH
mpiexec $PYTHON $SCRIPT_PATH $CONFIG_PATH

