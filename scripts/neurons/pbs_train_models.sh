#PBS -lselect=115:ncpus=1
#PBS -lplace=free
#PBS -q highcore
#PBS -N neurons

set -e

export PATH=$PBS_O_PATH
export PYTHON=$(which python)
export CONFIG_PATH="$PBS_O_WORKDIR/scripts/neurons/config.json"
export SCRIPT_PATH="$PBS_O_WORKDIR/scripts/neurons/train_models.py"

conda activate pyenv

echo mpiexec --wdir $PBS_O_WORKDIR $PYTHON $SCRIPT_PATH $CONFIG_PATH
mpiexec --wdir $PBS_O_WORKDIR $PYTHON $SCRIPT_PATH $CONFIG_PATH

