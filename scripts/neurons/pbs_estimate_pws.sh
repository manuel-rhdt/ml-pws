#PBS -lselect=4:ncpus=4
#PBS -lplace=free
#PBS -q highcpu
#PBS -N pws_neurons

set -e

export PATH=$PBS_O_PATH
export PYTHON=$(which python)
export CONFIG_PATH="$PBS_O_WORKDIR/scripts/neurons/config.json"
export SCRIPT_PATH="$PBS_O_WORKDIR/scripts/neurons/estimate_pws.py"

echo "Starting PWS estimation for all neurons"
echo "Config: $CONFIG_PATH"
echo "Script: $SCRIPT_PATH"
echo "Working directory: $PBS_O_WORKDIR"
echo ""

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo mpiexec --wdir $PBS_O_WORKDIR $PYTHON $SCRIPT_PATH $CONFIG_PATH
mpiexec  --bind-to none --wdir $PBS_O_WORKDIR $PYTHON $SCRIPT_PATH $CONFIG_PATH

echo ""
echo "PWS estimation completed!"
