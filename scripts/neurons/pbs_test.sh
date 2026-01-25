#PBS -lselect=16:ncpus=1
#PBS -lplace=free
#PBS -q highcpu
#PBS -N torch_test

export PATH=$PBS_O_PATH
export PYTHON=$(which python)

SCRIPT='
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Intra-op threads: {torch.get_num_threads()}")
print(f"Inter-op threads: {torch.get_num_interop_threads()}")
print(f"MKL available: {torch.backends.mkl.is_available()}")
print(f"OpenMP available: {torch.has_openmp}")
'

mpiexec --wdir $PBS_O_WORKDIR $PYTHON -c "$SCRIPT"