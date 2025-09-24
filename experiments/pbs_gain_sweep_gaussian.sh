#!/bin/bash

#PBS -N gain-sweep
#PBS -lselect=1:ncpus=1
#PBS -q highcpu
#PBS -o pbs_logs/
#PBS -e pbs_logs/
#PBS -J 1-20

source ~/.bashrc
conda activate ml-pws

cd $PBS_O_WORKDIR

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# corresponds to `np.logspace(np.log10(0.025), np.log10(20), 20)`
gain_values=(0.025 0.03554137 0.05052757 0.07183277 0.10212142 0.14518143 0.2063979 0.2934266 0.41715138 0.59304534 0.84310586 1.19860564 1.70400367 2.4225053 3.44396672 4.89613243 6.96061104 9.89558733 14.06811098 20.0)
GAIN=${gain_values[$PBS_ARRAY_INDEX-1]}

python 05-ar_input_mlpws.py --gain $GAIN --ar_std 1.0 --output_noise 0.2 -o experiments/gain_sweep01/gaussian1_gain=$GAIN --estimator Gaussian
python 05-ar_input_mlpws.py --num_pairs 100000 --gain $GAIN --ar_std 1.0 --output_noise 0.2 -o experiments/gain_sweep01/gaussian2_gain=$GAIN --estimator Gaussian