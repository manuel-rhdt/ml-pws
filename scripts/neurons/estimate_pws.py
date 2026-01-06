"""
Script to perform PWS (Path Weight Sampling) estimation on trained neuron models.

This script loads trained model files and computes mutual information estimates
using the PWS method with stochastic harmonic oscillator dynamics.

Supports MPI parallelization for processing multiple models in parallel.
"""

import sys
import json
from pathlib import Path

from mpi4py import MPI

import numpy as np
import torch
import torch.nn.functional as F
import torchsde

from train_models import (
    SpikeModel,
    StochasticHarmonicOscillator,
    BIN_WIDTH,
    SECONDS_PER_UNIT,
)


def log_probability(model, s, x):
    """Compute log probability of spike observations given stimulus.

    Args:
        model: Neural network model (the .net attribute of SpikeModel)
        s: Stimulus tensor (seq_len, batch_size)
        x: Spike count tensor (seq_len, batch_size, n_neurons)

    Returns:
        Log probabilities (seq_len, batch_size)
    """
    model.eval()
    log_intensity = model(s, x)
    return -F.poisson_nll_loss(
        log_intensity, x, log_input=True, full=True, reduction="none"
    ).sum(-1)


def pws_estimate(model, t, N, M):
    """Perform PWS estimation of mutual information.

    Args:
        model: Trained SpikeModel instance
        t: Time grid tensor (seq_len,)
        N: Number of trajectories to sample
        M: Number of particles for marginal estimation

    Returns:
        Tensor of shape (N, 2, seq_len) containing log_conditional and log_marginal
    """
    stoch_osc = StochasticHarmonicOscillator()
    y0 = torch.zeros(N, 2)
    y = torchsde.sdeint(stoch_osc, y0, t, dt=1 / 60)
    s_samples = y[:, :, 0]
    x_samples = model.net.sample(s_samples)

    def pws_samples():
        log_conditional = log_probability(model.net, s_samples, x_samples).cumsum(0)
        for i in range(N):
            y0 = torch.zeros(M, 2)
            y = torchsde.sdeint(stoch_osc, y0, t, dt=1 / 60)
            s = y[:, :, 0]
            logp = log_probability(
                model.net, s, x_samples[:, [i], :].repeat(1, M, 1)
            ).cumsum(0)
            log_marginal = logp.logsumexp(1) - np.log(M)
            yield torch.stack((log_conditional[:, i], log_marginal))

    with torch.no_grad():
        samples = torch.stack(list(pws_samples()))

    return samples


def load_model(model_path, n_neurons, hidden_size, num_layers, model_type, kernel_size):
    """Load a trained SpikeModel from a checkpoint file.

    Args:
        model_path: Path to .pth checkpoint file
        n_neurons: Number of neurons in the model
        hidden_size: Hidden layer size
        num_layers: Number of layers
        model_type: Type of model ("RNN" or "CNN")
        kernel_size: Kernel size for CNN (None for RNN)

    Returns:
        Loaded SpikeModel instance
    """
    # Create model instance with same architecture
    model = SpikeModel(
        n_neurons=n_neurons,
        hidden_size=hidden_size,
        num_layers=num_layers,
        kernel_size=kernel_size,
        model_type=model_type,
    )

    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def process_model(spec, config):
    """Process a single model specification.

    Args:
        spec: Dictionary with model specification (neuron_id, model_path, output_path)
        config: Configuration dictionary with PWS parameters
    """
    neuron_id = spec["neuron_id"]
    model_path = Path(spec["model_path"])
    output_path = Path(spec["output_path"])

    # Check if output already exists (skip if present)
    if output_path.exists():
        print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Neuron {neuron_id} results already exist, skipping")
        return

    # Check if model exists
    if not model_path.exists():
        print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Warning: Model not found at {model_path}, skipping")
        return

    print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Processing neuron {neuron_id}...")

    # Load model
    model = load_model(
        model_path,
        n_neurons=config["n_neurons"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        model_type=config["model_type"],
        kernel_size=config.get("kernel_size") if config["model_type"] == "CNN" else None,
    )

    # Perform PWS estimation
    t = torch.arange(config["seq_len"]) * BIN_WIDTH * SECONDS_PER_UNIT
    pws_result = pws_estimate(model, t, config["N"], config["M"])

    # Compute mutual information
    mi = pws_result[:, 0] - pws_result[:, 1]

    # Prepare results dictionary
    results = {
        "neuron_id": neuron_id,
        "model_path": str(model_path),
        "args": {
            "n_neurons": config["n_neurons"],
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
            "model_type": config["model_type"],
            "kernel_size": config.get("kernel_size"),
            "N": config["N"],
            "M": config["M"],
            "seq_len": config["seq_len"],
        },
        "pws_result": {
            "t": t.tolist(),
            "log_conditional": pws_result[:, :, 0].mean(0).tolist(),
            "log_conditional_std": pws_result[:, :, 0].std(0).tolist(),
            "log_marginal": pws_result[:, :, 1].mean(0).tolist(),
            "log_marginal_std": pws_result[:, :, 1].std(0).tolist(),
            "mutual_information": mi.mean(0).tolist(),
            "mutual_information_std": mi.std(0).tolist(),
        },
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Neuron {neuron_id} completed. Final MI: {mi.mean(0)[-1].item():.4f} ± {mi.std(0)[-1].item():.4f}")


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) != 2:
        print("Usage: python estimate_pws.py <config_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get list of models to process
    models = config["models"]

    # Distribute work across MPI ranks
    local_models = models[rank::size]

    if rank == 0:
        print(f"Starting PWS estimation with {size} MPI processes")
        print(f"Total models: {len(models)}")
        print(f"Parameters: N={config['N']}, M={config['M']}, seq_len={config['seq_len']}")

    # Process models assigned to this rank
    for spec in local_models:
        process_model(spec, config)

    # Wait for all ranks to complete
    comm.Barrier()

    if rank == 0:
        print("All PWS estimations completed!")
