"""
Script to perform PWS (Path Weight Sampling) estimation on trained neuron models.

This script loads trained model files and computes mutual information estimates
using the PWS method with stochastic harmonic oscillator dynamics.

Supports MPI parallelization for processing multiple models in parallel.
Falls back to single-process execution if MPI is not available.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

# Optional MPI support - fall back to single process if not available
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _has_mpi = _size > 1  # Only use MPI if actually running with multiple processes
except ImportError:
    _comm = None
    _rank = 0
    _size = 1
    _has_mpi = False


def get_rank():
    """Get current MPI rank (0 if MPI not available)."""
    return _rank


def get_size():
    """Get total number of MPI processes (1 if MPI not available)."""
    return _size


def barrier():
    """MPI barrier (no-op if MPI not available)."""
    if _has_mpi and _comm is not None:
        _comm.Barrier()


def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


from ml_pws.models.neuron_models import SpikeModel
from ml_pws.mutual_information import pws_estimate

from train_models import (
    ModelSpec,
    BIN_WIDTH,
    SECONDS_PER_UNIT,
)


def load_model(ckpt_path: Path, device: torch.device | None = None):
    """Load a trained SpikeModel from a checkpoint file.

    Args:
        ckpt_path: Path to the checkpoint file
        device: Device to load the model to (default: best available)
    """
    if device is None:
        device = get_device()

    model = SpikeModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    # # To speed up on GPU
    # model.net = torch.compile(model.net, fullgraph=True)

    return model


def process_model(spec: ModelSpec, config, device: torch.device):
    """Process a single model specification.

    Args:
        spec: Dictionary with model specification (neuron_id, model_path, output_path)
        config: Configuration dictionary with PWS parameters
        device: Device to run computations on
    """
    ckpt_path = (
        spec.output_dir.parent
        / "training_logs"
        / spec.name
        / "checkpoints"
        / "epoch=9-step=540.ckpt"
    )
    output_path = spec.output_dir / "pws_estimate.json"

    # Check if output already exists (skip if present)
    if output_path.exists():
        print(f"Rank {get_rank()}: {spec.name} results already exist, skipping")
        return

    # Check if model exists
    if not ckpt_path.exists():
        print(f"Rank {get_rank()}: Warning: Model not found at {ckpt_path}, skipping")
        return

    print(f"Rank {get_rank()}: Processing {spec.name}...")

    # Load model
    model = load_model(ckpt_path, device=device)

    # Perform PWS estimation
    t = torch.arange(config["seq_len"]) * BIN_WIDTH * SECONDS_PER_UNIT
    pws_result, resamples = pws_estimate(
        model,
        t,
        config["N"],
        config["M"],
        bin_width=BIN_WIDTH * SECONDS_PER_UNIT,
        batch_size=64,
    )

    # Compute mutual information
    mi = pws_result[:, 0] - pws_result[:, 1]

    # Prepare results dictionary
    results = {
        "neurons": spec.neurons,
        "checkpoint_path": str(ckpt_path),
        "N": config["N"],
        "M": config["M"],
        "Bin width (s)": BIN_WIDTH * SECONDS_PER_UNIT,
        "Duration (s)": config["seq_len"] * BIN_WIDTH * SECONDS_PER_UNIT,
        "pws_result": {
            "t": t.tolist(),
            "log_conditional": pws_result[:, 0].mean(1).tolist(),
            "log_conditional_std": pws_result[:, 0].std(1).tolist(),
            "log_marginal": pws_result[:, 1].mean(1).tolist(),
            "log_marginal_std": pws_result[:, 1].std(1).tolist(),
            "mutual_information": mi.mean(1).tolist(),
            "mutual_information_std": mi.std(1).tolist(),
            "resamples": resamples.tolist(),
        },
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    sem = mi.std(1)[-1] / np.sqrt(config["N"])
    print(
        f"Rank {get_rank()}: {spec.name} completed. Final MI: {mi.mean(1)[-1].item():.4f} ± {sem:.4f}"
    )


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) != 2:
        print("Usage: python estimate_pws.py <config_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    rank = get_rank()
    size = get_size()

    # Get list of models to process
    models = []
    for spec in config["models"]:
        spec["output_dir"] = Path(spec["output_dir"])
        models.append(ModelSpec(**spec))

    # Distribute work across MPI ranks (or process all if single process)
    local_models = models[rank::size]

    device = get_device()

    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(device)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        else:
            print(f"Intra-op threads: {torch.get_num_threads()}")
            print(f"Inter-op threads: {torch.get_num_interop_threads()}")
            print(f"MKL available: {torch.backends.mkl.is_available()}")
            print(f"OpenMP available: {torch.has_openmp}")

        if _has_mpi:
            print(f"Starting PWS estimation with {size} MPI processes")
        else:
            print("Starting PWS estimation (single process, MPI not available)")
        print(f"Total models: {len(models)}")
        print(
            f"Parameters: N={config['N']}, M={config['M']}, seq_len={config['seq_len']}"
        )

    # Process models assigned to this rank
    for spec in local_models:
        process_model(spec, config, device)

    # Wait for all ranks to complete
    barrier()

    if rank == 0:
        print("All PWS estimations completed!")
