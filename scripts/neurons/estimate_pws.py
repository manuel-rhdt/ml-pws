"""
Script to perform PWS (Path Weight Sampling) estimation on trained neuron models.

This script loads a trained model file and computes mutual information estimates
using the PWS method with stochastic harmonic oscillator dynamics.
"""

import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Perform PWS estimation on trained neuron models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file path (default: <model_path>_pws_results.json)",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=1,
        help="Number of neurons in the model",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size (default: 4 * n_neurons)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["RNN", "CNN"],
        default="CNN",
        help="Model architecture type",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=20,
        help="Kernel size for CNN model",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=400,
        help="Number of trajectories to sample",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=2048,
        help="Number of particles for marginal estimation",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="Sequence length (number of time bins)",
    )

    args = parser.parse_args()

    # Set default hidden size
    if args.hidden_size is None:
        args.hidden_size = args.n_neurons * 4

    # Set default output path
    if args.output is None:
        args.output = args.model_path.parent / f"{args.model_path.stem}_pws_results.json"

    print(f"Loading model from {args.model_path}")
    model = load_model(
        args.model_path,
        n_neurons=args.n_neurons,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        model_type=args.model_type,
        kernel_size=args.kernel_size if args.model_type == "CNN" else None,
    )

    print(f"Performing PWS estimation with N={args.N}, M={args.M}")
    t = torch.arange(args.seq_len) * BIN_WIDTH * SECONDS_PER_UNIT
    pws_result = pws_estimate(model, t, args.N, args.M)

    # Compute mutual information
    mi = pws_result[:, 0] - pws_result[:, 1]

    # Prepare results dictionary
    results = {
        "args": {
            "model_path": str(args.model_path),
            "n_neurons": args.n_neurons,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "model_type": args.model_type,
            "kernel_size": args.kernel_size,
            "N": args.N,
            "M": args.M,
            "seq_len": args.seq_len,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Final MI estimate: {mi.mean(0)[-1].item():.4f} ± {mi.std(0)[-1].item():.4f}")


if __name__ == "__main__":
    main()
