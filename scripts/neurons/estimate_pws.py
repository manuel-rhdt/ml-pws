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
from torch import nn
import torch.nn.functional as F
import torchsde

from ml_pws.models.neuron_models import (
    StochasticHarmonicOscillator,
    SpikeModel,
    OMEGA_0,
    TAU,
)

from train_models import (
    ModelSpec,
    BIN_WIDTH,
    SECONDS_PER_UNIT,
)

DAMPING_COEFFICIENT = 1.0 / (2 * OMEGA_0 * TAU)
STATIONARY_SIGMA = torch.tensor([1.0, 1 / (4 * DAMPING_COEFFICIENT**2)])


def log_probability(model, s, x, cache=None):
    """Compute log probability of spike observations given stimulus.

    Args:
        model: Neural network model (the .net attribute of SpikeModel)
        s: Stimulus tensor (seq_len, batch_size)
        x: Spike count tensor (seq_len, batch_size, n_neurons)

    Returns:
        Log probabilities (seq_len, batch_size)
    """
    model.eval()
    log_intensity, cache = model(s, x, cache)
    return (
        -F.poisson_nll_loss(
            log_intensity, x, log_input=True, full=True, reduction="none"
        ).sum(-1),
        cache,
    )


def resample(arr, indices, batch_indices):
    # resampling with index tricks
    extra_dims = arr.ndim - 2
    ii = indices[(Ellipsis,) + (None,) * extra_dims].expand(-1, -1, *arr.shape[2:])
    arr[batch_indices] = torch.gather(arr[batch_indices], dim=1, index=ii)


class ParticleFilter(nn.Module):
    """
    A bootstrap particle filter.
    """

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        n_particles: int = 128,
        resample_criterion=None,
    ):
        super().__init__()
        seq_len, batch_size, _ = x.shape
        self.batch_size = batch_size
        self.model = model
        self.sho = StochasticHarmonicOscillator()
        self.n = 0  # discrete time
        self.y0 = torch.randn(batch_size, n_particles, 2) * STATIONARY_SIGMA
        self.x = x
        self.n_particles = n_particles
        self.log_marginals = []
        self.cache = None
        self.weights = torch.zeros(batch_size, n_particles)
        self.resample_criterion = (
            n_particles / 2 if resample_criterion is None else resample_criterion
        )
        self.resamples = []

    @torch.no_grad()
    def forward(self, n_end):
        """
        The main particle filter routine.

        Advance the particles to `n_end`. This propagates all particles forward in time
        to `n_end * delta_t` where delta_t is the discretization. Computes the
        likelihood-increment for all particles.

        Then whenever the effective sample size (ESS) drops below half the particle
        count, we resample.
        """
        NM = self.batch_size * self.n_particles
        n = self.n
        self.n = n_end

        # STEP 1: Propagate input
        n_range = torch.arange(n, n_end + 1)
        t_grid = n_range * BIN_WIDTH * SECONDS_PER_UNIT
        y = torchsde.sdeint(self.sho, self.y0.view(NM, 2), t_grid, dt=1 / 100)
        assert isinstance(y, torch.Tensor)

        y = y.view((-1, self.batch_size, self.n_particles, 2))
        self.y0 = y[-1, ...]
        s = y[1:, ..., 0]

        # STEP 2: Compute likelihood
        s = s.view(n_end - n, NM)
        x = self.x[n:n_end].repeat_interleave(self.n_particles, 1)

        logp_increment, self.cache = log_probability(
            self.model, s.view(-1, NM), x, self.cache
        )
        logp_increment = logp_increment.reshape(
            (-1, self.batch_size, self.n_particles)
        ).cumsum(0)
        self.log_marginals.append(
            torch.logsumexp(self.weights + logp_increment, dim=-1)
            - np.log(self.n_particles)
        )

        self.weights += logp_increment[-1]

        ess = 1 / torch.sum(F.softmax(self.weights, dim=1) ** 2, dim=1)

        # STEP 3: Resample particle state (cache and y0)
        batch_indices = torch.where(ess < self.resample_criterion)[0]
        if len(batch_indices) == 0:
            self.resamples.append((n_end, 0))
            return

        dist = torch.distributions.categorical.Categorical(
            logits=self.weights[batch_indices]
        )
        indices = dist.sample((self.n_particles,)).T

        # resampling
        cache_view = self.cache.view(
            (self.batch_size, self.n_particles) + self.cache.shape[-2:]
        )[:, :, 0]
        resample(cache_view, indices, batch_indices)
        resample(self.y0, indices, batch_indices)
        # reset the weights of all resampled particles to the current log_marginal estimate
        self.weights[batch_indices] = self.log_marginals[-1][-1, batch_indices, None]
        self.resamples.append((n_end, len(batch_indices)))


def smc_marginal_estimate(model, x, num_particles=256, block_size=5):
    n_max, _, _ = x.shape  # max sequence length
    pf = ParticleFilter(model, x, num_particles)
    for n_end in range(block_size, n_max + 1, block_size):
        pf(n_end)
    return pf


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
    s_test = y[:, :, 0]
    x_test = model.net.sample(s_test)

    log_conditional, _ = log_probability(model.net, s_test, x_test)
    log_conditional = log_conditional.cumsum(0)

    pf = smc_marginal_estimate(model.net, x_test, num_particles=M)
    log_marginal = torch.cat(pf.log_marginals)  # (seq_len, batch_size)

    return torch.stack([log_conditional, log_marginal], dim=1)  # (seq_len, 2, N)


def load_model(ckpt_path: Path):
    """Load a trained SpikeModel from a checkpoint file."""
    # Create model instance with same architecture
    model = SpikeModel.load_from_checkpoint(ckpt_path)
    model.eval()

    return model


def process_model(spec: ModelSpec, config):
    """Process a single model specification.

    Args:
        spec: Dictionary with model specification (neuron_id, model_path, output_path)
        config: Configuration dictionary with PWS parameters
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
        print(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: {spec.name} results already exist, skipping"
        )
        return

    # Check if model exists
    if not ckpt_path.exists():
        print(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: Warning: Model not found at {ckpt_path}, skipping"
        )
        return

    print(f"Rank {MPI.COMM_WORLD.Get_rank()}: Processing {spec.name}...")

    # Load model
    model = load_model(ckpt_path)

    # Perform PWS estimation
    t = torch.arange(config["seq_len"]) * BIN_WIDTH * SECONDS_PER_UNIT
    pws_result = pws_estimate(model, t, config["N"], config["M"])

    # Compute mutual information
    mi = pws_result[:, 0] - pws_result[:, 1]

    # Prepare results dictionary
    results = {
        "neurons": spec.neurons,
        "checkpoint_path": str(ckpt_path),
        "pws_result": {
            "t": t.tolist(),
            "log_conditional": pws_result[:, 0].mean(1).tolist(),
            "log_conditional_std": pws_result[:, 0].std(1).tolist(),
            "log_marginal": pws_result[:, 1].mean(1).tolist(),
            "log_marginal_std": pws_result[:, 1].std(1).tolist(),
            "mutual_information": mi.mean(1).tolist(),
            "mutual_information_std": mi.std(1).tolist(),
        },
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Rank {MPI.COMM_WORLD.Get_rank()}: {spec.name} completed. Final MI: {mi.mean(1)[-1].item():.4f} ± {mi.std(1)[-1].item():.4f}"
    )


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
    models = []
    for spec in config["models"]:
        spec["output_dir"] = Path(spec["output_dir"])
        models.append(ModelSpec(**spec))

    # Distribute work across MPI ranks
    local_models = models[rank::size]

    if rank == 0:
        print(f"Starting PWS estimation with {size} MPI processes")
        print(f"Total models: {len(models)}")
        print(
            f"Parameters: N={config['N']}, M={config['M']}, seq_len={config['seq_len']}"
        )

    # Process models assigned to this rank
    for spec in local_models:
        process_model(spec, config)

    # Wait for all ranks to complete
    comm.Barrier()

    if rank == 0:
        print("All PWS estimations completed!")
