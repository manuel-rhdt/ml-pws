"""
PWS (Path Weight Sampling) estimation of mutual information.

This module provides tools for estimating mutual information using
particle filtering and the PWS method with stochastic harmonic oscillator dynamics.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchsde

from ml_pws.models.neuron_models import (
    StochasticHarmonicOscillator,
    shift_right,
    OMEGA_0,
    TAU,
)

DAMPING_COEFFICIENT = 1.0 / (2 * OMEGA_0 * TAU)
STATIONARY_SIGMA_VALUES = [1.0, 1 / (4 * DAMPING_COEFFICIENT**2)]


def get_stationary_sigma(device=None):
    """Get stationary sigma tensor on the specified device."""
    return torch.tensor(STATIONARY_SIGMA_VALUES, device=device)


def log_probability(model, s, x, cache=None):
    """Compute log probability of spike observations given stimulus.

    Args:
        model: Neural network model (the .net attribute of SpikeModel)
        s: Stimulus tensor (seq_len, batch_size)
        x: Spike count tensor (seq_len, batch_size, n_neurons)
        cache: Optional cache tensor for the model

    Returns:
        Tuple of (log probabilities (seq_len, batch_size), updated cache)
    """
    model.eval()
    log_intensity, cache = model(s, x, cache)
    return (
        -F.poisson_nll_loss(
            log_intensity, x, log_input=True, full=True, reduction="none"
        ).sum(-1),
        cache,
    )


def resample(arr, indices):
    """
    Resamples arr along dimension 1 according to indices.

    This function is equivalent to:

    ```
    for i in range(len(batch_indices)):
        for j in range(arr.shape[1]):
            out[i, j] = arr[i, indices[i, j]]
    ```
    """
    extra_dims = arr.ndim - 2

    # Add extra singleton dimensions to indices and expand to match arr's shape
    # E.g., indices[B, N] -> indices[B, N, 1, 1, 1] -> indices[B, N, C, H, W]
    ii = indices[(Ellipsis,) + (None,) * extra_dims].expand(-1, -1, *arr.shape[2:])
    return torch.gather(arr, dim=1, index=ii)


class ParticleFilter(nn.Module):
    """
    A bootstrap particle filter for marginal likelihood estimation.

    Args:
        model: Neural network model for computing likelihoods
        x: Observation tensor (seq_len, batch_size, n_neurons)
        n_particles: Number of particles to use
        block_size: Number of time steps to process per forward pass
        resample_criterion: ESS threshold for resampling (default: n_particles/2)
        device: Device to run computations on
    """

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        n_particles: int = 128,
        block_size: int = 5,
        resample_criterion=None,
        device=None,
        bin_width: float = 0.01
    ):
        super().__init__()
        seq_len, batch_size, _ = x.shape
        self.device = device if device is not None else x.device
        self.batch_size = batch_size
        self.block_size = block_size
        self.model = model
        self.bin_width = bin_width
        self.sho = StochasticHarmonicOscillator().to(self.device)
        self.n = 0  # discrete time
        self.y0 = torch.randn(batch_size, n_particles, 2, device=self.device) * get_stationary_sigma(self.device)
        self.x = x.repeat_interleave(
            n_particles, dim=1
        )  # (seq_len, batch_size * n_particles)
        self.n_particles = n_particles
        self.log_marginals = []
        self.cache = None
        self.weights = torch.zeros(batch_size, n_particles, device=self.device) - np.log(n_particles)
        self.resample_criterion = (
            n_particles / 2 if resample_criterion is None else resample_criterion
        )
        self.resamples = []

    @torch.no_grad()
    def forward(self):
        """
        The main particle filter routine.

        Advance the particles to `n_end`. This propagates all particles forward in time
        to `n_end * delta_t` where delta_t is the discretization. Computes the
        likelihood-increment for all particles.

        Then whenever the effective sample size (ESS) drops below half the particle
        count, we resample.
        """
        n = self.n
        n_end = n + self.block_size

        # STEP 1: Propagate input
        n_range = torch.arange(n, n_end + 1, device=self.device)
        t_grid = n_range * self.bin_width
        y = torchsde.sdeint(self.sho, self.y0.flatten(0, 1), t_grid, dt=1 / 100)
        assert isinstance(y, torch.Tensor)

        y = y.unflatten(
            1, (self.batch_size, self.n_particles)
        )  # (n_end - n + 1, B, P, 2)
        self.y0 = y[-1, ...]
        s = y[:-1, :, :, 0]  # (n_end - n, B, P)

        self.n = n_end

        # STEP 2: Compute likelihood
        s_flat = s.flatten(1, 2)  # (n_end - n, B * P)

        # note that x_flat is shifted by one time step wrt s_flat, which is correct
        # since the model predicts x_n given s_1:n and x_1:n-1
        if n == 0:
            x_flat = torch.cat(
                [torch.zeros_like(self.x[0]).unsqueeze(0), self.x[0 : n_end - 1]]
            )  # (n_end - n, B * P)
        else:
            x_flat = self.x[n - 1 : n_end - 1]  # (n_end - n, B * P)

        logp_increment, self.cache = log_probability(
            self.model, s_flat, x_flat, self.cache
        )
        logp_increment = logp_increment.cumsum(0).unflatten(
            1, (self.batch_size, self.n_particles)
        )
        self.log_marginals.append(
            torch.logsumexp(self.weights + logp_increment, dim=-1)
        )

        # increase particle weights and compute ESS
        self.weights += logp_increment[-1]
        ess = 1 / torch.sum(F.softmax(self.weights, dim=1) ** 2, dim=1)

        # STEP 3: Resample particle state (cache and y0)
        batch_indices = torch.where(ess < self.resample_criterion)[0]
        if len(batch_indices) == 0:
            # skip resampling
            self.resamples.append((n_end, 0))
            return

        dist = torch.distributions.categorical.Categorical(
            logits=self.weights[batch_indices]
        )
        indices = dist.sample(
            (self.n_particles,)
        ).T  # (len(batch_indices), n_particles)

        # resampling
        state = self.cache[:, 0].unflatten(0, (self.batch_size, self.n_particles))
        state[batch_indices] = resample(state[batch_indices], indices)
        self.y0[batch_indices] = resample(self.y0[batch_indices], indices)

        # reset the weights of all resampled particles to the current log_marginal estimate
        current_log_marginal_est = self.log_marginals[-1][-1]
        self.weights[batch_indices] = \
            current_log_marginal_est[batch_indices].unsqueeze(-1) - np.log(self.n_particles)
        self.resamples.append((n_end, len(batch_indices)))


def smc_marginal_estimate(
    model,
    x,
    num_particles=256,
    block_size=5,
    batch_size=8,
    device=None,
    bin_width=0.01,
):
    """Estimate marginal likelihood using Sequential Monte Carlo.

    Args:
        model: Neural network model for computing likelihoods
        x: Observation tensor (seq_len, total_size, n_neurons)
        num_particles: Number of particles for SMC
        block_size: Number of time steps per particle filter forward pass
        batch_size: Batch size for processing observations
        device: Device to run computations on
        bin_width: Time bin width in seconds

    Returns:
        Tuple of (log_marginal estimates, average resamples per batch)
    """
    n_max, total_size, _ = x.shape
    device = device if device is not None else x.device

    filters = []
    for i in range(0, total_size, batch_size):
        pf = ParticleFilter(
            model,
            x[:, i:i+batch_size],
            num_particles,
            block_size=block_size,
            device=device,
            bin_width=bin_width,
        )
        for _ in range(n_max // block_size):
            pf()
        filters.append(pf)
    log_marginal = torch.cat([torch.cat(pf.log_marginals) for pf in filters], dim=1)
    resamples = torch.stack([torch.tensor(pf.resamples, device=device) for pf in filters])
    return log_marginal, resamples[:, :, 1].sum(0) / total_size  # average resamples over batches


def pws_estimate(
    model,
    t,
    N,
    M,
    device=None,
    bin_width=1.0,
    batch_size=64,
):
    """Perform PWS estimation of mutual information.

    Args:
        model: Trained SpikeModel instance
        t: Time grid tensor (seq_len,)
        N: Number of trajectories to sample
        M: Number of particles for marginal estimation
        device: Device to run computations on (default: infer from model)
        bin_width: Time bin width in seconds

    Returns:
        Tuple of:
            - Tensor of shape (seq_len, 2, N) containing log_conditional and log_marginal
            - Resampling statistics
    """
    if device is None:
        device = next(model.parameters()).device

    t = t.to(device)
    stoch_osc = StochasticHarmonicOscillator().to(device)
    # Initialize from stationary distribution
    y0 = torch.randn(N, 2, device=device) * get_stationary_sigma(device)
    y = torchsde.sdeint(stoch_osc, y0, t, dt=1 / 60)
    assert isinstance(y, torch.Tensor)

    s_test = y[:, :, 0]
    x_test = model.net.sample(s_test)

    log_conditional, _ = log_probability(model.net, s_test, shift_right(x_test))
    log_conditional = log_conditional.cumsum(0)

    log_marginal, resamples = smc_marginal_estimate(
        model.net,
        x_test,
        num_particles=M,
        device=device,
        bin_width=bin_width,
        batch_size=batch_size,
    )
    return torch.stack([log_conditional, log_marginal], dim=1), resamples  # (seq_len, 2, N)
