import torch
import torch.nn as nn

import numpy as np

from scipy.special import expit

def generate_stable_ar_coefficients(n, seed=0):
    """
    Generates stable AR(n) coefficients.

    Parameters:
        n (int): Order of the AR process.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Stable AR coefficients of shape (n,).
    """
    rng = np.random.default_rng(seed)

    # Generate random roots outside the unit circle
    min_magnitude = 1.1
    max_magnitude = 2.0
    roots = []
    while len(roots) < n:
        # Decide whether to add a real root or a complex conjugate pair
        remaining = n - len(roots)
        if remaining >= 2:
            # Randomly choose to add a real root or a complex pair
            add_complex = rng.random() > 0.5
        else:
            add_complex = False  # Only enough space for a real root

        if add_complex and remaining >= 2:
            # Generate a complex conjugate pair
            magnitude = rng.uniform(min_magnitude, max_magnitude)
            angle = rng.uniform(0, 2 * np.pi)
            root = magnitude * np.exp(1j * angle)
            roots.append(root)
            roots.append(np.conj(root))
        else:
            # Generate a real root
            magnitude = rng.uniform(min_magnitude, max_magnitude)
            # Randomly choose positive or negative real root
            sign = rng.choice([-1, 1])
            root = sign * magnitude
            roots.append(root)

    # Truncate to 'n' roots
    roots = np.array(roots[:n])

    # Convert roots to polynomial coefficients
    # c[0] + c[1]*x + c[2] * x**2 + ...
    coefficients = np.poly(roots).real[
        ::-1
    ]  # Convert roots to coefficients, only real part needed

    # Normalize to make the leading coefficient 1
    coefficients /= coefficients[0]

    # AR coefficients are the negatives of the remaining coefficients
    return -coefficients[1:]


class ARModel(nn.Module):
    def __init__(self, n, noise_std=1.0, init_coeffs=generate_stable_ar_coefficients):
        """
        Autoregressive (AR) model of specified order.

        Parameters:
            order (int): Order of the AR process (n in AR(n)).
        """
        super(ARModel, self).__init__()
        self.n = n

        if callable(init_coeffs):
            coeffs = init_coeffs(n)
        else:
            coeffs = init_coeffs

        # Trainable coefficients
        self.coefficients = nn.Parameter(torch.tensor(coeffs).flip(0))
        self.log_noise_std = nn.Parameter(torch.log(torch.tensor(noise_std)))

    @torch.no_grad()
    def sample(self, initial_values, steps, seed = None):
        """
        Generate a sequence from the AR process.

        Parameters:
            initial_values (tensor): Initial values of shape (batch_size, n).
            steps (int): Number of steps to generate.

        Returns:
            tensor: Generated AR sequences of shape (batch_size, steps).
        """
        generator = torch.Generator(initial_values.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        batch_size = initial_values.shape[0]
        sequence = torch.zeros(batch_size, steps + self.n, device=initial_values.device)
        sequence[:, : self.n] = initial_values

        noise_std = torch.exp(self.log_noise_std)
        for t in range(self.n, steps + self.n):
            # Compute AR value
            ar_value = torch.sum(sequence[:, t - self.n : t] * self.coefficients, dim=1)
            # Add noise
            noise = torch.randn(batch_size, generator=generator) * noise_std
            sequence[:, t] = ar_value + noise

        return sequence[:, self.n:]


# OUTPUT MODEL

from dataclasses import dataclass


@dataclass
class LogisticModel:
    gain: float
    decay: float
    noise: float

    def __call__(self, s, x):
        bias = expit(self.gain * s)
        x_prev = np.zeros_like(x)
        x_prev[:, 1:] = x[:, :-1]
        dx = x - self.decay * x_prev - bias
        return -0.5 * (dx / self.noise) ** 2 + np.log(2 * np.pi * self.noise**2)

    def sample(self, s, seed=None):
        if s.ndim > 2:
            raise ValueError("s has more than 2 dimensions")

        # Shape (Batch, Time)
        s = np.reshape(s, (-1, s.shape[-1]))

        rng = np.random.default_rng(seed=seed)

        x_prev = np.zeros(s.shape[:1])
        x = np.zeros_like(s)
        logistic_s = expit(self.gain * s)
        for t in range(s.shape[-1]):
            bias = logistic_s[:, t]
            x[:, t] = (
                bias + self.decay * x_prev + self.noise * rng.normal(size=x_prev.shape)
            )
            x_prev = x[:, t]

        return x


def generate_nonlinear_data(
    num_pairs=1000, length=10, seed=0, order=3, coeffs=generate_stable_ar_coefficients, ar_noise=0.2, gain=10.0, decay=0.2, noise=0.2
):
    n = order
    ar_model = ARModel(n, noise_std=ar_noise, init_coeffs=coeffs)

    initial_values = torch.zeros(num_pairs, n)
    s_trajs = ar_model.sample(initial_values, length, seed=seed+1)

    x_model = LogisticModel(gain, decay, noise)
    x_trajs = x_model.sample(s_trajs.numpy(), seed=seed + 2)

    return s_trajs, torch.tensor(x_trajs)
