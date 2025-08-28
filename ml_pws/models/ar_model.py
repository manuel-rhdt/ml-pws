import jax
import jax.numpy as jnp
from jax import random
from flax import nnx


class ARCell(nnx.Module):
    """
    A JAX-based Autoregressive AR(p) cell.
    """
    def __init__(self, coefficients: jax.Array, noise_std: float, *, rngs: nnx.Rngs | None = None):
        """
        Initializes the AR(p) model.

        Args:
            coefficients (jnp.ndarray): A 1D JAX array of shape (p,)
                                        containing the AR coefficients
                                        (phi_1, phi_2, ..., phi_p).
            nsoise_std (float): The standard deviation (sigma) of the
                               Gaussian white noise.
        """
        self.coeffs = jnp.flip(jnp.asarray(coefficients))
        self.noise_std = jnp.asarray(noise_std)
        self.p = len(coefficients)
        self.rngs = rngs

    def __call__(self, carry: jax.Array, s_t: jax.Array, *, generate: bool = False):
        
        # coefficients are already flipped
        predicted_mean = jnp.dot(carry, self.coeffs)

        if generate:
            s_t = predicted_mean + self.noise_std * random.normal(self.rngs(), s_t.shape)

        residual = s_t - predicted_mean

        log_prob = jax.scipy.stats.norm.logpdf(residual, scale=self.noise_std)

        # Update the carry: shift old values and add the current s_t
        new_carry = jnp.roll(carry, -1, axis=-1)  # Shift all elements to the left
        new_carry = new_carry.at[:, -1].set(s_t)  # Set the last element to s_t

        if generate:
            return new_carry, (log_prob, s_t)
        else:
            return new_carry, log_prob

    def initialize_carry(self, shape: tuple[int, int]):
        batch_size, _ = shape
        return jnp.zeros((batch_size, self.p))


    

class ARModel(nnx.Module):
    """
    A JAX-based Autoregressive AR(p) model.
    """

    def __init__(self, coefficients: jax.Array, noise_std: float, *, rngs: nnx.Rngs | None = None):
        """
        Initializes the AR(p) model.

        Args:
            coefficients (jnp.ndarray): A 1D JAX array of shape (p,)
                                        containing the AR coefficients
                                        (phi_1, phi_2, ..., phi_p).
            noise_std (float): The standard deviation (sigma) of the
                               Gaussian white noise.
        """
        self.cell = ARCell(coefficients, noise_std, rngs=rngs)

    def __call__(self, s: jax.Array, *, generate: bool = False):

        carry = self.cell.initialize_carry(s.shape)

        state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

        @nnx.split_rngs(splits=s.shape[-1])
        @nnx.scan(in_axes=(state_axes, nnx.Carry, 1), out_axes=(nnx.Carry, 1))
        def scan_fn(cell, carry, s_t):
            return cell(carry, s_t, generate=generate)

        _, result = scan_fn(self.cell, carry, s)

        return result
