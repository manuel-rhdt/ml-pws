import jax
import jax.numpy as jnp
from flax import nnx

class ARModel(nnx.Module):
    """
    A JAX-based Autoregressive AR(p) model.
    """

    def __init__(self, coefficients: jax.Array, noise_std: float):
        """
        Initializes the AR(p) model.

        Args:
            coefficients (jnp.ndarray): A 1D JAX array of shape (p,)
                                        containing the AR coefficients
                                        (phi_1, phi_2, ..., phi_p).
            noise_std (float): The standard deviation (sigma) of the
                               Gaussian white noise.
        """
        self.coeffs = jnp.asarray(coefficients)
        self.noise_std = jnp.asarray(noise_std)
        self.p = len(coefficients)

    def __call__(self, s: jax.Array):
        """
        Computes log P(s) for the AR(p) model.
        """
        coeffs = jnp.flip(self.coeffs)
        noise_std = self.noise_std

        def step_logp(carry, s_t):
            predicted_mean = jnp.dot(coeffs, carry)
            log_prob = (
                -0.5 * jnp.log(2 * jnp.pi * noise_std**2)
                - 0.5 * ((s_t - predicted_mean) / noise_std) ** 2
            )

            # Update the carry: shift old values and add the current s_t
            new_carry = jnp.roll(carry, -1)  # Shift all elements to the left
            new_carry = new_carry.at[-1].set(s_t)  # Set the last element to s_t

            return new_carry, log_prob

        def computation(s):
            carry, result = jax.lax.scan(step_logp, jnp.zeros(self.p), s)
            return result

        if s.ndim == 1:
            # If s is a 1D array, we can directly compute the log probabilities
            return computation(s)
        elif s.ndim == 2:
            # If s is a 2D array, we need to apply the computation across the first axis
            return jax.vmap(computation)(s)
        else:
            raise ValueError("Input array s must be 1D or 2D.")