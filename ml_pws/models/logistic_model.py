from dataclasses import dataclass
from typing import Optional

import jax
from jax import random
import jax.numpy as jnp
from flax import nnx


@dataclass
class LogisticCell(nnx.Module):
    gain: float
    decay: float
    noise: float

    def __call__(self, state, s: jax.Array, x: jax.Array, *, generate=False):
        key, x_prev = state

        bias = jax.nn.sigmoid(s * self.gain)
        if generate:
            next_key, key = random.split(key)
            x = (
                bias
                + self.decay * x_prev
                + self.noise * jax.random.normal(key, x.shape)
            )
            logp = jax.scipy.stats.norm.logpdf(
                x - self.decay * x_prev - bias, scale=self.noise
            )
            return (next_key, x), (logp, x)
        else:
            next_key = None
            logp = jax.scipy.stats.norm.logpdf(
                x - self.decay * x_prev - bias, scale=self.noise
            )
            return (next_key, x), logp

    def initialize_carry(self, shape, *, key=None):
        x_prev = jnp.zeros(shape[:1])
        return (key, x_prev)


class LogisticModel(nnx.Module):
    gain: float
    decay: float
    noise: float
    rngs: Optional[nnx.Rngs]

    def __init__(self, gain, decay, noise, *, rngs=None):
        self.cell = LogisticCell(gain, decay, noise)
        self.rngs = rngs

    def __call__(self, s, x, generate=False, rngs=None):
        if s.ndim > 2:
            raise ValueError("s has more than 2 dimensions")
        if x.ndim > 2:
            raise ValueError("x has more than 2 dimensions")
        s = jnp.reshape(s, (-1, s.shape[-1]))
        x = jnp.reshape(x, (-1, x.shape[-1]))

        rngs = self.rngs if rngs is None else rngs
        key = rngs["generate"]() if generate else None

        state = self.cell.initialize_carry(x.shape, key=key)

        @nnx.scan(in_axes=(nnx.Carry, 1, 1), out_axes=(nnx.Carry, 1))
        def scan_fn(state, s, x):
            return self.cell(state, s, x, generate=generate)
        _, result = scan_fn(state, s, x)
        return result

    def log_prob(self, s, x, full=False):
        def step(carry, val):
            logp, xprev = carry
            s, x = val
            delta = jax.scipy.stats.norm.logpdf(
                x - self.decay * xprev - jax.nn.sigmoid(s * self.gain), scale=self.noise
            )
            return (logp + delta, x), (logp + delta) if full else None

        (logp, _), logp_full = jax.lax.scan(step, (0.0, 0.0), [s, x])
        return logp_full if full else logp
