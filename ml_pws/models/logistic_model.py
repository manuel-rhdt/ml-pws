from dataclasses import dataclass
from typing import Optional

import jax
from jax import random
import jax.numpy as jnp
from flax import nnx


@dataclass
class LogisticModel(nnx.Module):
    gain: float
    decay: float
    noise: float
    rngs: Optional[nnx.Rngs] = None

    def __call__(self, s, x, generate=False, rngs=None):
        if s.ndim > 2:
            raise ValueError("s has more than 2 dimensions")
        if x.ndim > 2:
            raise ValueError("x has more than 2 dimensions")
        s = jnp.reshape(s, (-1, s.shape[-1]))
        x = jnp.reshape(x, (-1, x.shape[-1]))

        rngs = self.rngs if rngs is None else rngs
        key = rngs["generate"]() if generate else None
        x_prev = jnp.zeros(x.shape[:1])

        def step(state, val):
            key, x_prev = state
            s_cur, x_cur = val
            bias = jax.nn.sigmoid(s_cur * self.gain)
            if generate:
                next_key, key = random.split(key)
                x_cur = (
                    bias
                    + self.decay * x_prev
                    + self.noise * jax.random.normal(key, x_cur.shape)
                )
            else:
                next_key = None
            logp = jax.scipy.stats.norm.logpdf(
                x_cur - self.decay * x_prev - bias, scale=self.noise
            )
            return (next_key, x_cur), (logp, x_cur)

        _, (logp, x) = jax.lax.scan(step, (key, x_prev), (s.T, x.T))

        if generate:
            return logp.T, x.T
        else:
            return logp.T

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
