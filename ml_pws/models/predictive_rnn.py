import jax
from jax import random
import jax.numpy as jnp
from flax import nnx
from flax.nnx.transforms import iteration


def shift_right(x, axis=1):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    ind = [slice(None)] * len(x.shape)
    ind[axis] = slice(-1)
    return jnp.pad(x, pad_widths)[tuple(ind)]


class PredictiveCell(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.cell = nnx.GRUCell(2, hidden_size, rngs=rngs)
        self.dense = nnx.Linear(hidden_size, 2, rngs=rngs)

    def __call__(self, carry, s, x, generate=True):
        cell_state, prev_prediction = carry
        if generate:
            x = jnp.expand_dims(prev_prediction, -1)

        cell_state, y = self.cell(cell_state, jnp.concat((s, x), axis=-1))
        y = self.dense(y)
        mean, log_var = (jnp.squeeze(arr, -1) for arr in jnp.split(y, 2, axis=-1))

        if not generate:
            prediction = None
        else:
            rng = self.rngs["generate"]
            prediction = mean + random.normal(rng(), shape=log_var.shape) * jnp.exp(
                log_var / 2
            )
        return (cell_state, prediction), ((mean, log_var), prediction)

    def initial_state(self, batch_size: int, generate=True):
        cell_state = self.cell.initialize_carry((batch_size, 2), rngs=self.rngs)
        initial_prediction = jnp.zeros(batch_size) if generate else None
        return (cell_state, initial_prediction)

    @property
    def num_feature_axes(self):
        return 1


class PredictiveRnn(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.cell = PredictiveCell(hidden_size, rngs)

    def __call__(self, s, x, generate=False):
        s = jnp.expand_dims(s, -1)  # (batch, time, features=1)
        x = jnp.expand_dims(x, -1)  # (batch, time, features=1)

        if not generate:
            target = jnp.squeeze(x, -1)
            x = shift_right(x, -2)

        s, x = jnp.broadcast_arrays(s, x)

        scan_fn = lambda cell, carry, s, x: cell(carry, s, x, generate)
        carry = self.cell.initial_state(x.shape[0], generate)

        time_axis = -2
        state_axes = iteration.StateAxes({...: iteration.Carry})

        carry, ((mean, log_var), preds) = nnx.scan(
            scan_fn,
            in_axes=(state_axes, iteration.Carry, time_axis, time_axis),
            out_axes=(iteration.Carry, -1),
        )(self.cell, carry, s, x)

        if generate:
            target = preds

        logp = jax.scipy.stats.norm.logpdf(target, loc=mean, scale=jnp.exp(log_var / 2))
        if generate:
            return logp, preds
        else:
            return logp


class ConvolutionalAutoregressiveModel(nnx.Module):
    def __init__(
        self, kernel_size: int, hidden_features: int, num_layers: int, rngs: nnx.Rngs
    ):
        self.layers = [
            nnx.Conv(
                in_features=(2 if i == 0 else hidden_features),
                out_features=hidden_features,
                kernel_size=kernel_size,
                padding="CAUSAL",
                rngs=rngs,
            )
            for i in range(num_layers)
        ]
        self.loc = nnx.Linear(in_features=hidden_features, out_features=1, rngs=rngs)
        self.log_scale = nnx.Linear(
            in_features=hidden_features, out_features=1, rngs=rngs
        )
        self.rngs = rngs

    def forward(self, s, x):
        z = jnp.concat((s, x), axis=-1)
        for conv in self.layers:
            z = jax.nn.relu(conv(z))
        loc = self.loc(z)
        log_scale = self.log_scale(z)
        return loc, log_scale

    def __call__(self, s, x, generate=False):
        s = jnp.expand_dims(s, -1)

        if generate:
            x = jnp.zeros(x.shape + (1,))
        else:
            x = jnp.expand_dims(x, -1)
            target = x
            x = shift_right(x, -2)

        s, x = jnp.broadcast_arrays(s, x)

        def step(pred, mod, i):
            loc, log_scale = mod.forward(s, shift_right(pred, -2))
            key = mod.rngs()
            loc = loc[..., i, :]
            log_scale = log_scale[..., i, :]
            eps = loc + jax.random.normal(key, loc.shape) * jnp.exp(log_scale)
            pred = pred.at[..., i, :].set(eps)
            return pred, (loc, log_scale)

        if generate:
            pred, (loc, log_scale) = nnx.scan(step, state_axes={}, out_axes=-2)(
                x, self, jnp.arange(x.shape[-2])
            )
            target = pred
        else:
            loc, log_scale = self.forward(s, x)

        logp = jnp.squeeze(
            jax.scipy.stats.norm.logpdf(target, loc, jnp.exp(log_scale)), axis=-1
        )

        if generate:
            return logp, jnp.squeeze(pred, -1)
        else:
            return logp
