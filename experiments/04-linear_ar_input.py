import jax
import jax.numpy as jnp
from jax import random

import numpy as np
from flax import nnx
import optax
from clu import metrics, metric_writers
import polars as pl

from absl import logging

logging.set_verbosity(logging.INFO)

import os
import dataclasses
import argparse
from pathlib import Path
from typing import Optional

from ml_pws.data.nonlinear_dataset import generate_nonlinear_data

def parse_args():
    parser = argparse.ArgumentParser(description="Comparison of ML-PWS against DoE and InfoNCE for a linear AR model.")

    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        required=True,
        help="List AR coefficients for the input model (e.g., 0.5 0.3 -0.2)",
    )

    parser.add_argument(
        "-o",
        "--output_filename",
        type=Path,
        required=True,
        help="Path to the output JSON file",
    )

    return parser.parse_args()

class ARModel:
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
        coeffs = self.coeffs
        noise_std = self.noise_std

        def step_logp(carry, s_t):
            predicted_mean = jnp.dot(coeffs, jnp.flip(carry))
            log_prob = -0.5 * jnp.log(2 * jnp.pi * noise_std**2) - 0.5 * ((s_t - predicted_mean) / noise_std)**2

            # Update the carry: shift old values and add the current s_t
            new_carry = jnp.roll(carry, -1) # Shift all elements to the left
            new_carry = jax.ops.index_update(new_carry, jax.ops.index[-1], s_t) # Update the last element

            return new_carry, log_prob

        carry, result = jax.lax.scan(step_logp, jnp.zeros(self.p), s)
        return result


@dataclasses.dataclass
class LogisticModel(nnx.Module):
    gain: float
    decay: float
    noise: float
    rngs: Optional[nnx.Rngs] = None
    
    def __call__(self, s, x, generate=False, rngs = None):
        if s.ndim > 2:
            raise ValueError("s has more than 2 dimensions")
        if x.ndim > 2:
            raise ValueError("x has more than 2 dimensions")
        s = jnp.reshape(s, (-1, s.shape[-1]))
        x = jnp.reshape(x, (-1, x.shape[-1]))

        rngs = self.rngs if rngs is None else rngs
        key = rngs['generate']() if generate else None
        x_prev = jnp.zeros(x.shape[:1])
        logp = jnp.zeros(x.shape[:1]) + jnp.zeros(s.shape[:1]) # broadcasting
        
        def step(state, val):
            key, logp, x_prev = state
            s_cur, x_cur = val
            bias = jax.nn.sigmoid(s_cur * self.gain)
            if generate:
                next_key, key = random.split(key)
                x_cur = bias + self.decay * x_prev + self.noise * jax.random.normal(key, x_cur.shape)
            else:
                next_key = None
            logp += jax.scipy.stats.norm.logpdf(x_cur - self.decay * x_prev - bias, scale=self.noise)
            return (next_key, logp, x_cur), x_cur
    
        (_, logp, _), x = jax.lax.scan(step, (key, logp, x_prev), (s.T, x.T))

        if generate:
            return logp, x.T
        else:
            return logp

    def log_prob(self, s, x, full=False):
        def step(carry, val):
            logp, xprev = carry
            s, x = val
            delta = jax.scipy.stats.norm.logpdf(x - self.decay * xprev - jax.nn.sigmoid(s * self.gain), scale=self.noise)
            return (logp + delta, x), (logp+delta) if full else None
    
        (logp, _), logp_full = jax.lax.scan(step, (0.0, 0.0), [s, x])
        return logp_full if full else logp

def shift_right(x, axis=1):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    ind = [slice(None)] * len(x.shape)
    ind[axis] = slice(-1)
    return jnp.pad(x, pad_widths)[tuple(ind)]

class GRUCell(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        rngs: nnx.Rngs,
        kernel_init=nnx.initializers.lecun_normal(),
        recurrent_kernel_init=nnx.initializers.orthogonal(),
    ):
        self.hidden_size = hidden_size
        self.linear_ir = nnx.Linear(
            in_features, hidden_size, use_bias=True, kernel_init=kernel_init, rngs=rngs
        )
        self.linear_iz = nnx.Linear(
            in_features, hidden_size, use_bias=True, kernel_init=kernel_init, rngs=rngs
        )
        self.linear_in = nnx.Linear(
            in_features, hidden_size, use_bias=True, kernel_init=kernel_init, rngs=rngs
        )
        self.linear_hr = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=False,
            kernel_init=recurrent_kernel_init,
            rngs=rngs,
        )
        self.linear_hz = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=False,
            kernel_init=recurrent_kernel_init,
            rngs=rngs,
        )
        self.linear_hn = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            kernel_init=recurrent_kernel_init,
            rngs=rngs,
        )

    def __call__(self, h: jax.Array, x: jax.Array):
        r = jax.nn.sigmoid(self.linear_ir(x) + self.linear_hr(h))
        z = jax.nn.sigmoid(self.linear_iz(x) + self.linear_hz(h))
        n = jax.nn.tanh(self.linear_in(x) + r * self.linear_hn(h))
        h = (1.0 - z) * n + z * h
        return h, h

    def initial_state(self, batch_size: int):
        return jnp.zeros((batch_size, self.hidden_size))


class RNN(nnx.Module):
    def __init__(self, cell, reverse=False):
        self.reverse = reverse
        self.cell = cell

    def __call__(self, x: jax.Array, *args, return_carry=False):
        def scan_fn(carry: jax.Array, cell: GRUCell, x: jax.Array, *args):
            return cell(carry, x, *args)

        carry = self.cell.initial_state(x.shape[0])
        carry, y = nnx.scan(
            scan_fn, state_axes={}, in_axes=1, out_axes=1, reverse=self.reverse
        )(carry, self.cell, x, *args)

        if return_carry:
            return carry, y
        else:
            return y


def shift_right(x: jax.Array, axis=1):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    ind = [slice(None)] * len(x.shape)
    ind[axis] = slice(-1)
    return jnp.pad(x, pad_widths)[tuple(ind)]


def logistic_logpdf(x, mean, log_scale):
    z = (x - mean) * jnp.exp(-log_scale)
    z_half = z / 2
    return -2 * jnp.logaddexp(z_half, -z_half) - log_scale


def logistic_logcdf(x, mean, log_scale):
    z = (x - mean) * jnp.exp(-log_scale)
    return jax.nn.log_sigmoid(z)


def mixlogistic_logpdf(x, prior_logits, means, log_scales):
    x = jnp.expand_dims(x, -1)
    log_pi = jax.nn.log_softmax(prior_logits, axis=-1)
    return jax.nn.logsumexp(log_pi + logistic_logpdf(x, means, log_scales), axis=-1)


def mixlogistic_logcdf(x, prior_logits, means, log_scales):
    x = jnp.expand_dims(x, -1)
    log_pi = jax.nn.log_softmax(prior_logits, axis=-1)
    return jax.nn.logsumexp(log_pi + logistic_logcdf(x, means, log_scales), axis=-1)


# coupling function from flow++
def coupling_function(x: jax.Array, pi: jax.Array, mu: jax.Array, log_s: jax.Array):
    # clipping is needed for numerical stability
    log_cdf = jnp.clip(mixlogistic_logcdf(x, pi, mu, log_s), a_max=-1e-12)
    # inverse sigmoid, i.e. logit(exp(log_cdf))
    val = log_cdf - jnp.log(-jnp.expm1(log_cdf))

    log_jac = mixlogistic_logpdf(x, pi, mu, log_s)
    # manually evaluate derivative log(1/(exp(log_cdf)*(1 - exp(log_cdf))))
    log_jac += -log_cdf - jnp.log(-jnp.expm1(log_cdf))
    return val, log_jac


class DecoderCell(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        rngs: nnx.Rngs,
        mixture_components: int = 5,
    ):
        self.cell = GRUCell(in_features, hidden_size, rngs=rngs)
        self.shift = nnx.Linear(hidden_size, 1, rngs=rngs)
        self.log_scale = nnx.Linear(hidden_size, 1, rngs=rngs)
        self.pi = nnx.Linear(hidden_size, mixture_components, rngs=rngs)
        self.mu = nnx.Linear(hidden_size, mixture_components, rngs=rngs)
        self.log_s = nnx.Linear(hidden_size, mixture_components, rngs=rngs)

    def __call__(self, carry, input, rngs: nnx.Rngs):
        rnn_state, last_prediction, logp = carry
        input = input.at[..., 0].set(last_prediction)

        rnn_state, y = self.cell(rnn_state, input)

        shift = jnp.squeeze(self.shift(y), -1)
        log_scale = jnp.squeeze(self.log_scale(y), -1)
        pi = self.pi(y)
        mu = self.mu(y)
        log_s = self.log_s(y)

        epsilon = random.logistic(rngs(), shape=y.shape[:-1])

        val, log_jac = coupling_function(epsilon, pi, mu, log_s)
        prediction = shift + val * jnp.exp(log_scale)

        logp += jax.scipy.stats.logistic.logpdf(epsilon) - log_scale - log_jac

        return (rnn_state, prediction, logp), prediction

    def initial_state(self, batch_size: int):
        cell_state = self.cell.initial_state(batch_size)
        initial_prediction = jnp.zeros(batch_size)
        logp = jnp.zeros(batch_size)
        return (cell_state, initial_prediction, logp)


class VariationalRnn(nnx.Module):
    def __init__(self, hidden_size: int, mixture_components: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.encoder_x = RNN(GRUCell(1, hidden_size, rngs=rngs), reverse=True)
        self.decoder_rnn = RNN(
            DecoderCell(
                1 + hidden_size,
                hidden_size,
                mixture_components=mixture_components,
                rngs=rngs,
            )
        )

    def __call__(self, s: jax.Array, x: jax.Array, rngs=None):
        # s, x: Make input tensors of shape [batch_size, seq_length, 1]
        s = s.reshape((-1, s.shape[-1]) + (1,))
        x = x.reshape((-1, x.shape[-1]) + (1,))

        # apply reverse rnn to x
        h_x = self.encoder_x(x)

        rngs = self.rngs if rngs is None else rngs

        h_x = jnp.broadcast_to(h_x, s.shape[:-1] + (h_x.shape[-1],))
        s_with_context = jnp.concatenate((s, h_x), axis=-1)
        (_, _, logp), preds = self.decoder_rnn(s_with_context, rngs, return_carry=True)

        return logp, preds


class IAFBlock(nnx.Module):
    def __init__(self, kernel_size: int, hidden_features: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=1,
            out_features=hidden_features,
            kernel_size=kernel_size,
            padding="CAUSAL",
            rngs=rngs,
        )

        self.linear1 = nnx.Linear(
            in_features=hidden_features, out_features=hidden_features, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=hidden_features,
            out_features=hidden_features,
            use_bias=False,
            rngs=rngs,
        )

        self.mu = nnx.Linear(in_features=hidden_features, out_features=1, rngs=rngs)
        self.s = nnx.Linear(in_features=hidden_features, out_features=1, rngs=rngs)
        self.rngs = rngs

    def __call__(self, carry, context):
        logp, z_prev = carry
        x = self.conv(shift_right(z_prev, -2))
        x = nnx.relu(self.linear1(x) + self.linear2(context))
        mu = self.mu(x)
        s = self.s(x)
        sigma = jax.nn.sigmoid(s)
        z = sigma * z_prev + (1 - sigma) * mu
        logp = logp - jnp.sum(jax.nn.log_sigmoid(s), axis=-2)
        return (logp, z), None


class IAF(nnx.Module):
    def __init__(
        self, kernel_size: int, hidden_features: int, depth: int, rngs: nnx.Rngs
    ):
        self.rngs = rngs
        self.encoder_x = RNN(GRUCell(1, hidden_features, rngs=rngs), reverse=True)
        self.initial_loc = nnx.Linear(
            in_features=hidden_features, out_features=1, rngs=rngs
        )
        self.initial_log_scale = nnx.Linear(
            in_features=hidden_features, out_features=1, rngs=rngs
        )
        self.decoder = nnx.Scan.constructor(IAFBlock, in_axes=None, length=depth)(
            kernel_size, hidden_features, rngs=rngs
        )

    def __call__(self, s, x, rngs=None):
        # s, x: Make input tensors of shape [batch_size, seq_length, 1]
        s = s.reshape((-1, s.shape[-1]) + (1,))
        x = x.reshape((-1, x.shape[-1]) + (1,))

        rngs = self.rngs if rngs is None else rngs

        # apply reverse rnn to x
        h_x = self.encoder_x(x)
        mu = self.initial_loc(h_x)
        log_sigma = self.initial_log_scale(h_x)

        eps = random.normal(rngs(), s.shape)
        z = jnp.exp(log_sigma) * eps + mu
        logp = -jnp.sum(log_sigma + 0.5 * eps**2 + 0.5 * jnp.log(2 * jnp.pi), axis=-2)

        h_x = jnp.broadcast_to(h_x, s.shape[:-1] + (h_x.shape[-1],))
        result, _ = self.decoder((logp, z), h_x)

        return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, -1), result)


@dataclasses.dataclass
class CombinedModel(nnx.Module):
    prior: nnx.Module
    forward: nnx.Module
    backward: nnx.Module

    def importance_weight(self, x):
        s = jnp.empty_like(x)
        logq_s, preds = self.backward(s, x)
        logp_s = self.prior(preds)
        logp_x_given_s = self.forward(preds, x)
        return logp_s + logp_x_given_s - logq_s

    def elbo(self, x, num_samples=1):
        inner = lambda mod: mod.importance_weight(x)
        return nnx.vmap(inner, state_axes={}, out_axes=1, axis_size=num_samples)(self)

    @nnx.jit
    def conditional_probability(self, s, x):
        return self.forward(s, x)

    @nnx.jit
    def marginal_probability(self, x, N=2**14):
        x = jnp.reshape(x, (-1, x.shape[-1]))

        def log_p(carry, mod, x):
            log_weights = mod.elbo(x, num_samples=N)
            ess = 1 / jnp.sum(jax.nn.softmax(log_weights, axis=1) ** 2) / N
            logp = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(N)
            return None, (logp, ess)

        _, (logp, ess) = nnx.scan(log_p, state_axes={})(None, self, x)
        return jnp.squeeze(logp, 1), ess


class TrainerModule:
    def __init__(self, model, s, x, logdir):
        self.s = s
        self.x = x
        self.model = model
        self.logdir = logdir
        self.create_functions()

    @staticmethod
    def forward_loss(model, s, x):
        loss = -model(s, x)
        return jnp.mean(loss), {'loss': loss}

    @staticmethod
    def backward_loss(model, x, subsample=1):
        elbo = model.elbo(x, num_samples=subsample)
        loss = -jnp.mean(elbo)
        return loss, {'elbo': elbo}

    def create_functions(self):
        def train_step_forward(model, optimizer, s_batch, x_batch):
            grad_fn = nnx.value_and_grad(self.forward_loss, has_aux=True)
            (loss, metrics), grads = grad_fn(model, s_batch, x_batch)
            optimizer.update(grads=grads)
            return loss, metrics
        self.train_step_forward = nnx.jit(train_step_forward)

        def train_step_backward(model, optimizer, x, subsample=1):
            grad_fn = nnx.value_and_grad(self.backward_loss, has_aux=True, wrt=optimizer.wrt)
            (loss, metrics), grads = grad_fn(model, x, subsample)
            optimizer.update(grads=grads)
            return loss, metrics
        self.train_step_backward = nnx.jit(train_step_backward, static_argnames='subsample')

    def train_forward_model(self, key, num_steps=500, batch_size=64, learning_rate=1e-2):
        num_samples = self.s.shape[0]
        schedule = optax.cosine_decay_schedule(
            learning_rate, num_steps *  len(range(0, num_samples, batch_size)), 
            alpha=0.1
        )
        optimizer = nnx.Optimizer(self.model.forward, optax.adamw(schedule))

        AverageLoss = metrics.Average.from_output('loss')
        writer = metric_writers.create_default_writer(os.path.join(self.logdir, 'Forward'))

        for epoch in range(num_steps):
            epoch_key = random.fold_in(key, epoch)
            perm = random.permutation(epoch_key, num_samples)
            s_shuffle = self.s[perm]
            x_shuffle = self.x[perm]

            average_loss = AverageLoss.empty()
            for j in range(0, num_samples, batch_size):
                s_batch = s_shuffle[j:j+batch_size]
                x_batch = x_shuffle[j:j+batch_size]
                loss, train_metrics = self.train_step_forward(
                    self.model.forward, 
                    optimizer, 
                    s_batch, 
                    x_batch
                )
                average_loss = average_loss.merge(
                    AverageLoss.from_model_output(
                        loss=train_metrics['loss']
                    )
                )
            scalars = {
                'loss': average_loss.compute(), 
                'learning rate': schedule(optimizer.step.value)
            }
            writer.write_scalars(epoch + 1, scalars)

    def train_backward_model(
        self, 
        key, 
        num_steps=500, 
        batch_size=64, 
        subsample=16, 
        learning_rate=5e-3
    ):
        num_samples = self.x.shape[0]
        
        # only optimize parameters of backward model
        backward_filter = nnx.All(nnx.Param, lambda path, val: 'backward' in path)

        schedule = optax.exponential_decay(
            learning_rate, num_steps *  len(range(0, num_samples, batch_size)), 
            0.5
        )
        
        optimizer = nnx.Optimizer(self.model, optax.adamw(schedule), backward_filter)

        AverageLoss = metrics.Average.from_output('loss')
        writer = metric_writers.create_default_writer(os.path.join(self.logdir, 'Backward'))

        for epoch in range(num_steps):
            epoch_key = random.fold_in(key, epoch)
            perm = random.permutation(epoch_key, num_samples)
            x_shuffle = self.x[perm]

            average_loss = AverageLoss.empty()
            for j in range(0, num_samples, batch_size):
                x_batch = x_shuffle[j:j+batch_size]
                loss, train_metrics = self.train_step_backward(
                    self.model, 
                    optimizer, 
                    x_batch, 
                    subsample=subsample
                )
                average_loss = average_loss.merge(
                    AverageLoss.from_model_output(
                        loss=-train_metrics['elbo']
                    )
                )
            scalars = {
                'loss': average_loss.compute(), 
                'learning rate': schedule(optimizer.step.value)
            }
            writer.write_scalars(epoch + 1, scalars)

    def mutual_information(self, s, x):
        cond = self.model.conditional_probability(s, x)
        marg, ess = self.model.marginal_probability(x)

        return jnp.mean(cond - marg), jnp.mean(ess)


def run_simulation(rho=1.0, mu=1.0, n=1.0, K=100.0, length=500, dt=1e-2, sample_size = 1000):
    input_model = InputModel(dt=dt, rngs=nnx.Rngs(1))
    output_model = NonlinearModel(rho=rho, mu=mu, n=n, K=K, dt=dt, rngs=nnx.Rngs(2))

    comb_model = CombinedModel(
        prior=input_model,
        forward=output_model,
        backward=VariationalRnn(64, 16, rngs=nnx.Rngs(3)),
        # backward=IAF(21, 256, 8, rngs=nnx.Rngs(3)),
    )

    _, s = input_model(jnp.empty((sample_size, length)), generate=True)
    _, x = output_model(s, jnp.empty_like(s), generate=True)

    trainer = TrainerModule(comb_model)
    trainer.train_backward_model(random.key(1), x, 100, subsample=1)

    data = []
    for length in [1] + list(range(100, length + 1, 100)):
        mi, ess = trainer.mutual_information(length=length, sample_size=sample_size)
        df = pl.DataFrame(
            {"mi": np.asarray(mi), "ess": np.asarray(ess), "length": length}
        )
        print(df.mean())
        data.append(df)
    return pl.concat(data)


if __name__ == "__main__":
    args = parse_args()
    
    # output_file = args.output_filename

    # torch.set_num_threads(args.threads)
    # spike_model = train_model(args.dataset_path, args.neurons)

    # t = torch.arange(100) * BIN_WIDTH * SECONDS_PER_UNIT

    # pws_result = pws_estimate(spike_model, t, 400, 2048)
    # mi = pws_result[:, 0] - pws_result[:, 1]

    # results = {
    #     "args": {
    #         "neurons": args.neurons,
    #         "N": 400,
    #         "M": 2048,
    #     },
    #     "pws_result": {
    #         "t": t.tolist(),
    #         "log_conditional": pws_result[:, 0].tolist(),
    #         "log_marginal": pws_result[:, 1].tolist(),
    #         "mutual_information": mi.tolist(),
    #     },
    # }

    # output_file.parent.mkdir(parents=True, exist_ok=True)

    # # Write to JSON
    # with output_file.open("w") as f:
    #     json.dump(results, f, indent=2)

    # print(f"Results saved to {output_file}")

