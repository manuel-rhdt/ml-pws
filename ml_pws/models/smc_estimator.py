import jax
import jax.numpy as jnp
from jax import random

from flax import nnx, struct

import optax
import polars as pl
import numpy as np
from tqdm import trange

from clu import metric_writers

from .variational_rnn import FlowRNNCell


def resample(key, logits, data):
    indices = jax.random.categorical(key, logits, shape=logits.shape)

    def shuffle(d):
        if d is None or d.ndim < 1:
            return d
        else:
            return d[indices]

    return jax.tree_util.tree_map(shuffle, data)


@struct.dataclass
class SMCMetrics:
    t: int
    ess_values: jax.Array
    resampling_flags: jax.Array

    @classmethod
    def initialize(cls, n_steps: int):
        """Initialize metrics container with zeros."""
        return cls(
            t=0,
            ess_values=jnp.zeros(n_steps),
            resampling_flags=jnp.zeros(n_steps, dtype=bool),
        )

    def update(self, ess: float, resampled: bool):
        """Return a new SMCMetrics with updated step t (functional update)."""
        return SMCMetrics(
            t=self.t + 1,
            ess_values=self.ess_values.at[self.t].set(ess),
            resampling_flags=self.resampling_flags.at[self.t].set(resampled),
        )


class SMCEstimator(nnx.Module):
    def __init__(
        self,
        input_cell,
        variational_cell: FlowRNNCell | None,
        output_cell,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_cell = input_cell
        self.variational_cell = variational_cell
        self.output_cell = output_cell
        self.rngs = rngs

        state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
        self.scan_step = nnx.scan(SMCEstimator.step, in_axes=(state_axes, nnx.Carry, 0))
        self.scan_logp = nnx.scan(
            SMCEstimator.step_output,
            in_axes=(None, nnx.Carry, 1, 1),
            out_axes=(nnx.Carry, 1),
        )

    def step_input(self, c1, s):
        return self.input_cell(c1, s, generate=False)

    def step_variational(self, c2, epsilon, x):
        if self.variational_cell is None:
            return None, (0.0, epsilon)
        sx = jnp.stack((epsilon, x), axis=-1)
        return self.variational_cell(c2, sx)

    def step_output(self, c3, s, x):
        return self.output_cell(c3, s, x, generate=False)

    def elbo_loss(self, x: jax.Array):
        carry = self.initialize_carry(*x.shape)[2]

        state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

        @nnx.split_rngs(splits=x.shape[1])
        @nnx.scan(in_axes=(state_axes, nnx.Carry, 1), out_axes=(nnx.Carry, 0, 1, 1))
        def scan_fn(smc_estimator, carry, x):
            (c1, c2, c3) = carry

            epsilon = random.normal(smc_estimator.rngs(), x.shape)
            c2, (log_jac, preds) = self.step_variational(c2, epsilon, x)
            logq_variational = jax.scipy.stats.norm.logpdf(epsilon) + log_jac
            c1, logp_s = self.step_input(c1, preds)
            c3, log_cond = self.step_output(c3, preds, x)
            kl_div = logq_variational - logp_s

            elbo = jnp.mean(log_cond - kl_div)

            return (c1, c2, c3), elbo, kl_div, log_cond

        _, elbo, kl_div, log_cond = scan_fn(self, carry, x)
        return -jnp.mean(elbo), {"kl_div": kl_div, "log_cond": log_cond}

    def step(self, carry, x: jax.Array):
        logits, log_marginal_estimate, (c1, c2, c3), metrics = carry

        n_particles = len(logits)
        x = jnp.repeat(x, n_particles)

        epsilon = random.normal(self.rngs(), (n_particles,))
        c2, (log_jac, preds) = self.step_variational(c2, epsilon, x)
        logq_variational = jax.scipy.stats.norm.logpdf(epsilon) + log_jac
        c1, logp_s = self.step_input(c1, preds)
        c3, log_cond = self.step_output(c3, preds, x)
        kl_div = logq_variational - logp_s

        logits += log_cond - kl_div
        current_log_marginal = (
            log_marginal_estimate + jax.nn.logsumexp(logits) - jnp.log(n_particles)
        )

        # Calculate effective sample size (ESS)
        ess = 1.0 / jnp.sum(jax.nn.softmax(logits) ** 2)

        # we resample only if needed

        def smc_resample(key, carry):
            logits, log_marginal_estimate, c = carry
            c = resample(key, logits, c)
            log_marginal_estimate += jax.nn.logsumexp(logits) - jnp.log(len(logits))
            logits = jnp.zeros_like(logits)
            return logits, log_marginal_estimate, c

        key = self.rngs()
        # Threshold for adaptive resampling, set to N/2 for particle count N
        resample_cond = ess < n_particles / 2
        carry = jax.lax.cond(
            resample_cond,
            smc_resample,
            lambda _, *args: tuple(*args),
            key,
            (logits, log_marginal_estimate, (c1, c2, c3)),
        )

        metrics = metrics.update(ess, resample_cond)

        return carry + (metrics,), current_log_marginal

    def predict(self, x, num_samples: int):
        carry = self.initialize_carry(num_samples, *x.shape)[2]

        state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

        @nnx.split_rngs(splits=x.shape[0])
        @nnx.scan(in_axes=(state_axes, nnx.Carry, 0), out_axes=(nnx.Carry, 1))
        def scan_fn(smc_estimator, carry, x):
            (c1, c2, c3) = carry
            x = jnp.repeat(x, num_samples)

            epsilon = random.normal(smc_estimator.rngs(), (num_samples,))
            c2, (_, preds) = self.step_variational(c2, epsilon, x)

            return (c1, c2, c3), preds

        _, preds = scan_fn(self, carry, x)
        return preds

    def log_conditional_estimate(self, s, x):
        carry = self.output_cell.initialize_carry(x.shape, key=None)
        return jnp.cumsum(self.scan_logp(self, carry, s, x)[1], 1)

    def log_marginal_estimate(self, x, *, n_particles: int):
        carry = self.initialize_carry(n_particles, len(x))
        with nnx.split_rngs(self, splits=len(x)):
            (_, _, _, metrics), result = self.scan_step(self, carry, x)
        return result, metrics

    def initialize_carry(self, n_particles: int, n_steps: int):
        shape = (n_particles, n_steps)
        c1 = self.input_cell.initialize_carry(shape)
        if self.variational_cell is None:
            c2 = None
        else:
            c2 = self.variational_cell.initialize_carry(shape, rngs=nnx.Rngs(0))
        c3 = self.output_cell.initialize_carry(shape, key=None)

        logits = jnp.zeros(n_particles)
        log_marginal_estimate = 0.0

        metrics = SMCMetrics.initialize(n_steps)

        return (logits, log_marginal_estimate, (c1, c2, c3), metrics)


def train_forward_model(
    smc_estimator,
    train_s,
    train_x,
    *,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    rng_key=random.key(0),
):
    filter = nnx.All(nnx.Param, nnx.PathContains("output_cell"))
    diffstate = nnx.DiffState(0, filter)
    optimizer = nnx.Optimizer(smc_estimator, optax.adamw(learning_rate), wrt=filter)

    def forward_loss(smc_estimator, s, x):
        logp = smc_estimator.log_conditional_estimate(s, x)
        return -jnp.mean(logp[:, -1])

    @nnx.jit
    def train_step(smc_estimator, optimizer, s, x):
        grad_fn = nnx.value_and_grad(forward_loss, argnums=diffstate, has_aux=False)
        loss, grads = grad_fn(smc_estimator, s, x)
        optimizer.update(grads)
        return loss

    def create_batches(data, batch_size, rng_key):
        """Creates shuffled mini-batches from a dataset."""
        num_samples = data[0].shape[0]
        indices = jax.random.permutation(rng_key, num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            yield jax.tree_util.tree_map(lambda x: x[batch_indices], data)

    writer = metric_writers.LoggingWriter()
    train_log = []
    for epoch in range(num_epochs):
        rng_key, _ = jax.random.split(rng_key)
        batches = create_batches((train_s, train_x), batch_size, rng_key)
        epoch_loss = 0.0
        num_batches = 0
        for s, x in batches:
            loss = train_step(smc_estimator, optimizer, s, x)
            epoch_loss += loss
            num_batches += 1
        train_log.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / num_batches,
                "num_batches": num_batches,
                "batch_size": batch_size,
            }
        )
        writer.write_scalars(epoch, {"loss": epoch_loss / num_batches})
    return pl.DataFrame(train_log)


def train_backward_model(smc_estimator, train_x, *, num_epochs=200, learning_rate=1e-2):
    filter = nnx.All(nnx.Param, nnx.PathContains("variational_cell"))
    diffstate = nnx.DiffState(0, filter)
    optimizer = nnx.Optimizer(smc_estimator, optax.adamw(learning_rate), wrt=filter)

    @nnx.jit
    def train_step(smc_estimator, optimizer, x):
        grad_fn = nnx.value_and_grad(
            SMCEstimator.elbo_loss, argnums=diffstate, has_aux=True
        )
        (loss, metrics), grads = grad_fn(smc_estimator, x)
        optimizer.update(grads)
        return loss, metrics

    writer = metric_writers.LoggingWriter()
    train_log = []
    for epoch in range(num_epochs):
        loss, metrics = train_step(smc_estimator, optimizer, train_x)
        scalars = {
            "loss": loss,
            "kl_div": jnp.mean(metrics["kl_div"]),
            "log_cond": jnp.mean(metrics["log_cond"]),
        }
        writer.write_scalars(epoch, scalars)
        train_log.append({"epoch": epoch + 1} | scalars)
    return pl.DataFrame(train_log)


def estimate_mi(smc_estimator, val_s, val_x, *, n_particles=512):
    log_conditional = smc_estimator.log_conditional_estimate(val_s, val_x)

    dfs = []
    for i in trange(val_x.shape[0], desc="MI Estimation"):
        x = val_x[i]
        log_marginal, metrics = smc_estimator.log_marginal_estimate(
            x, n_particles=n_particles
        )
        df = pl.DataFrame(
            {
                "step": np.arange(x.shape[0]) + 1,
                "log_marginal": np.asarray(log_marginal),
                "log_conditional": np.asarray(log_conditional[i]),
                "ess": np.asarray(metrics.ess_values),
                "resample_flags": np.asarray(metrics.resampling_flags),
                "n_particles": n_particles,
            }
        )
        dfs.append(df)
    return pl.concat(dfs)
