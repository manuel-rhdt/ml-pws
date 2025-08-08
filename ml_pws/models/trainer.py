from dataclasses import dataclass
import jax
from jax import random
import jax.numpy as jnp
from flax import nnx
import optax
from clu import metrics, metric_writers


@dataclass
class CombinedModel(nnx.Module):
    prior: nnx.Module
    forward: nnx.Module
    backward: nnx.Module

    def importance_weight(self, x, reduction="sum"):
        s = jnp.empty_like(x)
        logq_s, preds = self.backward(s, x)
        logp_s = self.prior(preds)
        logp_x_given_s = self.forward(preds, x)
        if reduction == "sum":
            return logp_s.sum(-1) + logp_x_given_s.sum(-1) - logq_s.sum(-1)
        else:
            return logp_s + logp_x_given_s - logq_s

    def elbo(self, x, num_samples=1, reduction="sum"):
        inner = lambda mod, x: mod.importance_weight(x, reduction)
        return nnx.vmap(inner, in_axes=(None, None), out_axes=1, axis_size=num_samples)(
            self, x
        )

    @nnx.jit
    def conditional_probability(self, s, x):
        return self.forward(s, x)

    @nnx.jit(static_argnames="N")
    def marginal_probability(self, x, N=2**14):
        x = jnp.reshape(x, (-1, x.shape[-1]))

        def log_p(mod, x):
            # shape (1, N, length)
            log_weights = mod.elbo(jnp.expand_dims(x, 0), num_samples=N, reduction=None)
            ess = 1 / jnp.sum(jax.nn.softmax(log_weights.sum(-1), axis=1) ** 2) / N
            logp = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(N)
            return jnp.squeeze(logp, 0), ess

        logp, ess = nnx.scan(log_p, in_axes=(None, 0), out_axes=0)(self, x)
        return logp, ess


class Trainer:
    def __init__(self, model, s, x, logdir):
        self.s = s
        self.x = x
        self.model = model
        self.logdir = logdir
        self.create_functions()

    @staticmethod
    def forward_loss(model, s, x):
        loss = -model(s, x)  # negative log likelihood loss
        return jnp.mean(loss), {"loss": loss}

    @staticmethod
    def backward_loss(model, x, subsample=1):
        elbo = model.elbo(x, num_samples=subsample)
        loss = -jnp.mean(elbo)
        return loss, {"elbo": elbo}

    def create_functions(self):
        def train_step_forward(model, optimizer, s_batch, x_batch):
            grad_fn = nnx.value_and_grad(self.forward_loss, has_aux=True)
            (loss, metrics), grads = grad_fn(model, s_batch, x_batch)
            optimizer.update(grads=grads)
            return loss, metrics

        self.train_step_forward = nnx.jit(train_step_forward)

        def train_step_backward(model, optimizer, x, subsample=1):
            grad_fn = nnx.value_and_grad(
                self.backward_loss,
                argnums=nnx.DiffState(0, optimizer.wrt),
                has_aux=True,
            )
            (loss, metrics), grads = grad_fn(model, x, subsample)
            optimizer.update(grads=grads)
            return loss, metrics

        self.train_step_backward = nnx.jit(
            train_step_backward, static_argnames="subsample"
        )

    def train_forward_model(
        self, key, num_steps=500, batch_size=64, learning_rate=1e-2
    ):
        num_samples = self.s.shape[0]
        schedule = optax.cosine_decay_schedule(
            learning_rate, num_steps * len(range(0, num_samples, batch_size)), alpha=0.1
        )
        optimizer = nnx.Optimizer(self.model.forward, optax.adamw(schedule))

        AverageLoss = metrics.Average.from_output("loss")
        writer = metric_writers.create_default_writer(self.logdir / "Forward")

        for epoch in range(num_steps):
            epoch_key = random.fold_in(key, epoch)
            perm = random.permutation(epoch_key, num_samples)
            s_shuffle = self.s[perm]
            x_shuffle = self.x[perm]

            average_loss = AverageLoss.empty()
            for j in range(0, num_samples, batch_size):
                s_batch = s_shuffle[j : j + batch_size]
                x_batch = x_shuffle[j : j + batch_size]
                loss, train_metrics = self.train_step_forward(
                    self.model.forward, optimizer, s_batch, x_batch
                )
                average_loss = average_loss.merge(
                    AverageLoss.from_model_output(loss=train_metrics["loss"])
                )
            scalars = {
                "loss": average_loss.compute(),
                "learning rate": schedule(optimizer.step.value),
            }
            writer.write_scalars(epoch + 1, scalars)

    def train_backward_model(
        self, key, num_steps=500, batch_size=64, subsample=16, learning_rate=5e-3
    ):
        num_samples = self.x.shape[0]

        # only optimize parameters of backward model
        backward_filter = nnx.All(nnx.Param, nnx.PathContains("backward"))

        schedule = optax.exponential_decay(
            learning_rate, num_steps * len(range(0, num_samples, batch_size)), 0.5
        )

        optimizer = nnx.Optimizer(self.model, optax.adamw(schedule), backward_filter)

        AverageLoss = metrics.Average.from_output("loss")
        writer = metric_writers.create_default_writer(self.logdir / "Backward")

        for epoch in range(num_steps):
            epoch_key = random.fold_in(key, epoch)
            perm = random.permutation(epoch_key, num_samples)
            x_shuffle = self.x[perm]

            average_loss = AverageLoss.empty()
            for j in range(0, num_samples, batch_size):
                x_batch = x_shuffle[j : j + batch_size]
                loss, train_metrics = self.train_step_backward(
                    self.model, optimizer, x_batch, subsample=subsample
                )
                average_loss = average_loss.merge(
                    AverageLoss.from_model_output(loss=-train_metrics["elbo"])
                )
            scalars = {
                "loss": average_loss.compute(),
                "learning rate": schedule(optimizer.step.value),
            }
            writer.write_scalars(epoch + 1, scalars)

    def mutual_information(self, s: jax.Array, x: jax.Array):
        cond = self.model.conditional_probability(s, x)
        marg, ess = self.model.marginal_probability(x)

        return jnp.cumsum(cond - marg, -1), ess
