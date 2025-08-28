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

    def importance_weight(self, s, x, reduction="sum"):
        kl_div, preds = self.backward(s, x)
        logp_x_given_s = self.forward(preds, x)
        if reduction == "sum":
            return logp_x_given_s.sum(-1) - kl_div.sum(-1)
        else:
            return logp_x_given_s - kl_div

    def elbo(self, s, x, reduction="sum"):
        inner = lambda mod, x: mod.importance_weight(x, reduction)
        state_axes = nnx.StateAxes({"default": 0, ...: None})
        return nnx.vmap(inner, in_axes=(state_axes, None, None), out_axes=1)(self, s, x)

    @nnx.jit
    def conditional_probability(self, s, x):
        return self.forward(s, x)

    @nnx.jit(static_argnames="N")
    def marginal_probability(self, x, N=2**10):
        x = jnp.reshape(x, (-1, x.shape[-1]))

        def log_p(mod, x):
            # shape (1, N, length)
            log_weights = mod.elbo(jnp.expand_dims(x, 0), reduction=None)
            ess = 1 / jnp.sum(jax.nn.softmax(log_weights.sum(-1), axis=1) ** 2) / N
            # shape (1, length)
            logp = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(N)
            return jnp.squeeze(logp, 0), ess

        with nnx.split_rngs(self.backward, splits=N):
            logp, ess = nnx.scan(log_p, in_axes=(None, 0), out_axes=0)(self, x)
            return logp, ess


class Trainer:
    def __init__(self, model, s, x, val_s, val_x, logdir):
        self.s = s
        self.x = x
        self.val_s = val_s
        self.val_x = val_x
        self.model = model
        self.logdir = logdir
        self.create_functions()

    @staticmethod
    def forward_loss(model, s, x):
        loss = -model(s, x)  # negative log likelihood loss
        return jnp.mean(loss), {"loss": loss}

    @staticmethod
    def backward_loss(model, s, x, subsample):
        # shape: (batch_size, subsample)
        elbo = model.elbo(s, x, reduction="sum")
        loss = -jnp.mean(elbo)
        return loss, {"elbo": elbo.mean(-1)}

    def create_functions(self):
        def train_step_forward(model, optimizer, s_batch, x_batch):
            grad_fn = nnx.value_and_grad(self.forward_loss, has_aux=True)
            (loss, metrics), grads = grad_fn(model, s_batch, x_batch)
            optimizer.update(grads=grads)
            return loss, metrics

        self.train_step_forward = nnx.jit(train_step_forward)

        def train_step_backward(model, optimizer, s, x, subsample):
            grad_fn = nnx.value_and_grad(
                self.backward_loss,
                argnums=nnx.DiffState(0, optimizer.wrt),
                has_aux=True,
            )
            (loss, metrics), grads = grad_fn(model, s, x, subsample=subsample)
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
        optimizer = nnx.Optimizer(
            self.model.forward, optax.adamw(schedule, weight_decay=1e-3)
        )

        AverageLoss = metrics.Average.from_output("loss")
        writer = metric_writers.create_default_writer(self.logdir / "Forward")

        forward_loss = nnx.jit(self.forward_loss)

        for epoch in range(num_steps):
            epoch_key = random.fold_in(key, epoch)
            perm = random.permutation(epoch_key, num_samples)
            s_shuffle = self.s[perm]
            x_shuffle = self.x[perm]

            self.model.forward.train()
            average_loss = AverageLoss.empty()
            for j in range(0, num_samples, batch_size):
                s_batch = s_shuffle[j : j + batch_size]
                x_batch = x_shuffle[j : j + batch_size]
                _, train_metrics = self.train_step_forward(
                    self.model.forward, optimizer, s_batch, x_batch
                )
                average_loss = average_loss.merge(
                    AverageLoss.from_model_output(loss=train_metrics["loss"])
                )

            self.model.forward.eval()
            _, val_metrics = forward_loss(self.model.forward, self.val_s, self.val_x)
            scalars = {
                "loss": average_loss.compute(),
                "val_loss": jnp.mean(val_metrics["loss"]),
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
            learning_rate, num_steps * len(range(0, num_samples, batch_size)), 0.1
        )

        optimizer = nnx.Optimizer(self.model, optax.adam(schedule), backward_filter)

        BackwardsMetrics = metrics.Collection.create(
            loss=metrics.Average.from_output("loss"),
        )
        writer = metric_writers.create_default_writer(self.logdir / "Backward")

        for epoch in range(num_steps):
            epoch_key1, epoch_key2 = random.split(random.fold_in(key, epoch))
            x_shuffle = random.permutation(epoch_key1, self.x)
            s_shuffle = random.permutation(epoch_key2, self.s)

            backwards_metrics = BackwardsMetrics.empty()
            for j in range(0, num_samples, batch_size):
                s_batch = s_shuffle[j : j + batch_size]
                x_batch = x_shuffle[j : j + batch_size]
                with nnx.split_rngs(self.model.backward, splits=subsample):
                    loss, train_metrics = self.train_step_backward(
                        self.model, optimizer, s_batch, x_batch, subsample=subsample
                    )
                backwards_metrics = backwards_metrics.merge(
                    BackwardsMetrics.single_from_model_output(
                        loss=-train_metrics["elbo"],
                    )
                )

            # mi, ess = self.mutual_information(self.val_s[perm[:8]], self.val_x[perm[:8]])

            scalars = backwards_metrics.compute()
            # scalars = scalars | {
            #     "learning rate": schedule(optimizer.step.value),
            #     "mi": jnp.mean(mi, 0)[-1] / jnp.log(2),
            #     "ess": jnp.mean(ess, 0)
            # }
            writer.write_scalars(epoch + 1, scalars)

    def mutual_information(self, s: jax.Array, x: jax.Array):
        cond = self.model.conditional_probability(s, x)
        marg, ess = self.model.marginal_probability(x)

        return jnp.cumsum(cond - marg, -1), ess
