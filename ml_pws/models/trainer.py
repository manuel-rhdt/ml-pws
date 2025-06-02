from jax import random
import jax.numpy as jnp
from flax import nnx
import optax
from clu import metrics, metric_writers

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