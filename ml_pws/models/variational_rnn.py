import jax
from jax import random
import jax.numpy as jnp
from flax import nnx

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


class DecoderCell(nnx.RNNCellBase):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        rngs: nnx.Rngs,
        mixture_components: int = 5,
    ):
        self.cell = nnx.GRUCell(in_features, hidden_size, rngs=rngs)
        self.shift = nnx.Linear(hidden_size, 1, rngs=rngs)
        self.log_scale = nnx.Linear(hidden_size, 1, rngs=rngs)
        self.pi = nnx.Linear(hidden_size, mixture_components, rngs=rngs)
        self.mu = nnx.Linear(hidden_size, mixture_components, rngs=rngs)
        self.log_s = nnx.Linear(hidden_size, mixture_components, rngs=rngs)

    def __call__(self, carry, input):
        rnn_state, last_prediction = carry
        input = input.at[..., 0].set(last_prediction)

        rnn_state, y = self.cell(rnn_state, input)

        shift = jnp.squeeze(self.shift(y), -1)
        log_scale = jnp.squeeze(self.log_scale(y), -1)
        pi = self.pi(y)
        mu = self.mu(y)
        log_s = self.log_s(y)

        epsilon = random.logistic(self.rngs(), shape=y.shape[:-1])

        val, log_jac = coupling_function(epsilon, pi, mu, log_s)
        prediction = shift + val * jnp.exp(log_scale)

        logp = jax.scipy.stats.logistic.logpdf(epsilon) - log_scale - log_jac

        return (rnn_state, prediction), (logp, prediction)

    def initialize_carry(self, input_shape: tuple[int, ...], rngs):
        batch_size = input_shape[0]
        self.rngs = rngs
        cell_state = self.cell.initialize_carry(input_shape, rngs)
        initial_prediction = jnp.zeros(batch_size)
        return (cell_state, initial_prediction)

    @property
    def num_feature_axes(self):
        return 1


class VariationalRnn(nnx.Module):
    def __init__(self, hidden_size: int, mixture_components: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.encoder_x = nnx.RNN(nnx.GRUCell(1, hidden_size, rngs=rngs), reverse=True)
        self.decoder_rnn = nnx.RNN(
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
        (logp, preds) = self.decoder_rnn(s_with_context, return_carry=False)

        return logp, preds

