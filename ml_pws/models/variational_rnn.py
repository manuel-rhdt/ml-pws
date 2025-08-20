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


class CouplingTransform(nnx.Module):
    def __init__(
        self, in_features: int, mixture_components: int = 5, *, rngs: nnx.Rngs
    ):
        super().__init__()
        self.shift = nnx.Linear(in_features, 1, rngs=rngs)
        self.log_scale = nnx.Linear(in_features, 1, rngs=rngs)
        self.pi = nnx.Linear(in_features, mixture_components, rngs=rngs)
        self.mu = nnx.Linear(in_features, mixture_components, rngs=rngs)
        self.log_s = nnx.Linear(in_features, mixture_components, rngs=rngs)

    def __call__(self, params, epsilon):
        shift = jnp.squeeze(self.shift(params), -1)
        log_scale = jnp.squeeze(self.log_scale(params), -1)
        pi = self.pi(params)
        mu = self.mu(params)
        log_s = self.log_s(params)

        val, log_jac = coupling_function(epsilon, pi, mu, log_s)
        prediction = shift + val * jnp.exp(log_scale)

        return (log_scale + log_jac, prediction)


class DecoderCell(nnx.RNNCellBase):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        rngs: nnx.Rngs,
        mixture_components: int = 5,
    ):
        self.cell = nnx.GRUCell(in_features, hidden_size, rngs=rngs)
        self.cell.rngs = None
        self.transform = CouplingTransform(hidden_size, mixture_components, rngs=rngs)

    def __call__(self, carry, input):
        rnn_state, last_prediction = carry
        epsilon = input[..., 0]
        context = input[..., 1:]
        y = jnp.concatenate([jnp.expand_dims(last_prediction, -1), context], axis=-1)
        rnn_state, y = self.cell(rnn_state, y)
        log_jac, prediction = self.transform(y, epsilon)
        return (rnn_state, prediction), (log_jac, prediction)

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: nnx.Rngs | None = None):
        batch_size = input_shape[0]
        cell_state = self.cell.initialize_carry(input_shape, rngs)
        initial_prediction = jnp.zeros(batch_size)
        return (cell_state, initial_prediction)

    @property
    def num_feature_axes(self):
        return 1


class VariationalRnn(nnx.Module):
    def __init__(self, hidden_size: int, mixture_components: int, num_layers: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.encoder_x = nnx.RNN(
            nnx.SimpleCell(1, hidden_size, rngs=rngs), reverse=True
        )

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(axis_size=num_layers)
        def create_block(rngs: nnx.Rngs):
            return nnx.RNN(
                DecoderCell(
                    1 + hidden_size,
                    hidden_size,
                    mixture_components=mixture_components,
                    rngs=rngs,
                ),
                rngs=False
            )
        self.decoder_rnn = create_block(rngs)
        self.rngs = rngs

    # generate new sequence conditional on x
    def __call__(self, x: jax.Array):
        # Make input tensor of shape [batch_size, seq_length, 1]
        x = x.reshape((-1, x.shape[-1]) + (1,))

        # apply reverse rnn to x
        h_x = self.encoder_x(x)

        epsilon = random.logistic(self.rngs(), shape=x.shape[:-1])
        logp = jax.scipy.stats.logistic.logpdf(epsilon)

        @nnx.scan
        def scan_fn(carry, rnn_block: nnx.RNN):
            logp, epsilon = carry
            rnn_input = jnp.concatenate((jnp.expand_dims(epsilon, -1), h_x), axis=-1)
            (log_jac, preds) = rnn_block(rnn_input, return_carry=False, rngs=nnx.Rngs(0))
            return (logp - log_jac, preds), None

        (logp, preds), _ = scan_fn((logp, epsilon), self.decoder_rnn)
        return logp, preds


def shift_right(x, axis=1):
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    ind = [slice(None)] * len(x.shape)
    ind[axis] = slice(-1)
    return jnp.pad(x, pad_widths)[tuple(ind)]


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
        self.encoder_x = nnx.RNN(
            nnx.GRUCell(1, hidden_features, rngs=rngs), reverse=True
        )
        self.initial_loc = nnx.Linear(
            in_features=hidden_features, out_features=1, rngs=rngs
        )
        self.initial_log_scale = nnx.Linear(
            in_features=hidden_features, out_features=1, rngs=rngs
        )

        @nnx.split_rngs(splits=depth)
        @nnx.vmap(axis_size=depth)
        def create_block(rngs):
            return IAFBlock(kernel_size, hidden_features, rngs)

        self.decoder = create_block(rngs)

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
