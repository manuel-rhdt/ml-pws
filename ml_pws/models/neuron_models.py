
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr

import lightning as L

TAU = 50e-3
OMEGA_0 = 9.42
DAMPING_COEFFICIENT = 1.0 / (2 * OMEGA_0 * TAU)

class StochasticHarmonicOscillator(nn.Module):
    noise_type = "scalar"
    sde_type = "ito"

    def __init__(self, damping_coefficient=DAMPING_COEFFICIENT, tau=TAU):
        super().__init__()

        self.register_buffer(
            "A",
            torch.tensor(
                [[0.0, 1.0], [-1 / (4 * damping_coefficient**2), -1.0]],
                dtype=torch.float32,
            )
            / tau,
        )

        self.register_buffer(
            "B",
            torch.tensor(
                [[0.0, 1.0 / (np.sqrt(2.0) * damping_coefficient)]], dtype=torch.float32
            )
            / np.sqrt(tau),
        )

    # Drift
    def f(self, t, y):
        return y @ self.A.T

    # Diffusion
    def g(self, t, y):
        return self.B.expand(y.size(0), 2).unsqueeze(-1)


class ConditionalSpikeCNN(nn.Module):
    def __init__(self, n_neurons: int, hidden_size: int, num_layers: int = 1, kernel_size=10):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.n_neurons = n_neurons
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size - 1) * num_layers

        layers = []
        for i in range(num_layers):
            in_size = (1 + n_neurons) if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_size, hidden_size, kernel_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

        self.conv_net = nn.Sequential(
            *layers,
            # 1x1 convolution layer to output spike intensity for each neuron
            nn.Conv1d(hidden_size, n_neurons, 1)
        )

    def forward(self, s: torch.Tensor, x: torch.Tensor, cache: torch.Tensor | None = None):
        """
        Forward pass of the CNN. S is the stimulus, x the neuronal outputs.
        When using this function, x needs be shifted-right wrt so s. I.e.
        s = [s_0, s_1, s_2, ...]
        x = [0.0, x_0, x_1, ...]
        """
        seq_len, _batch_size, n_neurons = x.size()

        if seq_len != s.size(0):
            raise ValueError(f"Sequence lengths do not match. s: {s.size(0)} x: {seq_len}")

        if n_neurons != self.n_neurons:
            raise ValueError(f"Wrong shape of x: {x.size()}. Expected last dimension {self.n_neurons}.")

        if s.ndim == 2:
            s = s.transpose(0, 1).unsqueeze(1) # (batch_size, 1, seq_len)
        else:
            raise ValueError(f"Wrong shape of s: {s.shape}, expected (seq_len, batch_size)")

        conv_input = torch.cat([s, x.permute((1, 2, 0))], dim=1) # (batch_size, n_neurons + 1, seq_len)

        if cache is None:
            conv_input = F.pad(conv_input, (self.pad_size, 0), mode='constant', value=0)  # (batch_size, n_neurons + 1, seq_len + self.pad_size)
        else:
            assert cache.size(-1) == self.pad_size, f"Cache has wrong size {cache.size(-1)}, expected {self.pad_size}"
            conv_input = torch.cat([cache, conv_input], dim=-1)
        cache = conv_input[..., -self.pad_size:]
        
        output = self.conv_net(conv_input) # (batch_size, n_neurons, seq_len)
        
        # returns the log-intensities (seq_len, batch_size, n_neurons)
        return output.permute((2, 0, 1)), cache
    
    @torch.no_grad()
    def sample(self, s: torch.Tensor):
        seq_len, batch_size = s.size()
        device = s.device

        # output tensor
        x = torch.zeros((seq_len, batch_size, self.n_neurons), device=device)

        cache = None

        for t in range(seq_len):
            if t == 0:
                out, cache = self(s[[t]], x[[0]], cache)
            else:
                out, cache = self(s[[t]], x[[t-1]], cache)
            log_intensity = out.squeeze(0).clamp(-10, 10)
            x[t] = torch.poisson(log_intensity.exp())
            
        return x


def shift_right(x):
    x = torch.roll(x, 1, dims=0)
    x[0].zero_()
    return x

class SpikeModel(L.LightningModule):
    def __init__(self, n_neurons: int, hidden_size: int, num_layers: int = 1, kernel_size=None, model_type='RNN'):
        super().__init__()
        self.save_hyperparameters()
        if model_type == 'RNN':
            raise ValueError('RNN currently not supported.')
        elif model_type == 'CNN':
            if kernel_size is None:
                kernel_size = 10
            self.net = ConditionalSpikeCNN(n_neurons, hidden_size, num_layers, kernel_size=kernel_size)
        else:
            raise ValueError(f"unsupported model type {model_type}")
        self.loss_fn = nn.PoissonNLLLoss(log_input=True)

    def training_step(self, batch, batch_idx):
        _, s, x = batch
        log_intensity, _ = self.net(s, shift_right(x))
        train_loss = self.loss_fn(log_intensity, x)
        self.log("train_loss", train_loss, prog_bar=True, batch_size=x.size(1))
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        _, s, x = batch
        log_intensity, _ = self.net(s, shift_right(x))
        val_loss = self.loss_fn(log_intensity, x)

        # compute correlation
        actual_rate = x.mean(1, dtype=torch.float32)
        assert isinstance(self.net, ConditionalSpikeCNN)
        self.net.eval()
        with torch.no_grad():
            predicted_rate = self.net.sample(s).mean(1)
        self.net.train()
        corr = pearsonr(predicted_rate, actual_rate, axis=0).statistic # pyright: ignore[reportAttributeAccessIssue]

        values = {"val_loss": val_loss, "corr": np.nanmean(corr)}
        # Log validations
        self.log_dict(values, prog_bar=True, batch_size=x.size(1))
        return val_loss

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        assert isinstance(total_steps, int)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, total_steps=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
