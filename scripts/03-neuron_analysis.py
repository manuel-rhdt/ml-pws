import os

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchsde

from matplotlib import pyplot as plt

from torch.utils.data import Dataset

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import os
import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Process neural data.")

    parser.add_argument("dataset_path", type=Path, help="Path to the dataset file")

    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        required=True,
        help="List of neuron indices to process (e.g. --neurons 1 5 9)",
    )

    parser.add_argument(
        "-o",
        "--output_filename",
        type=Path,
        required=True,
        help="Path to the output JSON file",
    )

    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use (default: 1)"
    )

    return parser.parse_args()


BIN_WIDTH = 100
SECONDS_PER_UNIT = 1e-4
TAU = 50e-3
OMEGA_0 = 9.42
DAMPING_COEFFICIENT = 1.0 / (2 * OMEGA_0 * TAU)


class NeuronDataSet(Dataset):
    def __init__(self, filename, neurons=[0], intervals=None, delta_t=BIN_WIDTH):
        super().__init__()
        self.data = scipy.io.loadmat(filename, squeeze_me=True)
        self.neurons = neurons
        if intervals is None:
            self.intervals = np.stack(
                (self.data["rep_begin_time"], self.data["rep_end_time"]), axis=-1
            )
        else:
            self.intervals = np.array(intervals).reshape((-1, 2))
        self.delta_t = delta_t
        self.stimulus_std = np.std(self.data["stim"])

    def __len__(self):
        return self.intervals.shape[0]

    def __getitem__(self, index):
        t1, t2 = self.intervals[index]
        return self.time_discretized_spikes(t1, t2)

    def time_discretized_spikes(self, t_start, t_end):
        grid = np.arange(t_start, t_end, self.delta_t)

        peak_times = self.data["peak_times"][:-2]
        idx = np.searchsorted(peak_times, grid)
        stimulus = (
            torch.tensor(
                self.data["stim"][idx],
                dtype=torch.float32,
            )
            / self.stimulus_std
        )

        spikes_binned = []
        # spikes_embedded = []
        for k, n in enumerate(self.neurons):
            spike_times = self.data["SpikeTimes"][n]
            hist, _ = np.histogram(spike_times, bins=np.concat([grid, [t_end]]))
            spikes_binned.append(torch.tensor(hist, dtype=torch.uint8))

        t_seconds = torch.tensor(
            (grid - t_start) * SECONDS_PER_UNIT, dtype=torch.float32
        )
        return (
            t_seconds,
            stimulus,
            torch.stack(spikes_binned, dim=-1),
            # , torch.cat(spikes_embedded, dim=0)
        )


def collate_time_series(batch):
    # first unzip the batch
    t_batch, s_batch, x_batch = zip(*batch)

    # Find the minimum sequence length
    min_len = min(s.shape[0] for s in s_batch)

    t_truncated = torch.stack([t[:min_len] for t in t_batch], dim=1)
    s_truncated = torch.stack(
        [s[:min_len] for s in s_batch], dim=1
    )  # (min_len, batch_size, ...)
    x_truncated = torch.stack(
        [x[:min_len] for x in x_batch], dim=1
    )  # (min_len, batch_size, ...)

    return t_truncated, s_truncated, x_truncated  # , list(x_embedded)


class StochasticHarmonicOscillator(nn.Module):
    noise_type = "scalar"
    sde_type = "ito"

    def __init__(self, damping_coefficient=DAMPING_COEFFICIENT, tau=TAU):
        super().__init__()

        self.A = (
            torch.tensor(
                [[0.0, 1.0], [-1 / (4 * damping_coefficient**2), -1.0]],
                dtype=torch.float32,
            )
            / tau
        )

        self.B = torch.tensor(
            [[0.0, 1.0 / (np.sqrt(2.0) * damping_coefficient)]], dtype=torch.float32
        ) / np.sqrt(tau)

    # Drift
    def f(self, t, y):
        return y @ self.A.T

    # Diffusion
    def g(self, t, y):
        return self.B.expand(y.size(0), 2).unsqueeze(-1)


class ConditionalSpikeRNN(nn.Module):
    def __init__(
        self, n_neurons: int, hidden_size: int, num_layers: int = 1, kernel_size=10
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.n_neurons = n_neurons

        # initial convolution layer over input
        self.conv = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size - 1)

        # RNN layer (stimulus dimension is 1)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=False)
        self.h_0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))

        # Linear layer to output spike prob for each neuron
        self.output_layer = nn.Linear(hidden_size, n_neurons)

    def forward(self, s: torch.Tensor, x: torch.Tensor):
        seq_len, batch_size, n_neurons = x.size()

        if seq_len != s.size(0):
            raise ValueError(f"Sequence lengths do not match.")

        if n_neurons != self.n_neurons:
            raise ValueError(
                f"Wrong shape of x: {x.size()}. Expected last dimension {self.n_neurons}."
            )

        if s.ndim == 2:
            s = s.unsqueeze(-1)  # (seq_len, batch_size, 1)

        # shift right
        # x = x.roll(1, 0)  # (seq_len, batch_size, n_neurons)
        # x[0, :, :] = 0.0

        # first convolution layer
        x = self.conv(s.permute((1, 2, 0)))[
            :, :, :seq_len
        ]  # (batch_size, hidden_size, seq_len)

        # Expand h_0 to match batch size
        h_0 = self.h_0.expand(-1, batch_size, -1).contiguous()

        # Forward through RNN
        x, _ = self.rnn(x.permute((2, 0, 1)), h_0)

        output = self.output_layer(x)  # (seq_len, batch_size, n_neurons)

        # returns the intensities
        return output


class ConditionalSpikeCNN(nn.Module):
    def __init__(
        self, n_neurons: int, hidden_size: int, num_layers: int = 1, kernel_size=10
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.n_neurons = n_neurons

        layers = []
        for i in range(num_layers):
            in_size = (1 + n_neurons) if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_size, hidden_size, kernel_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

        self.conv_net = nn.Sequential(
            nn.ZeroPad1d(((kernel_size - 1) * num_layers, 0)),  # needed for causality
            *layers,
            # 1x1 convolution layer to output spike intensity for each neuron
            nn.Conv1d(hidden_size, n_neurons, 1),
        )

    def forward(self, s: torch.Tensor, x: torch.Tensor):
        seq_len, batch_size, n_neurons = x.size()

        if seq_len != s.size(0):
            raise ValueError(f"Sequence lengths do not match.")

        if n_neurons != self.n_neurons:
            raise ValueError(
                f"Wrong shape of x: {x.size()}. Expected last dimension {self.n_neurons}."
            )

        if s.ndim == 2:
            s = s.transpose(0, 1).unsqueeze(1)  # (batch_size, 1, seq_len)
        else:
            raise ValueError(f"Wrong shape of s: {s.shape}")

        # # shift right
        x = x.roll(1, 0)  # (seq_len, batch_size, n_neurons)
        x[0, :, :] = 0.0

        output = self.conv_net(
            torch.cat([s, x.permute((1, 2, 0))], dim=1)
        )  # (batch_size, n_neurons, seq_len)
        # output = self.conv_net(s) # (batch_size, n_neurons, seq_len)

        # returns the log-intensities (seq_len, batch_size, n_neurons)
        return output.permute((2, 0, 1))

    @torch.no_grad()
    def sample(self, s: torch.Tensor):
        seq_len, batch_size = s.size()

        x = torch.zeros((seq_len, batch_size, self.n_neurons))
        for t in range(seq_len):
            log_intensities = self.forward(s[: t + 1], x[: t + 1]).clamp(-10, 10)
            x[t] = torch.poisson(log_intensities[-1].exp())

        return x


class SpikeModel(L.LightningModule):
    def __init__(
        self,
        n_neurons: int,
        hidden_size: int,
        num_layers: int = 1,
        kernel_size=None,
        model_type="RNN",
    ):
        super().__init__()
        self.save_hyperparameters()
        if model_type == "RNN":
            if kernel_size is not None:
                raise ValueError("cannot set kernel size for RNN.")
            self.net = ConditionalSpikeRNN(n_neurons, hidden_size, num_layers)
        elif model_type == "CNN":
            self.net = ConditionalSpikeCNN(
                n_neurons, hidden_size, num_layers, kernel_size=kernel_size
            )
        else:
            raise ValueError(f"unsupported model type {model_type}")
        self.loss_fn = nn.PoissonNLLLoss(log_input=True)

    def training_step(self, batch, batch_idx):
        _, s, x = batch

        log_intensity = self.net(s, x)
        train_loss = self.loss_fn(log_intensity, x)
        self.log("train_loss", train_loss, prog_bar=True, batch_size=x.size(1))
        return train_loss

    def validation_step(self, batch, batch_idx):
        _, s, x = batch
        log_intensity = self.net(s, x)

        val_loss = self.loss_fn(log_intensity, x)

        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True, batch_size=x.size(1))
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-1, total_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def plot_network_performance(model, train_dataset, validation_dataset, out_path=None):
    t_train, s_train, x_train = collate_time_series([v for v in train_dataset])
    t_val, s_val, x_val = collate_time_series([v for v in validation_dataset])
    model.net.eval()

    fig, axs = plt.subplots(
        1 + model.net.n_neurons, 2, width_ratios=[0.7, 0.3], constrained_layout=True
    )

    for ax, t, s, title in zip(
        axs[0, :], [t_train, t_val], [s_train, s_val], ["training", "validation"]
    ):
        ax.plot(t[:, 0], s[:, 0])
        ax.set_title(title)

    for j, t, s, x in zip([0, 1], [t_train, t_val], [s_train, s_val], [x_train, x_val]):
        with torch.no_grad():
            log_intensity = model.net(s, x)

        intensity = log_intensity.exp().mean(1)
        for i in range(model.net.n_neurons):
            ax = axs[i + 1, j]
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(-0.2, 0.2)
            ax.plot(t[:, 0], intensity[:, i], linewidth=1)
            ax.plot(t[:, 0], -x[:, :, i].mean(1, dtype=torch.float32), linewidth=1)

    if out_path is not None:
        fig.savefig(out_path)
    return fig


def train_model(dataset_path, neurons=[0]):
    data = scipy.io.loadmat(dataset_path, squeeze_me=True)
    fraction = 0.7  # fraction of data to use for training
    intervals = np.column_stack((data["rep_begin_time"], data["rep_end_time"]))
    c = (intervals[:, 0] + fraction * (intervals[:, 1] - intervals[:, 0])).astype(int)
    train_intervals = np.column_stack((intervals[:, 0], c))
    validation_intervals = np.column_stack((c + 1, intervals[:, 1]))
    train_dataset = NeuronDataSet(dataset_path, neurons, train_intervals)
    validation_dataset = NeuronDataSet(dataset_path, neurons, validation_intervals)

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_time_series
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
        collate_fn=collate_time_series,
    )
    spike_model = SpikeModel(
        len(neurons), len(neurons) * 4, 2, model_type="CNN", kernel_size=20
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=os.environ.get("PBS_JOBID", "neuron"),
        version=f"neuron_{neurons[0]}:{neurons[0] + len(neurons)}",
    )

    trainer = L.Trainer(
        max_epochs=50, logger=logger, callbacks=[lr_monitor], log_every_n_steps=10
    )
    trainer.fit(
        model=spike_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    plot_network_performance(spike_model, train_dataset, validation_dataset)

    return spike_model


def log_probability(model, s, x):
    model.eval()
    log_intensity = model(s, x)
    return -F.poisson_nll_loss(
        log_intensity, x, log_input=True, full=True, reduction="none"
    ).sum(-1)


def pws_estimate(model, t, N, M):
    stoch_osc = StochasticHarmonicOscillator()
    y0 = torch.zeros(N, 2)
    y = torchsde.sdeint(stoch_osc, y0, t, dt=1 / 60)
    s_samples = y[:, :, 0]
    x_samples = model.net.sample(s_samples)

    def pws_samples():
        log_conditional = log_probability(model.net, s_samples, x_samples).cumsum(0)
        for i in range(N):
            y0 = torch.zeros(M, 2)
            y = torchsde.sdeint(stoch_osc, y0, t, dt=1 / 60)
            s = y[:, :, 0]
            logp = log_probability(
                model.net, s, x_samples[:, [i], :].repeat(1, M, 1)
            ).cumsum(0)
            log_marginal = logp.logsumexp(1) - np.log(M)
            yield torch.stack((log_conditional[:, i], log_marginal))

    with torch.no_grad():
        samples = torch.stack(list(pws_samples()))

    return samples


if __name__ == "__main__":
    args = parse_args()

    output_file = args.output_filename

    torch.set_num_threads(args.threads)
    spike_model = train_model(args.dataset_path, args.neurons)

    t = torch.arange(100) * BIN_WIDTH * SECONDS_PER_UNIT

    pws_result = pws_estimate(spike_model, t, 400, 2048)
    mi = pws_result[:, 0] - pws_result[:, 1]

    results = {
        "args": {
            "neurons": args.neurons,
            "N": 400,
            "M": 2048,
        },
        "pws_result": {
            "t": t.tolist(),
            "log_conditional": pws_result[:, 0].tolist(),
            "log_marginal": pws_result[:, 1].tolist(),
            "mutual_information": mi.tolist(),
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

