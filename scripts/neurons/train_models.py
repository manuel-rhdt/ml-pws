from dataclasses import dataclass
import os
from pathlib import Path
import sys
import json

import numpy as np
import scipy

import torch

# Optional MPI support - fall back to single process if not available
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _has_mpi = _size > 1  # Only use MPI if actually running with multiple processes
except ImportError:
    _comm = None
    _rank = 0
    _size = 1
    _has_mpi = False


def get_rank():
    """Get current MPI rank (0 if MPI not available)."""
    return _rank


def get_size():
    """Get total number of MPI processes (1 if MPI not available)."""
    return _size


def barrier():
    """MPI barrier (no-op if MPI not available)."""
    if _has_mpi and _comm is not None:
        _comm.Barrier()
from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt

import shutil

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ml_pws.models.neuron_models import SpikeModel


@dataclass
class ModelSpec:
    neurons: list[int]
    name: str
    output_dir: Path
    hidden_size: int = 40
    num_layers: int = 4
    kernel_size: int = 20


with open(sys.argv[1]) as f:
    CONFIG = json.load(f)

BIN_WIDTH = 100 # 10 millisecond time discretization (in 0.1 ms units)
SECONDS_PER_UNIT = 1e-4 # 0.1 ms per time unit in the dataset

# training interval parameters
INTERVAL_LENGTH = int(4 / SECONDS_PER_UNIT)
INTERVAL_OVERLAP = INTERVAL_LENGTH // 2


class NeuronDataSet(Dataset):
    def __init__(
        self, filename, neurons=[0], intervals=None, delta_t=BIN_WIDTH, cnn=False
    ):
        super().__init__()
        self.data = scipy.io.loadmat(filename, squeeze_me=True, appendmat=False)
        self.cnn = cnn
        self.neurons = neurons
        if intervals is None:
            self.intervals = np.stack(
                (self.data["rep_begin_time"], self.data["rep_end_time"]), axis=-1
            )
        else:
            self.intervals = np.array(intervals).reshape((-1, 2))
        self.delta_t = delta_t
        self.stimulus_std = float(np.std(self.data["stim"]))

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
        for n in self.neurons:
            spike_times = self.data["SpikeTimes"][n]
            hist, _ = np.histogram(spike_times, bins=np.concat([grid, [t_end]]))
            spikes_binned.append(torch.tensor(hist, dtype=torch.uint8))

        t_seconds = torch.tensor(
            (grid - t_start) * SECONDS_PER_UNIT, dtype=torch.float32
        )
        dim = -1 if not self.cnn else 0
        return (
            t_seconds,
            stimulus,
            torch.stack(spikes_binned, dim=dim),
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


def plot_network_performance(model, validation_dataset, out_path=None):
    t_val, s_val, x_val = collate_time_series([v for v in validation_dataset])
    model_output = model.net.sample(s_val)

    fig, axs = plt.subplots(1 + model.net.n_neurons, constrained_layout=True)

    axs[0].plot(t_val[:, 0], s_val[:, 0])
    axs[0].set_title("Validation")
    axs[0].set_yticklabels([])

    for i in range(model.net.n_neurons):
        ax = axs[i + 1]
        # ax.set_ylabel(f'$n_{{{i+1}}}$')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(-0.5, 0.5)
        ax.plot(t_val[:, 0], model_output[:, :, i].mean(1), linewidth=1)
        ax.plot(t_val[:, 0], -x_val[:, :, i].mean(1, dtype=torch.float32), linewidth=1)

    if out_path is not None:
        fig.savefig(out_path)
    return fig


def partition(x1, x2, length, overlap=0):
    """
    Splits the interval [x1, x2] into subintervals of length `length` with configurable overlap between the intervals.

    Returns `floor((x2 - x1 - overlap) / (length - overlap))` intervals
    """
    if x1 >= x2:
        raise ValueError("x1 must be less than x2.")
    if length <= 0:
        raise ValueError("length must be a positive number.")
    if overlap < 0 or overlap >= length:
        raise ValueError("overlap must be in range [0, length).")

    step = length - overlap
    num_intervals = int((x2 - x1 - overlap) // step)

    start_points = np.arange(num_intervals) * step + x1
    end_points = start_points + length

    intervals = np.column_stack((start_points, end_points))

    return intervals


# We want to use training intervals that avoid the stimulus repetitions since
# those are reserved for validation. Furthermore, we want overlapping intervals to
# increase the amount of training data.
def get_training_intervals(data, interval_length, overlap=0):
    # initial partition
    yield partition(
        100 / SECONDS_PER_UNIT, data["rep_begin_time"][0], interval_length, overlap
    )

    for x1, x2 in zip(data["rep_end_time"][:-1], data["rep_begin_time"][1:]):
        yield partition(x1, x2, interval_length, overlap)

    # final partition
    yield partition(
        data["rep_end_time"][-1], 8000 / SECONDS_PER_UNIT, interval_length, overlap
    )


def train_model(dataset_path, model_spec: ModelSpec):
    neurons = model_spec.neurons
    name = model_spec.name
    output_dir = model_spec.output_dir

    data = scipy.io.loadmat(dataset_path, squeeze_me=True, appendmat=False)
    training_intervals = np.concatenate(
        [x for x in get_training_intervals(data, INTERVAL_LENGTH, INTERVAL_OVERLAP)]
    )
    validation_intervals = np.column_stack(
        (data["rep_begin_time"], data["rep_end_time"])
    )
    train_dataset = NeuronDataSet(dataset_path, neurons, training_intervals)
    validation_dataset = NeuronDataSet(dataset_path, neurons, validation_intervals)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_time_series,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
        collate_fn=collate_time_series,
    )
    spike_model = SpikeModel(
        len(neurons),
        model_spec.hidden_size,
        model_spec.num_layers,
        model_type="CNN",
        kernel_size=model_spec.kernel_size,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )
    logger = CSVLogger(
        save_dir=output_dir.parent, name=name, version=""
    )

    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=10,
        logger=logger,
        enable_progress_bar=False,
    )
    trainer.fit(
        model=spike_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # training finished
    spike_model.eval()

    plot_network_performance(
        spike_model, validation_dataset, output_dir / "validation.png"
    )

    # Save best checkpoint to output_dir
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        shutil.copy(best_ckpt_path, output_dir / "best.ckpt")

    return spike_model


if __name__ == "__main__":
    rank = get_rank()
    size = get_size()

    if rank == 0:
        if _has_mpi:
            print(f"Starting training with {size} MPI processes")
        else:
            print("Starting training (single process, MPI not available)")

    dataset_path = CONFIG["dataset_path"]
    models = []
    for spec in CONFIG["models"]:
        spec["output_dir"] = Path(spec["output_dir"])
        models.append(ModelSpec(**spec))

    # Distribute work across MPI ranks (or process all if single process)
    local_models = models[rank::size]

    if rank == 0:
        print(f"Total models: {len(models)}")

    for spec in local_models:
        print(f"Rank {rank}: Training {spec.name}...")
        os.makedirs(spec.output_dir, exist_ok=True)
        model = train_model(dataset_path, spec)
        print(f"Rank {rank}: Completed {spec.name}")

    # Wait for all ranks to complete
    barrier()

    if rank == 0:
        print("All training completed!")
