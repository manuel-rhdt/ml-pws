import numpy as np
import polars as pl
import torch

from scipy import linalg

from absl import logging

import argparse
from pathlib import Path
import json
import datetime
import socket


def logdet_cholesky(matrix):
    L = linalg.cholesky(matrix, lower=True)
    return 2.0 * np.sum(np.log(np.diag(L)))


def gaussian_mi(s, x):
    s = s - np.mean(s)
    x = x - np.mean(x)
    L = s.shape[-1]
    data = np.concat((s.T, x.T), axis=0)
    cov_z = np.cov(data)
    cov_ss = cov_z[:L, :L]
    cov_xx = cov_z[L:, L:]

    logdet_ss = logdet_cholesky(cov_ss)
    logdet_xx = logdet_cholesky(cov_xx)
    logdet_z = logdet_cholesky(cov_z)

    return 0.5 * (logdet_ss + logdet_xx - logdet_z)


def pws_simulate(
    s,
    x,
    val_s,
    val_x,
    input_cell,
    variational_cell,
    output_cell,
    *,
    optimize_forward=True,
    args=None,
    rngs=None,
):
    from jax import random
    from ml_pws.models.smc_estimator import (
        SMCEstimator,
        train_forward_model,
        train_backward_model,
        estimate_mi,
    )

    smc_estimator = SMCEstimator(input_cell, variational_cell, output_cell, rngs=rngs)

    k1, k2 = random.split(random.key(args.seed))
    if optimize_forward:
        log = train_forward_model(
            smc_estimator, s, x, num_epochs=args.forward_epochs, rng_key=k1
        )
        log.write_csv(args.o / "forward_training_log.csv")
    log = train_backward_model(smc_estimator, x, num_epochs=args.backward_epochs)
    log.write_csv(args.o / "backward_training_log.csv")

    result = estimate_mi(smc_estimator, val_s, val_x)

    # get summary statistics
    df = result.group_by("step", maintain_order=True).agg(
        (pl.col("log_conditional") - pl.col("log_marginal")).mean().alias("mean"),
        pl.col("ess").mean(),
        pl.col("ess").count().alias("count"),
        pl.col("resample_flags").mean().alias("resample_fraction"),
    )

    return df


def run_simulation(s, x, val_s, val_x, args, seed=0):
    if args.estimator == "ML-PWS":
        from jax import numpy as jnp
        from flax import nnx
        from ml_pws.models.ar_model import ARCell
        from ml_pws.models.predictive_rnn import PredictiveCell
        from ml_pws.models.variational_rnn import FlowRNNCell

        rngs = nnx.Rngs(seed)
        input_model = ARCell(
            coefficients=args.ar_coeffs, noise_std=args.ar_std, rngs=rngs
        )
        variational_model = FlowRNNCell(16, rngs=rngs)
        output_model = PredictiveCell(args.hidden_features, rngs=rngs)
        return pws_simulate(
            jnp.array(s),
            jnp.array(x),
            jnp.array(val_s),
            jnp.array(val_x),
            input_model,
            variational_model,
            output_model,
            optimize_forward=True,
            args=args,
            rngs=rngs,
        )
    elif args.estimator == "PWS":
        import jax.numpy as jnp
        from flax import nnx
        from ml_pws.models.ar_model import ARCell
        from ml_pws.models.variational_rnn import FlowRNNCell
        from ml_pws.models.logistic_model import LogisticCell

        rngs = nnx.Rngs(seed)
        input_model = ARCell(
            coefficients=args.ar_coeffs, noise_std=args.ar_std, rngs=rngs
        )
        variational_model = FlowRNNCell(16, rngs=rngs)
        output_model = LogisticCell(
            gain=args.gain, decay=args.decay, noise=args.output_noise
        )
        return pws_simulate(
            jnp.array(s),
            jnp.array(x),
            jnp.array(val_s),
            jnp.array(val_x),
            input_model,
            variational_model,
            output_model,
            optimize_forward=False,
            args=args,
            rngs=rngs,
        )
    elif args.estimator == "DoE":
        import lightning as L
        from torch.utils.data import DataLoader, TensorDataset
        from ml_pws.models.gaussian_rnn import DoeEstimator

        train_dataset = TensorDataset(s, x)
        validation_dataset = TensorDataset(val_s, val_x)

        train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
        doe_estimator = DoeEstimator(1, args.hidden_features, 4)
        trainer = L.Trainer(max_epochs=50, default_root_dir=args.o)
        trainer.fit(model=doe_estimator, train_dataloaders=train_loader)

        mi = doe_estimator.estimate_mutual_information(*validation_dataset.tensors)

        return pl.DataFrame(
            {
                "step": np.arange(mi.shape[-1]) + 1,
                "mean": mi.cumsum(1).mean(0).numpy(),
                "std": mi.cumsum(1).std(0).numpy(),
                "stderr": mi.cumsum(1).std(0).numpy() / np.sqrt(mi.shape[0]),
                "count": mi.shape[0],
            }
        )
    elif args.estimator == "InfoNCE":
        from torch.utils.data import DataLoader, TensorDataset
        import lightning as L
        from ml_pws.models.contrastive_mi import ContrastiveEstimator

        train_dataset = TensorDataset(s, x)
        validation_dataset = TensorDataset(val_s, val_x)

        lengths = np.concat([np.arange(1, 10, 2), np.arange(10, args.length + 1, 10)])
        mi = np.zeros(len(lengths))

        for i, length in enumerate(lengths):
            train_loader = DataLoader(
                TensorDataset(*train_dataset[:, :length]), batch_size=50, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(*validation_dataset[:, :length]), batch_size=100
            )

            contrastive_estimator = ContrastiveEstimator(1, args.hidden_features, 4)
            trainer = L.Trainer(max_epochs=10, default_root_dir=args.o)
            trainer.fit(
                model=contrastive_estimator,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

            size = args.num_pairs
            test_data = validation_dataset[:size, :length]
            with torch.no_grad():
                mi[i] = np.log(size) - contrastive_estimator(*test_data)

        return pl.DataFrame({"step": lengths, "mean": mi, "count": args.num_pairs})
    elif args.estimator == "Gaussian":
        mi = gaussian_mi(val_s.numpy(), val_x.numpy())
        return pl.DataFrame(
            {"step": [val_s.shape[-1]], "mean": [mi], "count": args.num_pairs}
        )


def main():
    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run a ML-PWS with AR(p) input and nonlinear output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ar_coeffs",
        nargs="+",
        type=float,
        default=[0.5, -0.3, 0.2],
        help="Autoregressive coefficients for the AR model.",
    )
    parser.add_argument(
        "--ar_std",
        type=float,
        default=0.2,
        help="Noise strength of AR model.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=10.0,
        help="Gain of output model.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.2,
        help="Decay of output model.",
    )
    parser.add_argument(
        "--output_noise",
        type=float,
        default=0.2,
        help="Output noise strength.",
    )
    parser.add_argument(
        "-o",
        type=Path,
        default=Path.cwd(),
        help="Output directory",
    )
    parser.add_argument(
        "--forward_epochs",
        type=int,
        default=100,
        help="Number of training epochs for the forward model.",
    )
    parser.add_argument(
        "--backward_epochs",
        type=int,
        default=100,
        help="Number of training epochs for the backward model.",
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        default=64,
        help="Number of hidden features in the RNN cells.",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=1000,
        help="Number of data pairs to generate for training.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Length of each training data sequence.",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument("--estimator", choices=["ML-PWS", "PWS", "DoE", "InfoNCE", "Gaussian"])

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.o.mkdir(parents=True)
    result_path = args.o / "result.csv"

    parameters = {
        "run_info": {
            "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
        },
        "ar_coeffs": args.ar_coeffs,
        "ar_std": args.ar_std,
        "gain": args.gain,
        "decay": args.decay,
        "output_noise": args.output_noise,
        "output_directory": str(args.o.resolve()),
        "result_path": str(result_path.resolve()),
        "forward_epochs": args.forward_epochs,
        "backward_epochs": args.backward_epochs,
        "hidden_features": args.hidden_features,
        "num_pairs": args.num_pairs,
        "length": args.length,
        "seed": args.seed,
        "estimator": args.estimator,
    }

    parameter_file_path = args.o / "parameters.json"
    with open(parameter_file_path, "w") as f:
        json.dump(parameters, f, indent=4)

    rng = np.random.default_rng(args.seed)

    from ml_pws.data.nonlinear_dataset import generate_nonlinear_data

    s, x = generate_nonlinear_data(
        num_pairs=args.num_pairs,
        length=args.length,
        coeffs=args.ar_coeffs,
        ar_noise=args.ar_std,
        gain=args.gain,
        decay=args.decay,
        noise=args.output_noise,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    val_s, val_x = generate_nonlinear_data(
        num_pairs=args.num_pairs,
        length=args.length,
        coeffs=args.ar_coeffs,
        ar_noise=args.ar_std,
        gain=args.gain,
        decay=args.decay,
        noise=args.output_noise,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    input_data = {"train_s": s, "train_x": x, "val_s": val_s, "val_x": val_x}
    torch.save(input_data, args.o / "input_data.pth")

    result = run_simulation(
        s,
        x,
        val_s,
        val_x,
        args,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    result.write_csv(args.o / "result.csv")


if __name__ == "__main__":
    main()
