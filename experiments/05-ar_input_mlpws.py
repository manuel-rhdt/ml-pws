import jax.numpy as jnp
from jax import random

import numpy as np
from flax import nnx
import polars as pl

from absl import logging

import argparse
from pathlib import Path
import json
import datetime
import socket

from ml_pws.data.nonlinear_dataset import generate_nonlinear_data
from ml_pws.models.ar_model import ARModel
from ml_pws.models.logistic_model import LogisticModel
from ml_pws.models.variational_rnn import VariationalRnn
from ml_pws.models.predictive_rnn import PredictiveRnn, ConvolutionalAutoregressiveModel
from ml_pws.models.trainer import Trainer, CombinedModel


def run_simulation(s, x, val_s, val_x, args, ground_truth=False, key=random.key(0)):
    input_model = ARModel(coefficients=args.ar_coeffs, noise_std=args.ar_std)

    rngs = nnx.Rngs(key)

    if ground_truth:
        output_model = LogisticModel(
            gain=args.gain, decay=args.decay, noise=args.output_noise, rngs=rngs
        )
    else:
        # output_model = PredictiveRnn(args.hidden_features, rngs=rngs)
        output_model = ConvolutionalAutoregressiveModel(10, 16, 8, rngs=rngs)

    comb_model = CombinedModel(
        prior=input_model,
        forward=output_model,
        backward=VariationalRnn(args.hidden_features, 16, rngs=rngs),
    )

    k1, k2 = random.split(key)

    trainer = Trainer(comb_model, s, x, args.o / "metrics")
    if not ground_truth:
        trainer.train_forward_model(k1, args.forward_epochs)
    trainer.train_backward_model(k2, args.backward_epochs, subsample=1)

    path_mi, ess = trainer.mutual_information(val_s, val_x)
    path_mi = np.asarray(path_mi)
    ess = np.asarray(ess)

    # get summary statistics
    df = pl.DataFrame(
        {
            "step": 1 + np.arange(path_mi.shape[1]),
            "mean": path_mi.mean(0),
            "std": path_mi.std(0),
            "stderr": path_mi.std(0) / np.sqrt(path_mi.shape[0]),
            "count": path_mi.shape[0],
            "ess": ess.mean(0),
        }
    )

    return df


def main():
    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run a ML-PWS with AR(p) input and nonlinear output."
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
        help="Noise strength of AR model (default 0.2).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=10.0,
        help="Gain of output model (default 10.0).",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.2,
        help="Decay of output model (default 0.2).",
    )
    parser.add_argument(
        "--output_noise",
        type=float,
        default=0.2,
        help="Output noise strength (default 0.2).",
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
        default=500,
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
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument("--ground_truth", action="store_true", default=False)

    args = parser.parse_args()

    key = random.key(args.seed)
    args.o.mkdir(parents=True)
    result_path = args.o / "result.csv"

    parameters = {
        "run_info": {
            "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": socket.gethostname()
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
        "ground_truth": args.ground_truth,
    }

    parameter_file_path = args.o / "parameters.json"
    with open(parameter_file_path, "w") as f:
        json.dump(parameters, f, indent=4)

    k1, k2 = random.split(key)
    seed1, seed2 = random.randint(k1, 2, 0, 2**31 - 1)
    s, x = generate_nonlinear_data(
        num_pairs=args.num_pairs,
        length=args.length,
        coeffs=args.ar_coeffs,
        gain=args.gain,
        decay=args.decay,
        noise=args.output_noise,
        seed=int(seed1),
    )
    val_s, val_x = generate_nonlinear_data(
        num_pairs=args.num_pairs,
        length=args.length,
        coeffs=args.ar_coeffs,
        gain=args.gain,
        decay=args.decay,
        noise=args.output_noise,
        seed=int(seed2),
    )

    result = run_simulation(
        jnp.asarray(s),
        jnp.asarray(x),
        jnp.asarray(val_s),
        jnp.asarray(val_x),
        args,
        ground_truth=args.ground_truth,
        key=k2,
    )

    result.write_csv(args.o / "result.csv")


if __name__ == "__main__":
    main()
