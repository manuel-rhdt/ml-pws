"""
Generate synthetic time series data using AR and logistic models.

This script generates two types of time series data:
1. A single trajectory using an AR(3) model
2. Multiple trajectories using a logistic model with different gain parameters

The generated data is saved to CSV files in the data directory:
- example_traj_s.csv: Contains the AR model trajectory
- example_traj_x.csv: Contains the logistic model trajectories with summary statistics

Parameters:
    LENGTH (int): Length of time series (50 points)
    AR parameters: [0.5, -0.3, 0.2] with noise variance 1.0
    Logistic parameters:
        - gains: [0.1, 1.0, 10.0]
        - decay: 0.2
        - noise: 0.2
"""

import numpy as np
import polars as pl

import jax.numpy as jnp
from jax import random
from flax import nnx

from ml_pws.models.ar_model import ARModel
from ml_pws.models.logistic_model import LogisticModel

import sys

LENGTH = 50

key = random.PRNGKey(42)
s_key, x_key = random.split(key)

input_model = ARModel(jnp.array([0.5, -0.3, 0.2]), 1.0, rngs=nnx.Rngs(s_key))
_, s = input_model(jnp.zeros((1, LENGTH)), generate=True)

test_s = s[1]
time = np.arange(1,51)

pl.DataFrame({'time': time, 's': np.array(test_s)}).write_csv("data/example_traj_s.csv")

data = []
for gain in [0.1, 1.0, 10.0]:
    var_model = LogisticModel(gain=gain, decay=0.2, noise=0.2)
    _, test_x = var_model(jnp.tile(test_s, (10_000, 1)), jnp.empty((10_000, LENGTH)), generate=True, rngs=nnx.Rngs(generate=x_key))
    mean = jnp.mean(test_x, axis=0)
    q10 = jnp.percentile(test_x, 10, axis=0)
    q90 = jnp.percentile(test_x, 90, axis=0)
    data.append(pl.DataFrame({
        'gain': gain,
        'time': time,
        'mean': np.array(mean),
        'q10': np.array(q10),
        'q90': np.array(q90)
    }))

pl.concat(data).write_csv("data/example_traj_x.csv")