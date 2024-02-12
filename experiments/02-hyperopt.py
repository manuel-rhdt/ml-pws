from importlib import import_module
from itertools import product

import torch
from torch.nn import functional as F
from hyperopt import fmin, hp, tpe
from sacred.observers import MongoObserver, FileStorageObserver

from src.models.rnn_sequence_model import generate

# compute true covariance

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

kappa = 0.5
lamda = 1.0
rho = 1.0
mu = 1.0

sigma_ss = kappa / lamda
sigma_sx = rho * sigma_ss / (lamda + mu)
sigma_xx = rho / mu * (sigma_ss + sigma_sx)

def exp2(x, decay, scale):
    return scale * (-decay * x).exp()

def exprel(x):
    return torch.where(x.abs() > 1e-15, torch.special.expm1(x) / x, 1.0)

def cov_ss(t):
    return exp2(t, lamda, sigma_ss)

def cov_sx(t):
    scale1 = rho * sigma_ss * t * exprel((lamda - mu) * t)
    return exp2(t, mu, sigma_sx) + exp2(t, lamda, scale1)

def cov_xs(t):
    return exp2(t, lamda, sigma_sx)

def cov_xx(t):
    scale1 = rho * sigma_sx * t * exprel((lamda - mu) * t)
    return exp2(t, lamda, scale1) + exp2(t, mu, sigma_xx)

t = torch.linspace(-10.0, 10.0, steps=201)

true_cov = torch.stack((
    torch.stack((
        cov_ss(t.abs()),
        torch.where(t >= 0, cov_sx(t), cov_xs(-t)),
    )),
    torch.stack((
        torch.where(t >= 0, cov_xs(t), cov_sx(-t)),
        cov_xx(t.abs()),
    ))
))

corpus = torch.load('data/gaussian_data.pt', device)
train_data = corpus['training']
val_data = corpus['validation']

def cross_cov(data, transpose=False):
    if transpose:
        data = data.transpose(0, 1)
    N, L = data.shape[1:]
    result = F.conv1d(data, data, padding=L) / N / L
    return torch.arange(-L,L+1), result.transpose(0, 1).cpu()

def get_mse_loss(model):
    pred, _ = generate(model, val_data[:,0,:])
    lags, cov = cross_cov(torch.stack((val_data[:,0,:], pred)))
    return F.mse_loss(cov, true_cov).item()

# SETUP EXPERIMENT

ex = import_module('01-model-size').ex

ex.observers.append(MongoObserver())
ex.observers.append(FileStorageObserver('runs'))

import math

space = (
    hp.choice('model', ('RNN', 'GRU', 'LSTM')),
    hp.qloguniform('hidden_size', math.log(8), math.log(256), 8),
    hp.choice('layers', [1,2,3,4])
)

def run_experiment(conf):
    model, hidden_size, layers = conf
    run = ex.run(config_updates={
        'model': model,
        'hidden_size': max(int(hidden_size), 1),
        'layers': layers,
        'epochs': 500,
    })
    model = run.result
    return get_mse_loss(model)

best = fmin(
    fn=run_experiment,
    space=space,
    algo=tpe.suggest,
    max_evals=100
)

print(best)
