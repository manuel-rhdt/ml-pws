import sys

from sacred import Experiment, Ingredient
import torch
from torch import nn, optim
import numpy as np

from src.models import rnn_sequence_model as stochseq_model
from src.models.rnn_sequence_model import get_data, fit

data_ingredient = Ingredient('dataset')
ex = Experiment('model-size-sweep', ingredients=[data_ingredient])

@data_ingredient.config
def data_config():
    filename = 'data/gaussian_data.pt'
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

@data_ingredient.capture
def load_data(filename, device):
    dataset = torch.load(filename, device)
    return dataset

@ex.config
def my_config():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    lr = 1e-4
    batch_size = 64
    epochs = 400
    model = 'LSTM'
    layers = 1
    hidden_size = 8

@ex.capture
def train_model(model, layers, hidden_size, _run, epochs, device, lr, batch_size):
    model = stochseq_model.SequenceModel(
        hidden_size, 
        layers, 
        model
    ).to(device)

    loss_func = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = load_data()
    train_dl, valid_dl = get_data(dataset, bs=batch_size)

    def logger(epoch, loss):
        _run.log_scalar("training.loss", loss, epoch)
        
    fit(epochs, model, loss_func, optimizer, train_dl, valid_dl, logger)

    return model


@ex.automain
def my_main(_seed):
    torch.manual_seed(_seed)
    train_model()