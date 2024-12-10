
import argparse
from pathlib import Path
import torch
from torch import nn, optim
import numpy as np

import rnn_sequence_model as model
from rnn_sequence_model import get_data, fit

parser = argparse.ArgumentParser(description='Stochastic sequence model')
parser.add_argument('-i', '--input', type=Path, required=True,
                    help='path to input file containing training data (default %(default)s)')
parser.add_argument('-o', '--output', type=Path, required=True,
                    help='path to store output file (default %(default)s)')
parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'],
                    help='Which model type to use (default %(default)s).')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default %(default)g)')
parser.add_argument('--hidden-size', type=int, default=16)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

corpus = torch.load(args.input, device)

# build the model
model = model.SequenceModel(args.hidden_size, args.layers, args.model).to(device)

loss_func = nn.GaussianNLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_dl, valid_dl = get_data(corpus, bs=32)
try:
    fit(args.epochs, model, loss_func, optimizer, train_dl, valid_dl)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

torch.save(model, args.output)

print(f'saved model to {args.output}')

