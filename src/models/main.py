
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import rnn_sequence_model as model

parser = argparse.ArgumentParser(description='Stochastic sequence model')
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
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

project_dir = Path(__file__).resolve().parents[2]

corpus = torch.load(project_dir / 'data' / 'gaussian_data.pt', device)

def make_dataset(data):
    s_series = data[:,0,:]
    x_series = data[:,1,:]

    # we need to shift the output by one wrt the input
    x_lagged = torch.roll(x_series, 1, dims=-1)
    x_lagged[..., 0] = 0.0

    input = torch.cat((s_series.unsqueeze(-1), x_lagged.unsqueeze(-1)), dim=-1)
    target = x_series.unsqueeze(-1)
    
    return TensorDataset(input, target)

def get_data(corpus, bs=64):
    train_data = corpus["training"]
    valid_data = corpus["validation"]
    train_ds = make_dataset(train_data)
    valid_ds = make_dataset(valid_data)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*4)
    )

def loss_batch(model, loss_func, input, target, opt=None):
    hidden = model.init_hidden(input)
    output, hidden = model(input, hidden)
    mean, var = output.split(1, dim=-1)
    loss = loss_func(mean, target, var)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(input)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            batch_losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.dot(batch_losses, nums) / np.sum(nums)
        
        epoch % 100 == 0 and print(epoch, val_loss)

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

filename = f"stochseq_{args.model}_layers={args.layers}_hidden={args.hidden_size}.pt"
save_path = project_dir / 'models' / filename
torch.save(model, save_path)

print(f'saved model to {save_path}')

