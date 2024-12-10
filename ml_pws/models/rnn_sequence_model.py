
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

# Define the model

class SequenceModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, type="LSTM"):
        super().__init__()
        # create an RNN with 2 input features
        if type == "RNN":
            rnn_type = nn.RNN
        elif type == "GRU":
            rnn_type = nn.GRU
        elif type == "LSTM":
            rnn_type = nn.LSTM
        else:
            raise ValueError(f"expected type one of 'rnn', 'lstm', or 'gru'. Received type={type}")

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = rnn_type(2, hidden_size, num_layers, batch_first=True)
        # create output layer with 2 features (mean, var)
        self.linear = nn.Linear(hidden_size+1, 2)
        
    def init_hidden(self, input):
        batches = input.size(0)
        device = input.device
        h_t = torch.zeros(self.num_layers, batches, self.hidden_size, device=device)
        if isinstance(self.rnn, nn.LSTM):
            c_t = torch.zeros(self.num_layers, batches, self.hidden_size, device=device)
            return h_t, c_t
        return h_t

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        mean, log_var = self.linear(torch.cat((output, input[...,[0]]), dim=-1)).split(1, dim=-1)
        outputs = torch.cat((mean, log_var.exp()), dim=-1)
        return outputs, hidden

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

def default_logger(epoch, loss):
    epoch % 100 == 0 and print(epoch, loss)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, logger=default_logger):
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
        logger(epoch, val_loss)

# input is a (N, L) tensor where N are the number of batches and L is the length
def generate(model, input):
    N, L = input.size()
    model.eval()
    with torch.no_grad():
        # the first x-value we feed into the RNN is zero, consistent with the training eamples
        pred = torch.zeros((N, 1, 1), device=input.device)
        predictions = []
        h_n = []
        hidden = None
        for i in range(L):
            inp = torch.cat((input[:, i].reshape((N, 1, 1)), pred), dim=-1)
            output, hidden = model(inp, hidden)
            pred_mean, pred_var = output.split(1, dim=-1)
            pred = torch.normal(pred_mean, pred_var.sqrt())
            
            predictions.append(pred)
            h_n.append(hidden)
        predictions = torch.cat(predictions, dim=1).squeeze(2)
        
        return predictions, h_n
