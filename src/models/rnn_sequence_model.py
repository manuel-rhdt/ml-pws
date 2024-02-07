
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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

def get_data(train_ds, valid_ds, bs=64):
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


