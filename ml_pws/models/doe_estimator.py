import torch
import torch.nn as nn

import lightning as L

class GaussianRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1):
        super(GaussianRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to output mean and log-variance
        self.output_layer = nn.Linear(hidden_size, 2)  # 2 for mean and log-variance

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size()
                
        # shift right
        x = x.roll(1, 1).unsqueeze(-1) # (batch_size, seq_len, 1)
        x[:, 0, :] = 0.0

        # Forward through RNN
        rnn_out, hidden = self.rnn(x)

        # Output mean and log-variance
        output = self.output_layer(rnn_out) # (batch_size, seq_len, 2)
        mean, log_var = torch.split(output, 1, -1) # (batch_size, seq_len, 1)        

        return mean.squeeze(-1), log_var.squeeze(-1)
    
    def log_p(self, x: torch.Tensor):
        mean, log_var = self(x)
        var = torch.exp(log_var) 
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)
        return log_prob
    
class ConditionalGaussianRNN(GaussianRNN):
    def forward(self, s: torch.Tensor, x: torch.Tensor):
        batch_size, seq_len = x.size()
        
        s = s.unsqueeze(-1)

        # shift right
        x = x.roll(1, 1).unsqueeze(-1) # (batch_size, seq_len, 1)
        x[:, 0, :] = 0.0

        # Forward through RNN
        rnn_out, hidden = self.rnn(torch.cat([s, x], dim=2))

        # Output mean and log-variance
        output = self.output_layer(rnn_out) # (batch_size, seq_len, 2)
        mean, log_var = torch.split(output, 1, -1) # (batch_size, seq_len, 1)        

        return mean.squeeze(-1), log_var.squeeze(-1)
    
    def log_p(self, s: torch.Tensor, x: torch.Tensor):
        mean, log_var = self(s, x)
        var = torch.exp(log_var) 
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)
        return log_prob
    

class DoeEstimator(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, learning_rate=1e-3):
        super().__init__()
        self.lr = learning_rate
        self.conditional_model = ConditionalGaussianRNN(2*input_dim, hidden_dim, num_layers)
        self.marginal_model = GaussianRNN(input_dim, hidden_dim, num_layers)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        s, x = batch

        criterion = nn.GaussianNLLLoss()

        # step conditional model
        mean, log_var = self.conditional_model(s, x)
        loss = criterion(mean, x, log_var.exp())

        # step marginal model
        mean, log_var = self.marginal_model(x)
        loss += criterion(mean, x, log_var.exp())

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def estimate_mutual_information(self, s, x):
        self.eval()
        with torch.no_grad():
            mi = self.conditional_model.log_p(s, x) - self.marginal_model.log_p(x)

        return mi