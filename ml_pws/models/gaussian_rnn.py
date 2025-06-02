import torch
import torch.nn as nn

import lightning as L


class GaussianRNN(nn.Module):
    def __init__(self, features: int, hidden_size: int, num_layers: int = 1):
        super(GaussianRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(features, hidden_size, num_layers, batch_first=True)

        # Linear layer to output mean and log-variance
        self.output_layer = nn.Linear(
            hidden_size, 2 * features
        )  # 2 for mean and log-variance

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size()

        # shift right
        x = x.roll(1, 1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        x[:, 0, :] = 0.0

        # Forward through RNN
        rnn_out, hidden = self.rnn(x)

        # Output mean and log-variance
        output = self.output_layer(rnn_out)  # (batch_size, seq_len, 2)
        mean, log_var = torch.split(output, 1, -1)  # (batch_size, seq_len, 1)

        return mean.squeeze(-1), log_var.squeeze(-1)

    def log_p(self, x: torch.Tensor):
        mean, log_var = self(x)
        var = torch.exp(log_var)
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)
        return log_prob


class ConditionalGaussianRNN(GaussianRNN):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__(input_features + output_features, hidden_size, num_layers)
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, s: torch.Tensor, x: torch.Tensor):
        s = s.unsqueeze(-1)

        # shift right
        x = x.roll(1, 1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        x[:, 0, :] = 0.0

        input_data = torch.cat([s, x], dim=2)

        # Forward through RNN
        rnn_out, hidden = self.rnn(input_data)

        # Output mean and log-variance
        output = self.output_layer(rnn_out)  # (batch_size, seq_len, 2)
        mean, log_var = torch.split(output, 1, -1)  # (batch_size, seq_len, 1)

        return mean.squeeze(-1), log_var.squeeze(-1)

    def log_p(self, s: torch.Tensor, x: torch.Tensor):
        mean, log_var = self(s, x)
        var = torch.exp(log_var)
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (x - mean) ** 2 / var)
        return log_prob


class MLPWSEstimator(L.LightningModule):
    def __init__(self, input_model, forward_model, backward_model):
        self.input_model = input_model
        self.forward_model = forward_model
        self.backward_model = backward_model

        # We need multiple optimizers for forward and backward model.
        # This is not possible with automatic optimization.
        self.automatic_optimization = False

    def elbo(self, s: torch.tensor, x: torch.tensor):
        logq_s = self.backward_model(s, x)
        logp_s = self.input_model(s)
        logp_x_given_s = self.forward_model(s, x)
        return logp_s + logp_x_given_s - logq_s

    def training_step(self, batch, batch_idx):
        s, x = batch

        opt_forward, opt_backward = self.optimizers()

        # forward step
        opt_forward.zero_grad()
        forward_loss = -self.forward_model(s, x)
        self.log("forward_loss", forward_loss)
        self.manual_backward(forward_loss)
        opt_forward.step()

        # backward step
        opt_backward.zero_grad()
        preds = self.backward_model.rsample(x)
        backward_loss = self.elbo(preds, x).mean()
        self.log("backward_loss", backward_loss)
        self.manual_backward(backward_loss)
        opt_backward.step()

    def configure_optimizers(self):
        forward_opt = torch.optim.Adam(self.forward_model.parameters(), lr=1e-3)
        backward_opt = torch.optim.Adam(self.backward_model.parameters(), lr=1e-3)
        return forward_opt, backward_opt


class DoeEstimator(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, learning_rate=1e-3):
        super().__init__()
        self.lr = learning_rate
        self.conditional_model = ConditionalGaussianRNN(
            input_dim, input_dim, hidden_dim, num_layers
        )
        self.marginal_model = GaussianRNN(input_dim, hidden_dim, num_layers)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        s, x = batch
        criterion = nn.GaussianNLLLoss()

        # conditional model
        mean, log_var = self.conditional_model(s, x)
        cond_loss = criterion(mean, x, log_var.exp())
        self.log("cond_loss", cond_loss)

        # marginal model
        mean, log_var = self.marginal_model(x)
        marg_loss = criterion(mean, x, log_var.exp())
        self.log("marg_loss", marg_loss)

        loss = cond_loss + marg_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_index):
        # training_step defines the train loop.
        s, x = batch
        criterion = nn.GaussianNLLLoss()

        # conditional model
        mean, log_var = self.conditional_model(s, x)
        cond_loss = criterion(mean, x, log_var.exp())

        # marginal model
        mean, log_var = self.marginal_model(x)
        marg_loss = criterion(mean, x, log_var.exp())

        self.log("val_loss", cond_loss + marg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def estimate_mutual_information(self, s, x):
        self.eval()
        with torch.no_grad():
            mi = self.conditional_model.log_p(s, x) - self.marginal_model.log_p(x)

        return mi
