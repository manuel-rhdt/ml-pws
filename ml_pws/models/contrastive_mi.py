import torch
from torch import nn
import torch.nn.functional as F

import lightning as L

class ScoringFunction(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Build convolutional layers
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            layers.append(
                nn.Conv1d(
                    in_channels, hidden_size, kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)

        # Final linear layer to output a scalar
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, s: torch.Tensor, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
        - s: Input tensor of shape (batch_size, seq_len).
        - x: Input tensor of shape (batch_size, seq_len).

        Returns:
        - Output tensor of shape (batch_size,).
        """
        # Permute input to match Conv1D expectations: (batch_size, num_features, seq_len)
        s = s.unsqueeze(-2)
        x = x.unsqueeze(-2)

        x = torch.cat((s, x), dim=-2)

        # Pass through convolutional layers
        x = self.conv_layers(x)  # (batch_size, hidden_size, seq_len)

        # Aggregate features by global average pooling
        x = x.mean(dim=-1)  # (batch_size, hidden_size)

        # Final linear layer
        x = self.output_layer(x).squeeze(-1)  # (batch_size,)
        return x


def info_nce_loss(
    s: torch.Tensor, x: torch.Tensor, score_func, temperature=0.1, neg_examples=20
):
    N, d = s.shape
    M = neg_examples  # number of negative examples

    positive_scores = score_func(s, x)  # shape (N,)

    # for each positive example, randomly choose M negative examples

    # `indices` is a NxN matrix where each row is 0,1,2,...,N-1
    indices = torch.arange(N).unsqueeze(0).expand(N, N)

    # Create a mask to exclude the diagonal (self-comparisons)
    negative_mask = ~torch.eye(N, dtype=torch.bool)

    # Apply the mask to `indices` to get valid negative indices for each example
    # Reshape the masked indices into shape (N, N-1), where each row contains 
    # the negatives for a sample
    negative_indices = indices[negative_mask].view(N, N - 1)

    # Randomly permute the negative indices for each sample and select the 
    # first M negatives
    permutations = torch.rand(*negative_indices.shape).argsort(dim=-1)[:, :M]

    # Use `torch.gather` to randomly select M values per row from `negative_indices`, 
    # based on the permutations
    i_neg = torch.gather(negative_indices, -1, permutations).flatten()
    x_neg = x[i_neg, :]

    s_neg = s.repeat_interleave(M, dim=0)
    negative_scores = score_func(s_neg, x_neg).view(N, M)

    # for each example, the correct target is index 0, and the other indices 
    # 1,...,M are negative examples
    logits = (
        torch.concat([positive_scores.unsqueeze(1), negative_scores], dim=-1)
        / temperature
    )  # shape(N,M+1)
    target = torch.zeros(N, dtype=torch.int64, device=s.device)
    return F.cross_entropy(logits, target)

class ContrastiveEstimator(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, learning_rate=1e-3, temperature=0.1):
        super().__init__()
        self.lr = learning_rate
        self.temperature = 0.1
        self.score_fn = ScoringFunction(2*input_dim, hidden_dim, num_layers)
        self.save_hyperparameters()

    def forward(self, s, x):
        N, d = s.size()
        scores = torch.zeros((N, N))
        for i in range(N):
            scores[:, i] = self.score_fn(s[i].unsqueeze(0).expand((N, d)), x)
        target = torch.arange(N)
        return F.cross_entropy(scores / self.temperature, target)

    def training_step(self, batch, batch_index):
        s, x = batch
        loss = self(s, x)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_index):
        s, x = batch
        val_loss = self(s, x)
        self.log("val_loss", val_loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer