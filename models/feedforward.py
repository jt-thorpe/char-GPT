import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """A single linear layer followed by a ReLU non-linearity.

    Essentially, Feed Forward is doing the following.
        - So each node has essentially looked at all the others
        - but hasn't had "time" to "think on" what they saw
        - Feed Forward allows each node (token) independently to consider what it has seen
    """

    def __init__(self, n_embd: int, dropout: float):
        """Initialises the feed-forward network.

        - The feed-forward network consists of two linear layers with a ReLU activation function.
        - The first linear layer projects the input into a higher-dimensional space (n_embd*4).
        - The ReLU activation function introduces non-linearity into the model, ReLU basically 
          says if the input is negative, output 0, otherwise output the input.
        - The second linear layer projects the output back to the original size of the input.
        - The dropout layer is applied right before the connection back into the residual pathway.

        Args:
            n_embd (int): The dimension of the embeddings.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the feed-forward network.

        Args:
            x: (B, T, C) tensor of input data

        Returns:
            out: (B, T, C) tensor of output data
        """
        return self.net(x)
