import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    """A transformer block.

    We apply the first layer normalisation to the original input, then pass it through the self-attention mechanism, and add the output to the
    original input via a residual connection.

    We then take the resulting output, apply the second layer normalisation, pass it through the feed-forward network, and add the output to the
    original input via a residual connection.

    NB: The Layer Normalisation is done before the self-attention and feed-forward, which has changed since the original "Attention Is All You Need" paper,
    where it was done after. Layer Normalisation is a form of Batch Normalisation, but it normalises over the last dimension, rather than the first.
    It is on a per-token basis, rather than a per-batch basis.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        """Initialises the TransformerBlock.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of heads in the multi-head attention mechanism.
            dropout (float): The dropout rate.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attention = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForwardNetwork(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the TransformerBlock.

        Args:
            x: (B, T, C) tensor of input data

        Returns:
            out: (B, T, C) tensor of output data, the final "result" of the transformer block
        """
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
