import torch
import torch.nn as nn


class Head(nn.Module):
    """Single head of the multi-head attention layer."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        """Initialise the head.

        Args:
            n_embd (int): The dimension of the embeddings.
            head_size (int): The size of the head.
            block_size (int): The size of the block.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-attention. A forward pass of the head.

        We project the input into key, query, value and compute the attention score. We then perform a weighted aggregation of the values.

        x is the input tensor, with shape (B, T, C), where B is the batch size, T is the sequence length, and C is the number of channels.

        Args:
            x: (B, T, C) tensor of input data

        Returns:
            out: (B, T, head_size) tensor of output data
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, dropout: float):
        """Initialise the multi-head attention layer.

        Args:
            num_heads (int): The number of heads.
            head_size (int): The size of each head.
            n_embd (int): The dimension of the embeddings.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Each head computes self-attention independently, and the outputs are concatenated and
        projected back to the original size of the input.

        Args:
            x: (B, T, C) tensor of input data

        Returns:
            out: (B, T, C) tensor of output data
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concat over the Channel dimension
        out = self.dropout(self.proj(out))  # Project back to the original size of the input, apply dropout
        return out
