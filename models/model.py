import torch
import torch.nn as nn

from .transformer_block import TransformerBlock


class BigramLanguageModel(nn.Module):
    """A simple bigram language model.

    This is a simple example of a language model that is trained to predict the next character given the previous character.

    The model consists of an embedding layer, a position embedding layer, a stack of transformer blocks, a layer normalisation layer, and a linear layer.

    The embedding layer encodes the tokens into embeddings, the position embedding layer encodes the position of the tokens, the stack of transformer blocks processes the input, the layer normalisation layer normalises the output of the transformer blocks, and the linear layer outputs the logits.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int, dropout: float) -> None:
        """Initialises the bigram language model.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_embd (int): The dimension of the embeddings.
            n_head (int): The number of heads in the multi-head attention.
            n_layer (int): The number of transformer blocks.
            block_size (int): The size of the block.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Linear model head: embedding -> logits

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """Compute the logits and loss.

        Args:
            idx: (B,T) tensor of integers
            targets: (B,T) tensor of integers, or None

        Returns:
            logits: (B,T,C) tensor of logits
            loss: scalar tensor, or None
        """
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # (B,T,C)
        positional_embedding = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # cross_entropy expects a 2D tensor for input
            targets = targets.view(B*T)  # cross_entropy expects a 1D tensor for target
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Generates new tokens given a context.

        Args:
            idx: (B,T) tensor, indices in the current context
            max_new_tokens: int, maximum number of new tokens to generate
            temperature: float, temperature of the softmax

        Returns:
            idx: (B,T+1) tensor of integers
        """
        for _ in range(max_new_tokens):
            # crop idx to last block_size many tokens, as embed_table can only hold up to block_size
            idx_cond = idx[:, -self.blocks[0].block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = nn.functional.softmax(logits / temperature, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, C)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
