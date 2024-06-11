from typing import Tuple

import torch


def encode(text: str, stoi: dict) -> list:
    """Encode text to a list of integers.

    Args:
        text: The text to encode.
        stoi: A dictionary mapping characters to integers.

    Returns:
        A list of integers representing the text.
    """
    return [stoi[c] for c in text]


def decode(encoded_text: list, itos: dict) -> str:
    """Decode a list of integers to text.

    Args:
        encoded_text: A list of integers.
        itos: A dictionary mapping integers to characters.

    Returns:
        A string representing the decoded text.
    """
    return ''.join([itos[i] for i in encoded_text])


def load_data(filepath: str, stoi: dict) -> torch.Tensor:
    """Load text data from a file and encode it.

    Args:
        filepath: The path to the file.
        stoi: A dictionary mapping characters to integers.

    Returns:
        A tensor of integers representing the text.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    return data


def create_vocab(text: str) -> Tuple[int, dict, dict]:
    """Create a vocabulary from a text.

    Args:
        text: The text to create the vocabulary from.

    Returns:
        vocab_size: The size of the vocabulary.
        stoi: A dictionary mapping characters to integers.
        itos: A dictionary mapping integers to characters.
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data.

    Args:
        data: The data to sample from.
        batch_size: The size of the batch.
        block_size: The size of the blocks to sample.
        device: The device to use.

    Returns:
        x: The input data.
        y: The target data.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
