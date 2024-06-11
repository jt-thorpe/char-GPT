import torch
import torch.nn as nn


def generate_text(model, context: int, max_new_tokens: int, temperature: float = 1.0,
                  block_size: int = 256, device: str = 'cpu') -> torch.Tensor:
    """Generate text from a given context using the model.

    Args:
        model (nn.Module): The language model.
        context (int): The initial context.
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for sampling.
        block_size (int): The block size for the model.
        device (str): The device to run the model on.

    Returns:
        torch.Tensor: The generated text.
    """
    model.eval()
    context = context.to(device)
    for _ in range(max_new_tokens):
        idx_cond = context[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = nn.functional.softmax(logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
    return context
