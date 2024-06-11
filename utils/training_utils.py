import copy
from typing import Callable, Tuple

import torch


@torch.no_grad()
def estimate_loss(model, eval_iters: int, get_batch: Callable) -> dict:
    """Average loss over multiple batches.

    Args:
        model (nn.Module): The model.
        eval_iters (int): The number of iterations to average over.
        get_batch (callable): A function that returns a batch.

    Returns:
        out: dictionary with keys 'train' and 'val', each with a scalar value
    """
    model.eval()
    losses = {'train': 0, 'val': 0}

    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean()

    model.train()
    return losses


def train(model, optimizer: Callable, get_batch: Callable, estimate_loss: Callable,
          max_iters: int, eval_interval: int, best_validation_loss: float,
          train_losses: list, eval_iters: int) -> Tuple[torch.nn.Module, list]:
    """Training loop for the model.

    Args:
        model (nn.Module): The model.
        optimizer (torch.optim): The optimizer.
        get_batch (callable): A function that returns a batch.
        estimate_loss (callable): A function that estimates the loss.
        max_iters (int): The maximum number of iterations.
        eval_interval (int): The number of iterations between evaluations.
        best_validation_loss (float): The best validation loss.
        train_losses (list): The training losses.
        eval_iters (int): The number of iterations to average over.

    Returns:
        tuple: The trained model and the training losses.
    """
    best_model_state = None

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, get_batch)
            train_losses.append(losses['train'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_validation_loss:
                best_validation_loss = losses['val']
                best_model_state = copy.deepcopy(model.state_dict())
                print('### checkpointed ')

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses
