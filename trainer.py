'''
Functions for training model.
'''

import torch
import numpy as np

def train_one_iter(model, train_dataloader, loss_func, optimiser, device, iteration, log_interval=10):
    '''
    Train network for one iteration
    Args:
        - model: instantiated model
        - train_dataloader: pytorch dataloader object
        - loss_func
        - optimiser
        - device: 'cpu'|'cuda'
        - iteration: number of iteration during training
    Returns:
        - Mean loss over iteration
        - Mean accuracy over iteration
    '''
    losses = []
    running_correct, running_total = 0, 0
    model.train()

    inputs, targets = iter(train_dataloader).next()
    inputs, targets = inputs.to(device), targets.to(device)
    targets = torch.max(targets, 1)[1]
    optimiser.zero_grad()

    outputs, preds, mask, probs = model(inputs, iteration)
    loss = loss_func.compute_loss(model, outputs, targets, probs)

    loss.backward()
    optimiser.step()
    losses.append(loss.item())

    preds = torch.max(preds, 1)[1]
    running_correct += torch.sum(preds==targets).item()
    running_total += len(targets)

    if iteration % log_interval == 0:
        print(f"[Iteration {iteration}] Loss: {np.mean(losses):.4f} Accuracy: {(100.*running_correct/running_total):.2f}")

    return np.mean(losses), 100.*running_correct/running_total