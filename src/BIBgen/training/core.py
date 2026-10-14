from typing import Callable

import torch
import numpy as np

from BIBgen.training.dataloaders import BaseDataLoader

def log_per_layer_grads(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name:45s} grad={p.grad.norm().item():.3e}  param={p.norm().item():.3e}")

def train(
    dataloader : BaseDataLoader,
    model : torch.nn.Module,
    loss_fn : Callable,
    optimizer : torch.optim.Optimizer,
    device : torch.device,
    scaler : torch.amp.GradScaler | None = None,
    max_steps_diagnostics : int = 0
):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    optimizer : torch.optim.Optimizer
        Optimizer for gradient descent
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")
    max_steps_diagnostics : int
        Maximum number of steps to print diagnostics

    Returns
    -------
    loss : float
        Training loss on the final iteration
    """
    # assert scaler is not None or device.type != "cuda"
    model.train()
    grad_norm = None
    for istep in range(dataloader.nsteps):
        X, y, tau = next(dataloader)
        X, y, tau = X.to(device), y.to(device), tau.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Compute prediction error
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(X, tau)
                loss = loss_fn(pred, y, tau)
        else:
            pred = model(X, tau)
            loss = loss_fn(pred, y, tau)

        # Backpropagation
        try:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if istep < max_steps_diagnostics:
                    log_per_layer_grads(model)
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print("Failed step {}, cuda out of memory".format(istep))
            print("X.size() : {}, y.size() : {}".format(X.size(), y.size()))
            if isinstance(pred, tuple):
                print("pred[0].size() (mu) : {}, pred[1].size() (var) : {}".format(pred[0].size(), pred[1].size()))
            else:
                print("pred.size() : {}".format(pred.size()))
            raise RuntimeError("Intentional exit")

    return loss.item()

def evaluate(dataloader : BaseDataLoader, model : torch.nn.Module, loss_fn : Callable, device : torch.device):
    """
    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")

    Returns
    -------
    test_loss : float
        Average loss over all validation events
    """

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for istep in range(dataloader.nsteps):
            X, y, tau = next(dataloader)
            X, y, tau = X.to(device), y.to(device), tau.to(device)
            pred = model(X, tau)
            test_loss += loss_fn(pred, y, tau).item()

            if not np.isfinite(test_loss):
                print("test_loss:", test_loss)
                print("X:", X)
                print("y:", y)
                print("pred:", pred)
                raise RuntimeError("Nonfinite test loss")

    return test_loss / dataloader.nsteps
