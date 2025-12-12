import torch
from torch import nn

def generate_sphered(model : nn.Module, n_members : int, device, demo : bool = False, verbosity : int = 0):
    model = model.to(device)
    current_event = torch.normal(mean=0, std=1, size=(n_members, 4)).to(device)
    if verbosity >= 1:
        print("White noise:", current_event[:10])

    with torch.no_grad():
        for tau in range(model.n_timesteps-1, -1, -1):
            mu, var = model(current_event, tau)
            current_event = torch.normal(mean=mu, std=torch.sqrt(var)) if not demo else mu
            if verbosity >= 2:
                print("Denoised to tau =", tau)

    return current_event