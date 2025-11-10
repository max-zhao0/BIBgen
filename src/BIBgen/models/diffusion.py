from typing import Callable, Collection

import numpy as np
import torch
from torch import nn

class FourierEncoding(nn.Module):
    def __init__(self, dimension : int, initial_frequencies : torch.Tensor | None = None, learned : bool = True):
        super().__init__()
        if not learned:
            raise NotImplementedError("Unlearned encoding not supported.")

        if initial_frequencies is None:
            initial_frequencies = 0.5**torch.linspace(1, 16, dimension)
        assert len(initial_frequencies) == dimension

        # Inverse sigmoid
        initial_values = - torch.log((1 / initial_frequencies) - 1)

        self.fourier_table = nn.Parameter(data=initial_values)
        self.fourier_activation = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        frequencies = 2 * np.pi * self.fourier_activation(self.fourier_table)
        thetas = torch.outer(x, frequencies)
        sin_elems = torch.sin(thetas)
        cos_elems = torch.cos(thetas)
        return torch.cat((sin_elems, cos_elems), dim=1)

class VarianceHead(nn.Module):
    SP_BETA = 1.0
    SP_THRESH = 20.0

    def __init__(self, max_steps : int, initial_variances : torch.Tensor | None = None):
        super().__init__()
        if initial_variances is None:
            initial_variances = torch.rand(max_steps)
        assert len(initial_variances) == max_steps

        # Inverse Softplus
        initial_values = torch.where(initial_variances < self.SP_THRESH, 
            torch.log(torch.exp(self.SP_BETA * initial_variances) - 1) / self.SP_BETA,
            initial_variances
        )

        self.varhead_lookup_table = nn.Parameter(data=initial_values)
        self.varhead_activation = nn.Softplus(beta=self.SP_BETA, threshold=self.SP_THRESH)
        
    def forward(self, tau : torch.Tensor) -> torch.Tensor:
        variance_lookup = self.varhead_activation(self.varhead_lookup_table)
        return variance_lookup[tau]
