from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch_cluster import knn as cluster_knn

from BIBgen.models import common

class EdgeConvBlock(nn.Module):
    def __init__(self,
        input_size : int,
        output_size : int,
        n_hidden_layers : int,
        conditional_size : int,
        n_neighbors : int,
        cylindrical : bool = True,
        mu : torch.Tensor | None = None,
        std : torch.Tensor | None = None,
    ):
        """
        EdgeConv block mostly based on https://arxiv.org/abs/1801.07829.
        Each hit transforms based on information from its k nearest neighbors.

        Parameters
        ----------
        input_size : int
            Size of each input set member. Note, not the number of members
        output_size : int
            Size of each output member.
        n_hidden_layers : int
            Number of hidden layers in the graph edge MLP
        conditional_size : int
            Size of global conditional vector (e.g. diffusion timestep embedding and total number of hits)
        n_neighbors : int
            Number of neighbors to pass messages between
        cylindrical : bool
            Whether the coordinates are cylindrical. If so, run k-nn with (s * phi, s, z). Otherwise, use directly (1, 2, 3).
        mu : torch.Tensor, optional
            Sphering mean with shape (4,). Shifts the coordinates back to physical values before k-nn. Required if cylindrical and std are provided
        std : torch.Tensor, optional
            If provided, rescales the coordinates in spatial space back to physical values before k-nn

        Returns
        -------
        self : EdgeConvBlock
            torch module useable in neural networks
        """
        super().__init__()
        self.k = n_neighbors
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)
        self.cyl = cylindrical

        self.edge = common.make_mlp(2*input_size, output_size, output_size, n_hidden_layers, activation="SiLU", final_activation="SiLU")
        self.norm = nn.LayerNorm(input_size)
        self.modulation = nn.Linear(conditional_size, 2*input_size)
        self.glob = nn.Linear(output_size, output_size)

    def knn(self, coords : torch.Tensor, input_set : torch.Tensor):
        """
        Find k nearest neighbors for each point.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates to determine neighbors with shape (nbatch, nhits, n_coords)
        input_set : torch.Tensor
            Vectors to output corresponding to neighbors found based on coords. With shape (nbatch, nhits, input_size)

        Returns
        -------
        neighbors : torch.Tensor
            Vector of neighbors selected from input_set based on coords with shape (nbatch, nhits, n_neighbors, input_size)
        """
        # Only the neighbor search is non-differentiable; the gather must stay in the graph
        with torch.no_grad():
            nbatch, nhits, ncoords = coords.shape
            flat = coords.reshape(nbatch * nhits, ncoords).float()
            batch = torch.arange(nbatch, device=coords.device).repeat_interleave(nhits)
            _, col = cluster_knn(flat, flat, self.k, batch, batch)  # rows come grouped per query, k each
            indices = (col - batch[col] * nhits).view(nbatch, nhits, self.k)

        nbatch, nhits, _ = indices.shape
        flat_indices = indices.reshape(nbatch, nhits * self.k, 1).expand(-1, -1, input_set.size(-1))
        return torch.gather(input_set, 1, flat_indices).view(nbatch, nhits, self.k, -1)

    def forward(self, input_set : torch.Tensor, conditional : torch.Tensor):
        """
        Runs forward pass of EdgeConv block

        Parameters
        ----------
        input_set : torch.Tensor
            Batched input set with shape (nbatch, nhits, input_size)
        conditional : torch.Tensor
            Conditional tensor with shape (nbatch, conditional_size)

        Returns
        -------
        output_set : torch.Tensor
        """
        scale, shift = self.modulation(conditional).unsqueeze(1).chunk(2, dim=-1) # (nbatch, 1, input_size)
        conditioned_input_set = self.norm(input_set) * (1 + scale) + shift # (nbatch, nhits, input_size)

        # If std is provided, go back to physical coordinates to perform knn, assumed to be indices 1,2,3 of input_set
        if self.std is None:
            coordinates = input_set # (nbatch, nhits, input_size)
        else:
            raw_coords = self.std[1:] * input_set[...,1:4] # (nbatch, nhits, 3)
            coordinates = raw_coords
            if self.cyl:
                coordinates += self.mu[1:]
                coordinates[...,0] *= raw_coords[...,1]
        neighbors = self.knn(coordinates, conditioned_input_set) # (nbatch, nhits, n_neighbors, input_size)

        condition_input_set_projected = conditioned_input_set.unsqueeze(2).expand(-1, -1, self.k, -1) # (nbatch, nhits, n_neighbors, input_size)
        edge_input = torch.cat([condition_input_set_projected, neighbors - condition_input_set_projected], dim=3) # (nbatch, nhits, n_neighbors, 2*input_size)
        output_set = self.edge(edge_input).mean(dim=2) # (nbatch, nhits, output_size)
        
        global_context = self.glob(output_set.mean(dim=1)) # (nbatch, output_size)
        return output_set + global_context.unsqueeze(1) # (nbatch, nhits, output_size)

class DGCNNDenoiser(nn.Module):
    def __init__(self,
        tau_encoding_dimension : int,
        tau_encoding_scale : float,
        pos_encoding_kind : str,
        pos_encoding_dimension : int,
        pos1_encoding_scale : float,
        pos2_encoding_scale : float,
        pos3_encoding_scale : float,
        n_double_blocks : int,
        n_single_blocks : int,
        hidden_layer_size : int,
        n_hidden_extraction_layers : int,
        extraction_layer_size : int,
        mu : torch.Tensor,
        std : torch.Tensor,
        n_neighbors : int,
        pos_encoding_seed : int | None = None,
        nhits_normalization : int = 9_000,
        cylindrical : bool = True,
        n_timesteps : int | None = None,
    ):
        """
        Dynamic Graph Convolutional Neural Network denoiser, architecture based on https://arxiv.org/abs/1801.07829.

        Parameters
        ----------
        tau_encoding_dimension : int
            Number of dimensions to Fourier encode diffusion timestep
        tau_encoding_scale : float
            Scale to encode diffusion timestep
        pos_encoding_kind : str
            One of "none", "positional", or "gaussian".
            "none" to feed only scalar positions.
            "positional" to apply Fourier encoding independently by axis.
            "gaussian" to encode jointly with a randomly selected 3-d frequencies. 
        pos_encoding_dimension : int
            If pos_encoding_kind="gaussian", number of dimensions to encode all coordinates in.
            If pos_encoding_kind="positional", number of dimensions for each coordinate.
        pos1_encoding_scale : float
            Scale to encode 1st spatial coordinate
        pos2_encoding_scale : float
            Scale to encode 2nd spatial coordinate
        pos3_encoding_scale : float
            Scale to encode 3rd spatial coordinate
        n_double_blocks : int
            Number of EdgeConv blocks with a hidden layer
        n_single_blocks : int
            Number of EdgeConv blocks with just a linear transformation
        hidden_layer_size : int
            Size of the hidden layer in the double blocks
        n_hidden_extraction_layers : int
            Number of hidden layers in the final extraction MLP from EdgeConv outputs
        extraction_layer_size : int
            Size of hidden layers in the final extraction MLP from EdgeConv outputs
        mu : torch.Tensor
            Sphering mean with shape (4,), shifts the coordinates back to physical values before k-nn
        std : torch.Tensor
            Sphering std with shape (4,), rescales the coordinates back to physical values before k-nn
        n_neighbors : int
            Number of nearest neighbors in each EdgeConv block
        pos_encoding_seed : int, optional
            Random seed for pos_encoding_kind="gaussian"
        nhits_normalization : int
            Number of hits to normalize the number of hits feature, defaults to 9000
        cylindrical : bool
            Whether the coordinates are cylindrical. If so, run k-nn with (s * phi, s, z). Otherwise, use directly (1, 2, 3).
        n_timesteps : int, optional
            Completely unused

        Returns
        -------
        self : DGCNNDenoiser
            denoising network
        """
        super().__init__()
        self.tau_encoding = common.PositionalEncoding(tau_encoding_dimension, tau_encoding_scale)
        self.nhits_norm = nhits_normalization

        if pos_encoding_kind not in {"none", "positional", "gaussian"}:
            raise ValueError('Unknown positional encoding type. Must be "none", "positional", or "gaussian".')
        self.pos_encoding_kind = pos_encoding_kind
        self.pos_encoding, self.pos1_encoding, self.pos2_encoding, self.pos3_encoding = None, None, None, None
        pos_encoding_size = 3 + 1 # Scalar positions + energy
        if self.pos_encoding_kind == "positional":
            self.pos1_encoding = common.PositionalEncoding(pos_encoding_dimension, scale=pos1_encoding_scale)
            self.pos2_encoding = common.PositionalEncoding(pos_encoding_dimension, scale=pos2_encoding_scale)
            self.pos3_encoding = common.PositionalEncoding(pos_encoding_dimension, scale=pos3_encoding_scale)
            pos_encoding_size += 3 * pos_encoding_dimension
        elif self.pos_encoding_kind == "gaussian":
            self.pos_encoding = common.GaussianFourierFeatures(
                pos_encoding_dimension,
                input_dim=3,
                scales=[pos1_encoding_scale, pos2_encoding_scale, pos3_encoding_scale],
                seed=pos_encoding_seed
            )
            pos_encoding_size += pos_encoding_dimension

        assert n_double_blocks >= 1
        double_blocks = [("doubleEdgeConv0", EdgeConvBlock(
            pos_encoding_size,
            hidden_layer_size,
            n_hidden_layers=1,
            conditional_size=tau_encoding_dimension + 1, # 1 for nhits feature
            n_neighbors=n_neighbors,
            mu=mu,
            std=std, # Only provide std for first block when k-nn is in physical space
            cylindrical=cylindrical
        ))]
        for idouble in range(1, n_double_blocks):
            double_blocks.append(("doubleEdgeConv{}".format(idouble), EdgeConvBlock(
                hidden_layer_size,
                hidden_layer_size,
                n_hidden_layers=1,
                conditional_size=tau_encoding_dimension + 1,
                n_neighbors=n_neighbors,
            )))
        self.double_blocks_stack = nn.ModuleDict(OrderedDict(double_blocks))

        self.single_blocks_stack = nn.ModuleDict(OrderedDict(("singleEdgeConv{}".format(isingle),
            EdgeConvBlock(
                hidden_layer_size,
                hidden_layer_size,
                n_hidden_layers=0,
                conditional_size=tau_encoding_dimension + 1,
                n_neighbors=n_neighbors,
            )
        ) for isingle in range(n_single_blocks)))

        assert n_hidden_extraction_layers >= 1
        extraction_input_size = (n_double_blocks + n_single_blocks) * hidden_layer_size # Size of all EdgeConv block outputs concatenated
        self.extraction_block = common.make_mlp(extraction_input_size, 4, extraction_layer_size, n_hidden_extraction_layers, activation="SiLU", initialize_zero=True)

    @staticmethod
    def need_mu_std():
        "Indicates this class needs mu and std from sphering provided"
        return True

    @staticmethod
    def need_n_timesteps():
        "Indicates this class needs number of diffusion time steps"
        return False

    @property
    def predict_variances(self):
        return False

    def forward(self, input_set : torch.Tensor, tau : torch.Tensor):
        """
        Runs forward pass of DGCNN denoising network

        Parameters
        ----------
        input_set : torch.Tensor
            Batched input set with shape (nbatch, nhits, 4)
        tau : torch.Tensor
            Diffusion timestep with shape (nbatch,)

        Returns
        -------
        output_set : torch.Tensor
            Predicted hits at timestep tau - 1
        """
        unbatched = input_set.ndim < 3
        if unbatched:
            input_set = input_set.unsqueeze(0)
        if tau.ndim < 1:
            tau = tau.unsqueeze(0)
        assert input_set.ndim == 3 and tau.ndim == 1

        tau_encoded = self.tau_encoding(tau.to(input_set.dtype)) # (nbatch, tau_encoding_dimension)
        nhits = input_set.shape[-2]
        log_nhits = torch.log(tau_encoded.new_full((*tau_encoded.shape[:-1], 1), nhits / self.nhits_norm + 1)) # (nbatch, 1)
        conditional = torch.cat([tau_encoded, log_nhits], dim=-1) # (nbatch, tau_encoding_dimension + 1)

        if self.pos_encoding_kind == "none":
            points = input_set # (nbatch, nhits, 4)
        elif self.pos_encoding_kind == "positional":
            points = torch.cat([
                input_set,
                self.pos1_encoding(input_set[...,1]),
                self.pos2_encoding(input_set[...,2]),
                self.pos3_encoding(input_set[...,3]),
            ], dim=2) # (nbatch, nhits, 4 + 3 * pos_encoding_dimension)
        elif self.pos_encoding_kind == "gaussian":
            points = torch.cat([input_set, self.pos_encoding(input_set[...,1:4])], dim=2) # (nbatch, nhits, 4 + pos_encoding_dimension)

        edgeconv_outputs = []
        for block in self.double_blocks_stack.values():
            points = block(points, conditional) # (nbatch, nhits, hidden_layer_size)
            edgeconv_outputs.append(points)
        for block in self.single_blocks_stack.values():
            points = block(points, conditional) # (nbatch, nhits, hidden_layer_size)
            edgeconv_outputs.append(points)

        extraction_input = torch.cat(edgeconv_outputs, dim=2) # (nbatch, nhits, (n_double_blocks + n_single_blocks) * hidden_layer_size)
        epsilon = self.extraction_block(extraction_input) # (nbatch, nhits, 4)
        output_set = input_set + epsilon
        return output_set.squeeze(0) if unbatched else output_set