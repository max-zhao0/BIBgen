from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

def whitening_matrices(features : ArrayLike):
    mu = np.mean(features, axis=0)
    centered = features - mu

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    eps = 1e-6
    D_inv_sqrt = np.diag(1 / np.sqrt(eigvals + eps))
    D_sqrt = np.diag(np.sqrt(eigvals + eps))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    W_inv = eigvecs @ D_sqrt @ eigvecs.T

    return W, W_inv, mu

def diffuse(features : ArrayLike, betas : Sequence) -> ArrayLike:
    result = np.empty((len(betas) + 1, len(features)))
    result[0] = features
    z = np.random.normal(size=(len(betas), len(features)))

    for tau in range(len(betas)):
        result[tau + 1] = np.sqrt(1 - betas[tau]) * result[tau] + np.sqrt(betas[tau]) * z[tau]

    return result