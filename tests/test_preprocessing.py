import pytest
import numpy as np

from BIBgen.preprocessing import *

def test_whitening_matrices():
    data = np.random.rand(50, 5)

    W, W_inv, mu = whitening_matrices(data)

    assert np.cov(W @ (data - mu).T) == pytest.approx(np.identity(5), abs=1e-4)