import numpy as np

from BIBgen.analysis.utils import *

def test_eta_computation():
    """Test pseudorapidity calculation."""

    # Test forward region (positive z)
    s = np.array([100.0])
    z = np.array([200.0])
    eta = eta_from_cylindrical(s, z)
    assert np.isfinite(eta[0])
    assert eta[0] > 0

    # Test backward region (negative z)
    z = np.array([-200.0])
    eta = eta_from_cylindrical(s, z)
    assert np.isfinite(eta[0])
    assert eta[0] < 0

def test_delta_r_computation():
    """Test delta R calculation."""

    eta1 = np.array([0.0, 1.0])
    phi1 = np.array([0.0, 0.0])
    eta2 = np.array([1.0, 1.0])
    phi2 = np.array([0.0, np.pi])

    dr = deltaR(eta1, phi1, eta2, phi2)

    # Check that delta R is positive
    assert np.all(dr >= 0)

    # Check that identical points have dr = 0
    dr_same = deltaR(eta1, phi1, eta1, phi1)
    assert np.allclose(dr_same, 0)