import numpy as np

def eta_from_cylindrical(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute pseudorapidity from cylindrical coordinates.
    
    Args:
        s: Radial distance from beam axis
        z: Z-position along beam axis
        
    Returns:
        Pseudorapidity eta
    """
    theta = abs(np.arctan2(s, z))
    eta = -np.log(np.tan((theta % (2*np.pi)) / 2.0 + 1e-10))
    return eta

def deltaR(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Compute Delta R metric between coordinate pairs."""
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    return np.sqrt(delta_eta**2 + delta_phi**2)