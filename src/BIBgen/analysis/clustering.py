from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from BIBgen.analysis.comparison_analyzer import ComparisonAnalyzer
from BIBgen.analysis.utils import eta_from_cylindrical, deltaR

def window_search(func, bottom, top, npoll=20, max_depth=5, cache=None):
    if cache is None:
        cache = {}

    to_poll = np.unique(np.round(np.linspace(bottom, top, npoll)).astype(int))

    scores = []
    for p in to_poll:
        if p not in cache:
            cache[p] = func(p)
        scores.append(cache[p])

    best_polled_idx = np.argmax(scores)
    if best_polled_idx == 0 or best_polled_idx == len(scores) - 1:
        print("WARNING: edge optimum found")
        print("to_poll:", to_poll)
        print("scores:", scores)
        
    new_bottom = to_poll[max(0, best_polled_idx - 1)]
    new_top = to_poll[min(len(scores) - 1, best_polled_idx + 1)]

def eta_phi_pairs(data):
    phi = data[:,1]
    s = data[:,2]
    z = data[:,3]

    eta = eta_from_cylindrical(s, z)
    return np.stack((eta, phi), axis=1)

def max_silhouette(data, scan_width=(0.01, 0.40)):
    pairs = eta_phi_pairs(data)
    return None, None, None

def silhouette_scan(data, to_poll : np.typing.ArrayLike):
    pairs = eta_phi_pairs(data)
    nhits = len(pairs)
    scores = [silhouette_score(pairs, KMeans(n_clusters=k, random_state=42).fit_predict(pairs)) for k in to_poll]
    return scores