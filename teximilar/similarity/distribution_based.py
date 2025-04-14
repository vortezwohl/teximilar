import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log(p / q))
