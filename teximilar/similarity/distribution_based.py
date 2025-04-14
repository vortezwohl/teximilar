# 越小越相似
import numpy as np

from teximilar.similarity.set_based import bias


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    q = np.where(q == 0, bias, q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2
