"""Small signed-distance helpers. Negative = solid."""
import numpy as np


def smin(a, b, k):
    """Smooth union of two fields, blend radius k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

