"""Small signed-distance helpers. Negative = solid."""
import numpy as np

CELL = 8.0            # gyroid / diamond cell size (mm)
T_IN, T_OUT = 1.00, 0.62   # gyroid level: dense on the soil side, open outside


def smin(a, b, k):
    """Smooth union of two fields, blend radius k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def cylindrical(X, Y):
    """(r, theta) of cartesian points."""
    return np.sqrt(X * X + Y * Y), np.arctan2(Y, X)


def gyroid(theta, z, r, level, n_theta):
    """Cylindrical gyroid with a low-frequency organic warp. n_theta cells
    around; every warp term has an integer frequency in theta so the pattern
    closes seamlessly."""
    u = theta * n_theta
    v = 2 * np.pi * z / CELL
    w = 2 * np.pi * r / CELL
    u = u + 1.1 * np.sin(3 * theta + 2 * np.pi * z / 55.0) + 0.5 * np.sin(7 * theta - 2 * np.pi * z / 31.0)
    v = v + 0.9 * np.sin(5 * theta + 2 * np.pi * z / 70.0) + 0.4 * np.cos(2 * theta + 2 * np.pi * z / 23.0)
    g = np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    return (np.abs(g) - level) * CELL / (2 * np.pi * 1.4)
