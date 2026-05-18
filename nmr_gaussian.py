from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, amplitude, center, width, offset):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2) + offset


def fit_gaussian_r2(signal_array) -> float:
    """Fit a 1-D Gaussian to signal_array and return R² goodness-of-fit.

    Returns nan if the fit fails or the input is degenerate.
    """
    y = np.asarray(signal_array, dtype=float)
    if y.size < 4:
        return float("nan")

    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0.0:
        return float("nan")

    x = np.arange(len(y), dtype=float)
    amplitude0 = y.max() - y.min()
    center0 = float(np.argmax(np.abs(y - y.mean())))
    width0 = len(y) / 4.0
    offset0 = y.min()

    try:
        popt, _ = curve_fit(
            _gaussian, x, y,
            p0=[amplitude0, center0, width0, offset0],
            maxfev=10_000,
        )
    except (RuntimeError, ValueError):
        return float("nan")

    residuals = y - _gaussian(x, *popt)
    ss_res = np.sum(residuals ** 2)
    return float(1.0 - ss_res / ss_tot)
