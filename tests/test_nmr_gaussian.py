import math

import numpy as np
import pytest

from nmr_gaussian import fit_gaussian_r2


def _perfect_gaussian(n=500, amplitude=5.0, center=None, width=50.0, offset=0.1):
    if center is None:
        center = n / 2
    x = np.arange(n, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2) + offset


def test_perfect_gaussian_high_r2():
    signal = _perfect_gaussian()
    r2 = fit_gaussian_r2(signal)
    assert r2 > 0.95


def test_white_noise_low_r2():
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(500)
    r2 = fit_gaussian_r2(noise)
    assert r2 < 0.5


def test_return_type_is_float():
    r2 = fit_gaussian_r2(_perfect_gaussian())
    assert isinstance(r2, float)

    rng = np.random.default_rng(0)
    r2_noise = fit_gaussian_r2(rng.standard_normal(500))
    assert isinstance(r2_noise, float)


def test_constant_array_returns_nan():
    # SS_tot = 0 → degenerate fit → nan
    constant = np.zeros(200)
    r2 = fit_gaussian_r2(constant)
    assert math.isnan(r2)


def test_too_short_returns_nan():
    r2 = fit_gaussian_r2(np.array([1.0, 2.0]))
    assert math.isnan(r2)
