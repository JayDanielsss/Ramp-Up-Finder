import math

import numpy as np
import pandas as pd
import pytest

from features import extract_features

EXPECTED_KEYS = {
    "start_pol", "end_pol", "max_pol", "net_slope",
    "monotonicity_fraction", "gradient_std", "nmr_gaussian_r2",
}


def test_known_series_without_rawsignal():
    pol = pd.Series([0.0, 0.05, 0.10, 0.15, 0.20])
    f = extract_features(pol)

    assert f["start_pol"] == pytest.approx(0.0)
    assert f["end_pol"] == pytest.approx(0.20)
    assert f["max_pol"] == pytest.approx(0.20)
    assert f["net_slope"] == pytest.approx(0.20 / 5)
    assert f["monotonicity_fraction"] == pytest.approx(1.0)
    # Uniform deltas of 0.05 → std = 0.0
    assert f["gradient_std"] == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(f["nmr_gaussian_r2"])


def test_all_keys_present():
    pol = pd.Series([0.0, 0.1, 0.2])
    f = extract_features(pol)
    assert set(f.keys()) == EXPECTED_KEYS


def test_negative_direction():
    pol = pd.Series([0.0, -0.1, -0.2, -0.3])
    f = extract_features(pol)
    assert f["start_pol"] == pytest.approx(0.0)
    assert f["end_pol"] == pytest.approx(-0.3)
    # max_pol: signed value with largest |magnitude| = -0.3
    assert f["max_pol"] == pytest.approx(-0.3)
    assert f["monotonicity_fraction"] == pytest.approx(1.0)
    assert f["net_slope"] < 0


def test_mixed_series_reduces_monotonicity():
    # One step goes the wrong direction → fraction < 1.0
    pol = pd.Series([0.0, 0.1, 0.05, 0.20])
    f = extract_features(pol)
    # Deltas: [+0.1, -0.05, +0.15] — net direction positive. 2 of 3 agree.
    assert f["monotonicity_fraction"] == pytest.approx(2 / 3)


def test_nmr_gaussian_r2_nan_without_rawsignal():
    pol = pd.Series([0.0, 0.1, 0.2, 0.3])
    f = extract_features(pol, raw_signal_df=None)
    assert math.isnan(f["nmr_gaussian_r2"])


def test_nmr_gaussian_r2_with_gaussian_rawsignal():
    x = np.arange(500, dtype=float)
    gaussian_signal = 5.0 * np.exp(-0.5 * ((x - 250) / 50) ** 2) + 0.1
    # Single-row DataFrame — mean across rows == the row itself
    raw_df = pd.DataFrame([gaussian_signal])
    pol = pd.Series(np.linspace(0.0, 0.3, 1))
    f = extract_features(pol, raw_signal_df=raw_df)
    assert not math.isnan(f["nmr_gaussian_r2"])
    assert f["nmr_gaussian_r2"] > 0.95
