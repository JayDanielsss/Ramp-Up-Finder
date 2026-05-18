import numpy as np
import pandas as pd
import pytest

from detector import Candidate, detect_ramp_ups

CONFIG = {
    "start_threshold": 0.05,
    "min_end_pol": 0.20,
    "min_ramp_rows": 100,
    "monotonicity_fraction": 0.85,
}


def _series(arr):
    return pd.Series(arr, dtype=float)


def test_flat_series_no_candidates():
    pol = _series(np.zeros(200))
    assert detect_ramp_ups(pol, CONFIG) == []


def test_single_positive_ramp():
    pol = _series(np.linspace(0.0, 0.5, 150))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.start_index == 0
    assert c.end_index == 149
    assert c.direction == 1
    assert abs(c.max_polarization) >= 0.20
    assert c.monotonicity_fraction >= 0.85


def test_single_negative_ramp():
    pol = _series(np.linspace(0.0, -0.5, 150))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.direction == -1
    assert c.max_polarization <= -0.20
    assert c.monotonicity_fraction >= 0.85


def test_two_ramps_separated_by_reversal():
    # Ramp 1 (0→0.5), then reversal (0.5→0), then ramp 2 (0→0.5).
    # The reversal drops monotonicity below threshold, ending ramp 1.
    # A new anchor near 0 starts ramp 2.
    part1 = np.linspace(0.0, 0.5, 150)
    part2 = np.linspace(0.5, 0.0, 50)
    part3 = np.linspace(0.0, 0.5, 150)
    pol = _series(np.concatenate([part1, part2, part3]))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert len(candidates) == 2
    assert candidates[0].direction == 1
    assert candidates[1].direction == 1
    # Second ramp must start after the first ends
    assert candidates[1].start_index > candidates[0].end_index


def test_noisy_series_below_monotonicity_threshold_no_candidates():
    # Alternating up/down pattern → monotonicity ~0.5, well below 0.85.
    rng = np.random.default_rng(0)
    n = 300
    # Build a series that starts near 0 and has ~50% disagreeing steps.
    steps = np.where(rng.random(n) < 0.5, 0.01, -0.01)
    steps[0] = 0.0  # anchor
    pol = _series(np.cumsum(steps))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert candidates == []


def test_too_short_no_candidates():
    # Clean ramp but only 50 rows — below min_ramp_rows=100.
    pol = _series(np.linspace(0.0, 0.5, 50))
    assert detect_ramp_ups(pol, CONFIG) == []


def test_does_not_reach_min_end_pol_no_candidates():
    # Clean ramp over 150 rows but only reaches 0.10 — below min_end_pol=0.20.
    pol = _series(np.linspace(0.0, 0.10, 150))
    assert detect_ramp_ups(pol, CONFIG) == []


def test_candidate_fields_present():
    pol = _series(np.linspace(0.0, 0.5, 150))
    c = detect_ramp_ups(pol, CONFIG)[0]
    assert isinstance(c, Candidate)
    assert hasattr(c, "start_index")
    assert hasattr(c, "end_index")
    assert hasattr(c, "direction")
    assert hasattr(c, "max_polarization")
    assert hasattr(c, "monotonicity_fraction")
