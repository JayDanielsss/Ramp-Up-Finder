import numpy as np
import pandas as pd
import pytest

from detector import Candidate, detect_ramp_ups

CONFIG = {
    "prominence": 0.10,
    "min_swing": 0.20,
    "min_ramp_rows": 5,
    "monotonicity_fraction": 0.80,
}


def _series(arr):
    return pd.Series(arr, dtype=float)


def test_flat_series_no_candidates():
    # No peaks/troughs; single segment has swing=0, below min_swing.
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
    assert abs(c.swing) >= 0.20
    assert c.swing > 0
    assert c.monotonicity_fraction >= 0.80


def test_single_negative_ramp():
    pol = _series(np.linspace(0.0, -0.5, 150))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.direction == -1
    assert c.swing < 0
    assert abs(c.swing) >= 0.20
    assert c.monotonicity_fraction >= 0.80


def test_back_to_back_ramps_without_zero_crossing():
    # Positive ramp, then negative ramp, then positive — no return to zero.
    # The prominence-based detector finds all three segments.
    part1 = np.linspace(0.0, 0.5, 150)
    part2 = np.linspace(0.5, -0.5, 150)
    part3 = np.linspace(-0.5, 0.5, 150)
    pol = _series(np.concatenate([part1, part2, part3]))
    candidates = detect_ramp_ups(pol, CONFIG)
    assert len(candidates) >= 2
    directions = [c.direction for c in candidates]
    assert 1 in directions
    assert -1 in directions


def test_too_short_no_candidates():
    # Clean ramp but fewer than min_ramp_rows=5 rows.
    pol = _series(np.linspace(0.0, 0.5, 3))
    assert detect_ramp_ups(pol, CONFIG) == []


def test_below_min_swing_no_candidates():
    # Ramp over enough rows but swing only 0.05, below min_swing=0.20.
    pol = _series(np.linspace(0.0, 0.05, 150))
    assert detect_ramp_ups(pol, CONFIG) == []


def test_candidate_fields_present():
    pol = _series(np.linspace(0.0, 0.5, 150))
    c = detect_ramp_ups(pol, CONFIG)[0]
    assert isinstance(c, Candidate)
    assert hasattr(c, "start_index")
    assert hasattr(c, "end_index")
    assert hasattr(c, "direction")
    assert hasattr(c, "start_polarization")
    assert hasattr(c, "end_polarization")
    assert hasattr(c, "swing")
    assert hasattr(c, "max_polarization")
    assert hasattr(c, "monotonicity_fraction")
    assert abs(c.swing - (c.end_polarization - c.start_polarization)) < 1e-9


def test_empty_series():
    pol = _series([])
    assert detect_ramp_ups(pol, CONFIG) == []


def test_single_element_series():
    pol = _series([0.5])
    assert detect_ramp_ups(pol, CONFIG) == []
