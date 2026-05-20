from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class Candidate:
    start_index: int
    end_index: int
    direction: int                  # +1 or -1
    start_polarization: float
    end_polarization: float
    swing: float                    # end_polarization - start_polarization
    max_polarization: float         # signed value with maximum absolute magnitude
    monotonicity_fraction: float


def detect_ramp_ups(polarization_series: pd.Series, config: dict) -> list[Candidate]:
    """Detect ramp-up windows using a two-pass prominence-based segmenter.

    Pass 1: find boundary indices from prominent peaks/troughs plus series endpoints.
    Pass 2: form one candidate per adjacent boundary pair; keep if it passes
    min_ramp_rows, min_swing, and monotonicity_fraction thresholds.

    Pure function — no file I/O or side effects.
    """
    prominence = config["prominence"]
    min_swing = config["min_swing"]
    min_ramp_rows = config["min_ramp_rows"]
    mono_threshold = config["monotonicity_fraction"]

    pol = np.asarray(
        polarization_series.values if hasattr(polarization_series, "values")
        else polarization_series,
        dtype=float,
    )
    n = len(pol)
    if n < 2:
        return []

    # Pass 1 — find boundaries
    peaks_pos, _ = find_peaks(pol, prominence=prominence)
    peaks_neg, _ = find_peaks(-pol, prominence=prominence)
    boundaries = sorted(set([0, n - 1]) | set(peaks_pos.tolist()) | set(peaks_neg.tolist()))

    # Pass 2 — segment and filter
    candidates: list[Candidate] = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        window = pol[start_idx: end_idx + 1]
        window_len = end_idx - start_idx + 1

        start_pol = float(pol[start_idx])
        end_pol = float(pol[end_idx])
        swing = end_pol - start_pol
        direction = 1 if swing >= 0 else -1

        deltas = np.diff(window)
        total = len(deltas)
        if total > 0:
            agree = int(np.sum(deltas > 0) if direction == 1 else np.sum(deltas < 0))
            mono_frac = agree / total
        else:
            mono_frac = 0.0

        max_pol = float(window[np.argmax(np.abs(window))])

        if (window_len >= min_ramp_rows
                and abs(swing) >= min_swing
                and mono_frac >= mono_threshold):
            candidates.append(Candidate(
                start_index=start_idx,
                end_index=end_idx,
                direction=direction,
                start_polarization=start_pol,
                end_polarization=end_pol,
                swing=swing,
                max_polarization=max_pol,
                monotonicity_fraction=mono_frac,
            ))

    return candidates
