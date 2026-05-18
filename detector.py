from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    start_index: int
    end_index: int
    direction: int                  # +1 or -1
    max_polarization: float         # signed value with maximum absolute magnitude
    monotonicity_fraction: float


def detect_ramp_ups(polarization_series: pd.Series, config: dict) -> list[Candidate]:
    """Detect ramp-up windows in a polarization series using the anchor+grow algorithm.

    Pure function — no file I/O or side effects.

    Args:
        polarization_series: pd.Series of float polarization values (already filtered
            for one QMeter and frequency range).
        config: QMeter threshold dict with keys start_threshold, min_end_pol,
            min_ramp_rows, monotonicity_fraction.

    Returns:
        List of Candidate objects; empty if none found.
    """
    start_threshold = config["start_threshold"]
    min_end_pol = config["min_end_pol"]
    min_ramp_rows = config["min_ramp_rows"]
    mono_threshold = config["monotonicity_fraction"]

    pol = np.asarray(
        polarization_series.values if hasattr(polarization_series, "values")
        else polarization_series,
        dtype=float,
    )
    n = len(pol)
    candidates: list[Candidate] = []

    i = 0
    while i < n - 1:
        # --- find anchor ---
        if abs(pol[i]) > start_threshold:
            i += 1
            continue

        anchor = i

        # Determine direction from first non-zero delta after anchor.
        direction = 0
        for k in range(anchor + 1, min(anchor + 50, n)):
            delta = pol[k] - pol[k - 1]
            if delta != 0.0:
                direction = 1 if delta > 0 else -1
                break

        if direction == 0:
            i += 1
            continue

        # --- grow forward ---
        agree = 0
        total = 0
        prev_agree = 0
        prev_total = 0
        end = anchor

        for j in range(anchor + 1, n):
            delta = pol[j] - pol[j - 1]
            prev_agree, prev_total = agree, total
            if (direction == 1 and delta > 0) or (direction == -1 and delta < 0):
                agree += 1
            total += 1

            frac = agree / total
            if frac < mono_threshold:
                # Ramp broken — restore counts from before this step
                agree, total = prev_agree, prev_total
                end = j - 1
                break

            end = j

        # --- evaluate candidate ---
        window_len = end - anchor + 1
        window = pol[anchor: end + 1]
        max_pol = float(window[np.argmax(np.abs(window))])
        mono_frac = agree / total if total > 0 else 0.0

        if (window_len >= min_ramp_rows
                and abs(max_pol) >= min_end_pol
                and mono_frac >= mono_threshold):
            candidates.append(Candidate(
                start_index=anchor,
                end_index=end,
                direction=direction,
                max_polarization=max_pol,
                monotonicity_fraction=mono_frac,
            ))

        i = end + 1

    return candidates
