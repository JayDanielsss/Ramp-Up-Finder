from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from nmr_gaussian import fit_gaussian_r2
except ImportError:
    fit_gaussian_r2 = None  # type: ignore[assignment]


def extract_features(
    polarization_series: pd.Series,
    raw_signal_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute a fixed-length feature vector for a polarization window.

    Args:
        polarization_series: pd.Series of float polarization values for the window.
        raw_signal_df: Optional wide DataFrame of RawSignal rows (one row per
            measurement step, columns = voltage sample positions). Used to compute
            the NMR Gaussian R² feature.

    Returns:
        Dict with exactly 7 keys: start_pol, end_pol, max_pol, net_slope,
        monotonicity_fraction, gradient_std, nmr_gaussian_r2.
    """
    values = np.asarray(
        polarization_series.values if hasattr(polarization_series, "values")
        else polarization_series,
        dtype=float,
    )

    start_pol = float(values[0])
    end_pol = float(values[-1])

    abs_idx = int(np.argmax(np.abs(values)))
    max_pol = float(values[abs_idx])

    net_slope = (end_pol - start_pol) / len(values)

    deltas = np.diff(values)
    net_direction = end_pol - start_pol
    if net_direction == 0.0 or len(deltas) == 0:
        monotonicity_fraction = 0.0
    elif net_direction > 0:
        monotonicity_fraction = float(np.sum(deltas > 0) / len(deltas))
    else:
        monotonicity_fraction = float(np.sum(deltas < 0) / len(deltas))

    gradient_std = float(np.std(deltas))

    nmr_gaussian_r2 = float("nan")
    if raw_signal_df is not None and not raw_signal_df.empty and fit_gaussian_r2 is not None:
        mean_signal = raw_signal_df.mean(axis=0).values
        nmr_gaussian_r2 = fit_gaussian_r2(mean_signal)

    return {
        "start_pol": start_pol,
        "end_pol": end_pol,
        "max_pol": max_pol,
        "net_slope": net_slope,
        "monotonicity_fraction": monotonicity_fraction,
        "gradient_std": gradient_std,
        "nmr_gaussian_r2": nmr_gaussian_r2,
    }
