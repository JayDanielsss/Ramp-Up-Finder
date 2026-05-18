#!/usr/bin/env python3
"""Extract a slice of event data to CSV files without plotting.

Loads the main event CSV and the RawSignal CSV for the given event,
applies QMeterName + frequency + index filters, and writes the matching
rows to new CSV files under a 'rampUps' output directory.

Output filenames encode the slice bounds, e.g.:
    rampUps/2005-03-03_10h41m57s-800-1200.csv
    rampUps/2005-03-03_10h41m57s-800-1200-RawSignal.csv

Example:
    python extract_event_slice.py "Top Proton" 2005-03-03_10h41m57s \\
        --min-index 800 --max-index 1200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

DEFAULT_MIN_FREQ = 138.0
DEFAULT_MAX_FREQ = float("inf")
DEFAULT_EVENTS_DIR = "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "rampUps"

logger = logging.getLogger("extract_event_slice")


def load_event_rows(event_dir: Path, qmeter_name: str,
                    min_freq: float = DEFAULT_MIN_FREQ,
                    max_freq: float = DEFAULT_MAX_FREQ,
                    min_index: int | None = None,
                    max_index: int | None = None) -> pd.DataFrame:
    """Load all columns for qmeter_name with uWaveFreq in [min_freq, max_freq],
    optionally restricted to 0-based row positions [min_index, max_index]
    (inclusive, counted after the QMeter + frequency filter).
    """
    csv_path = event_dir / f"{event_dir.name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    required = {"QMeterName", "Polarization", "uWaveFreq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    matched = df[df["QMeterName"] == qmeter_name].copy()
    matched["uWaveFreq"] = pd.to_numeric(matched["uWaveFreq"], errors="coerce")
    matched["Polarization"] = pd.to_numeric(matched["Polarization"], errors="coerce")
    matched = matched.dropna(subset=["uWaveFreq", "Polarization"])
    matched = matched[(matched["uWaveFreq"] >= min_freq) & (matched["uWaveFreq"] <= max_freq)]
    matched = matched.reset_index(drop=True)

    if min_index is not None or max_index is not None:
        lo = 0 if min_index is None else min_index
        hi = len(matched) - 1 if max_index is None else max_index
        matched = matched.loc[(matched.index >= lo) & (matched.index <= hi)]

    return matched


def load_raw_signal_rows(event_dir: Path,
                         min_index: int | None = None,
                         max_index: int | None = None) -> pd.DataFrame:
    """Load and slice the RawSignal CSV by 0-based row position."""
    csv_path = event_dir / f"{event_dir.name}-RawSignal.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"RawSignal CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, engine="python", on_bad_lines="skip")
    if df.shape[1] < 2:
        raise ValueError(f"RawSignal CSV {csv_path} has too few columns: {df.shape[1]}")

    if min_index is not None or max_index is not None:
        lo = 0 if min_index is None else max(0, min_index)
        hi = len(df) - 1 if max_index is None else max_index
        df = df.iloc[lo:hi + 1]

    return df


def build_output_stem(event_name: str, min_freq: float, max_freq: float,
                      min_index: int | None, max_index: int | None) -> str:
    parts = [event_name]
    if min_freq != DEFAULT_MIN_FREQ or max_freq != float("inf"):
        parts.append(f"f{min_freq:g}-{max_freq:g}" if max_freq != float("inf")
                     else f"f{min_freq:g}+")
    lo = str(min_index) if min_index is not None else "0"
    hi = str(max_index) if max_index is not None else "end"
    parts.append(f"{lo}-{hi}")
    return "-".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qmeter_name",
                        help='QMeterName to match exactly, e.g. "Top Proton"')
    parser.add_argument("event_name",
                        help='Event directory name, e.g. "2005-03-03_10h41m57s"')
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR,
                        help="root directory containing event subdirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory to write output CSVs (default: rampUps/ next to this script)")
    parser.add_argument("--min-freq", type=float, default=DEFAULT_MIN_FREQ,
                        help=f"keep rows with uWaveFreq >= this value in GHz "
                             f"(default: {DEFAULT_MIN_FREQ})")
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ,
                        help="keep rows with uWaveFreq <= this value in GHz "
                             "(default: no upper bound)")
    parser.add_argument("--min-index", type=int, default=None,
                        help="first 0-based row index to include (after QMeter + freq filter)")
    parser.add_argument("--max-index", type=int, default=None,
                        help="last 0-based row index to include (inclusive)")
    parser.add_argument("--no-raw-signal", action="store_true",
                        help="skip extracting the RawSignal CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.min_freq > args.max_freq:
        logger.error("--min-freq (%g) must not exceed --max-freq (%g)",
                     args.min_freq, args.max_freq)
        return 1

    if (args.min_index is not None and args.max_index is not None
            and args.min_index > args.max_index):
        logger.error("--min-index (%d) must not exceed --max-index (%d)",
                     args.min_index, args.max_index)
        return 1

    event_dir = args.events_dir / args.event_name
    if not event_dir.is_dir():
        logger.error("event directory not found: %s", event_dir)
        return 1

    stem = build_output_stem(args.event_name, args.min_freq, args.max_freq,
                              args.min_index, args.max_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Main event CSV
    try:
        rows = load_event_rows(event_dir, args.qmeter_name,
                               min_freq=args.min_freq, max_freq=args.max_freq,
                               min_index=args.min_index, max_index=args.max_index)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if rows.empty:
        logger.warning("no matching rows for QMeter=%r in %s with %g <= uWaveFreq <= %g, "
                       "index [%s, %s]",
                       args.qmeter_name, args.event_name,
                       args.min_freq, args.max_freq,
                       args.min_index, args.max_index)
        return 2

    out_csv = args.output_dir / f"{stem}.csv"
    rows.to_csv(out_csv, index=False)
    logger.info("wrote %d row(s) to %s", len(rows), out_csv)

    # RawSignal CSV
    if not args.no_raw_signal:
        try:
            raw = load_raw_signal_rows(event_dir,
                                       min_index=args.min_index,
                                       max_index=args.max_index)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping RawSignal: %s", exc)
        else:
            out_raw = args.output_dir / f"{stem}-RawSignal.csv"
            raw.to_csv(out_raw, index=False, header=False)
            logger.info("wrote %d RawSignal row(s) to %s", len(raw), out_raw)

    return 0


if __name__ == "__main__":
    sys.exit(main())
