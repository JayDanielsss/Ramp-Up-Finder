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

Pass --from-candidate CSV_PATH [ROW_NUMBER] to drive the extraction from a
candidates.csv (as produced by scan_events.py / used by plot_single_event.py).
With ROW_NUMBER: extract that one candidate. Without ROW_NUMBER: extract
every candidate. Outputs go to '<output-dir>/<candidates-stem>/' with
filenames '<event>-<qmeter>-<start_line>-<end_line>.csv' plus a matching
'-RawSignal.csv' (RawSignal rows are selected by EventNum, not position).

Example:
    python extract_event_slice.py --from-candidate 14NH3.csv
    python extract_event_slice.py --from-candidate 14NH3.csv 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable
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
                    max_index: int | None = None,
                    track_line: bool = False) -> pd.DataFrame:
    """Load all columns for qmeter_name with uWaveFreq in [min_freq, max_freq],
    optionally restricted to 0-based row positions [min_index, max_index]
    (inclusive, counted after the QMeter + frequency filter).

    When track_line=True, a 'csv_line' column carrying the source CSV line
    number (header = line 1, first data row = line 2) is added before slicing.
    """
    csv_path = event_dir / f"{event_dir.name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    if track_line:
        df["csv_line"] = df.index + 2

    required = {"QMeterName", "Polarization", "uWaveFreq", "EventNum"}
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


def _load_candidate_row(csv_path: str, row_number: int) -> dict:
    """Read a single row from a candidates.csv by 1-based line number (header = line 1)."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    data_index = row_number - 2
    if data_index < 0 or data_index >= len(rows):
        raise IndexError(
            f"Line {row_number} is out of range — candidates.csv has {len(rows)} data row(s) "
            f"(valid line numbers: 2–{len(rows) + 1})"
        )
    return rows[data_index]


def _load_all_candidates(csv_path: str) -> list[dict]:
    """Read all data rows from a candidates.csv, skipping blank trailing rows."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        return [r for r in reader if r.get("event_name") and r.get("qmeter_name")]


def _sanitize_qmeter(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_") or "qmeter"


def load_raw_signal_by_event_nums(event_dir: Path,
                                  event_nums: Iterable[int]) -> pd.DataFrame:
    """Return RawSignal rows whose first column (EventNum) is in *event_nums*."""
    csv_path = event_dir / f"{event_dir.name}-RawSignal.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"RawSignal CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, engine="python", on_bad_lines="skip")
    if df.shape[1] < 2:
        raise ValueError(f"RawSignal CSV {csv_path} has too few columns: {df.shape[1]}")
    nums = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return df.loc[nums.isin(list(event_nums))]


def extract_candidate_slice(event_dir: Path, qmeter_name: str,
                            start_line: int, end_line: int,
                            min_freq: float, max_freq: float,
                            output_dir: Path,
                            include_raw: bool) -> Path | None:
    """Extract one candidate's slice; returns the main CSV path or None on skip."""
    try:
        rows = load_event_rows(event_dir, qmeter_name,
                               min_freq=min_freq, max_freq=max_freq,
                               track_line=True)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("skipping %s: %s", event_dir.name, exc)
        return None

    rows = rows[(rows["csv_line"] >= start_line) & (rows["csv_line"] <= end_line)]
    if rows.empty:
        logger.warning("no surviving rows for %s [%s] in lines %d-%d — skipping",
                       event_dir.name, qmeter_name, start_line, end_line)
        return None

    stem = f"{event_dir.name}-{_sanitize_qmeter(qmeter_name)}-{start_line}-{end_line}"
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / f"{stem}.csv"
    rows.to_csv(main_path, index=False)
    logger.info("wrote %d row(s) to %s", len(rows), main_path)

    if include_raw:
        event_nums = pd.to_numeric(rows["EventNum"], errors="coerce").dropna().astype(int)
        try:
            raw = load_raw_signal_by_event_nums(event_dir, event_nums)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping RawSignal for %s: %s", event_dir.name, exc)
        else:
            if raw.empty:
                logger.warning("no RawSignal rows matched for %s [%s] lines %d-%d",
                               event_dir.name, qmeter_name, start_line, end_line)
            else:
                raw_path = output_dir / f"{stem}-RawSignal.csv"
                raw.to_csv(raw_path, index=False, header=False)
                logger.info("wrote %d RawSignal row(s) to %s", len(raw), raw_path)
    return main_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qmeter_name", nargs="?", default=None,
                        help='QMeterName to match exactly, e.g. "Top Proton". '
                             'Not required when --from-candidate is used.')
    parser.add_argument("event_name", nargs="?", default=None,
                        help='Event directory name, e.g. "2005-03-03_10h41m57s". '
                             'Not required when --from-candidate is used.')
    parser.add_argument(
        "--from-candidate", nargs="+", metavar="ARG", default=None,
        help="CSV_PATH [ROW_NUMBER] — drive extraction from a candidates.csv "
             "(header = line 1, first data row = line 2). With ROW_NUMBER: "
             "extract that one candidate. Without ROW_NUMBER: extract every "
             "candidate. Outputs go under '<output-dir>/<candidates-stem>/'. "
             "Overrides positional args and --min-index/--max-index.",
    )
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

    if args.from_candidate is not None:
        if len(args.from_candidate) > 2:
            logger.error("--from-candidate takes 1 or 2 arguments: CSV_PATH [ROW_NUMBER]")
            return 1
        csv_path_str = args.from_candidate[0]
        if len(args.from_candidate) == 2:
            try:
                row_number = int(args.from_candidate[1])
            except ValueError:
                logger.error("ROW_NUMBER must be an integer, got %r",
                             args.from_candidate[1])
                return 1
            try:
                cands = [_load_candidate_row(csv_path_str, row_number)]
            except (FileNotFoundError, IndexError) as exc:
                logger.error("%s", exc)
                return 1
        else:
            try:
                cands = _load_all_candidates(csv_path_str)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            if not cands:
                logger.error("no candidates found in %s", csv_path_str)
                return 2

        out_root = args.output_dir / Path(csv_path_str).stem
        wrote = 0
        for cand in cands:
            cand_event = cand.get("event_name", "")
            cand_qmeter = cand.get("qmeter_name", "")
            try:
                start_line = int(cand["start_line"])
                end_line = int(cand["end_line"])
            except (KeyError, ValueError):
                logger.warning("malformed candidate row %r — skipping", cand)
                continue
            event_dir = args.events_dir / cand_event
            if not event_dir.is_dir():
                logger.warning("event directory not found: %s — skipping", event_dir)
                continue
            if extract_candidate_slice(
                event_dir, cand_qmeter, start_line, end_line,
                min_freq=args.min_freq, max_freq=args.max_freq,
                output_dir=out_root,
                include_raw=not args.no_raw_signal,
            ) is not None:
                wrote += 1
        logger.info("extracted %d of %d candidate(s) into %s",
                    wrote, len(cands), out_root)
        return 0 if wrote else 2

    if args.qmeter_name is None or args.event_name is None:
        logger.error("qmeter_name and event_name are required unless --from-candidate is used")
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
