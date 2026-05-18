#!/usr/bin/env python3
"""Show polarization plots for a single event directory.

Loads '<events_dir>/<event_name>/<event_name>.csv', filters rows for the given
QMeterName and a tunable [min, max] uWaveFreq range, then displays:

  * Polarization vs uWaveFreq
  * Polarization vs row index (time proxy)

If '<event_name>-RawSignal.csv' exists, also writes a multi-page PDF with one
NMR raw-signal plot per row (default: '<event_name>_NMR_Signal.pdf').

Example:
    python plot_single_event.py "Top Proton" 2004-04-10_08h25m37s \\
        --min-freq 140.0 --max-freq 140.3
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

DEFAULT_MIN_FREQ = 138.0
DEFAULT_MAX_FREQ = float("inf")
nmr_signals = 100

DEFAULT_EVENTS_DIR = "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events"

logger = logging.getLogger("plot_single_event")


def load_event_rows(event_dir: Path, qmeter_name: str,
                    min_freq: float = DEFAULT_MIN_FREQ,
                    max_freq: float = DEFAULT_MAX_FREQ,
                    min_index: int | None = None,
                    max_index: int | None = None) -> pd.DataFrame:
    """Load rows for qmeter_name with numeric Polarization and uWaveFreq in
    [min_freq, max_freq], optionally restricted to row indices [min_index, max_index]
    (inclusive, 0-based positions after the QMeter + frequency filter).

    Setting min_freq == max_freq selects a single frequency.
    """
    csv_path = event_dir / f"{event_dir.name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    required = {"QMeterName", "Polarization", "uWaveFreq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    matched = df.loc[df["QMeterName"] == qmeter_name, ["uWaveFreq", "Polarization"]].copy()
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


def load_raw_signal_rows(event_dir: Path) -> pd.DataFrame:
    """Load '<event_dir>/<event_dir>-RawSignal.csv'.

    Headerless: column 0 is the event number, columns 1.. are NMR channels.
    Raises FileNotFoundError if missing, ValueError if too few columns.
    """
    csv_path = event_dir / f"{event_dir.name}-RawSignal.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"RawSignal CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, engine="python", on_bad_lines="skip")
    if df.shape[1] < 2:
        raise ValueError(f"RawSignal CSV {csv_path} has too few columns: {df.shape[1]}")
    return df


def plot_raw_signal(channels: pd.Series, title: str, event_number: int,
                    row_index: int | None = None) -> plt.Figure | None:
    channels = channels.dropna()
    if channels.empty:
        return None
    if row_index is not None:
        title = f"{title}  [index {row_index}]"
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(range(len(channels)), channels.values, linestyle="-", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("NMR channel")
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"Event: {event_number}  (n={len(channels)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def build_nmr_pdf(event_dir: Path, pdf_path: Path,
                  limit: int = nmr_signals,
                  min_index: int | None = None,
                  max_index: int | None = None) -> int:
    """Write NMR pages from the event's RawSignal CSV. Returns page count.

    If `min_index` or `max_index` is given, rows are sliced from the RawSignal
    CSV by 0-based row position (inclusive bounds) and `limit` is ignored.
    Otherwise the first `limit` rows are used.
    """
    df = load_raw_signal_rows(event_dir)
    if min_index is not None or max_index is not None:
        lo = 0 if min_index is None else max(0, min_index)
        hi = len(df) - 1 if max_index is None else max_index
        df = df.iloc[lo:hi + 1]
    elif limit is not None and limit >= 0:
        df = df.head(limit)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pages = 0
    with PdfPages(pdf_path) as out:
        for row in df.itertuples(index=True, name=None):
            row_index, event_number_raw, *channel_values = row
            try:
                event_number = int(float(event_number_raw))
            except (TypeError, ValueError):
                logger.debug("skipping raw row in %s: bad event number %r",
                             event_dir.name, event_number_raw)
                continue
            channels = pd.to_numeric(pd.Series(channel_values), errors="coerce")
            fig = plot_raw_signal(
                channels,
                f"{event_dir.name}  (event {event_number})",
                event_number,
                row_index=row_index,
            )
            if fig is not None:
                out.savefig(fig)
                plt.close(fig)
                pages += 1
        out.infodict()["Title"] = f"NMR raw signal — {event_dir.name}"
    return pages


def plot_pol_vs_freq(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure:
    pairs = rows.sort_values("uWaveFreq")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(pairs["uWaveFreq"], pairs["Polarization"], marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("uWaveFreq (GHz)")
    ax.set_ylabel("Polarization")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"QMeter: {qmeter_name}  (n={len(pairs)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def plot_pol_vs_time(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(rows.index, rows["Polarization"].values, marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Row index (time)")
    ax.set_ylabel("Polarization")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"QMeter: {qmeter_name}  (n={len(rows)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def _load_candidate_row(csv_path: str, row_number: int) -> dict:
    """Read a single row from a candidates.csv by 0-based row index."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if row_number < 0 or row_number >= len(rows):
        raise IndexError(
            f"Row {row_number} is out of range — candidates.csv has {len(rows)} row(s) "
            f"(valid indices: 0–{len(rows) - 1})"
        )
    return rows[row_number]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qmeter_name", nargs="?", default=None,
                        help='QMeterName to match exactly, e.g. "Top Proton". '
                             'Not required when --from-candidate is used.')
    parser.add_argument("event_name", nargs="?", default=None,
                        help='Event directory name, e.g. "2004-04-10_08h25m37s". '
                             'Not required when --from-candidate is used.')
    parser.add_argument(
        "--from-candidate",
        nargs=2,
        metavar=("CSV_PATH", "ROW_NUMBER"),
        default=None,
        help="Read event_name, qmeter_name, start_index, and end_index from row "
             "ROW_NUMBER (0-based) of a candidates.csv produced by scan_events.py. "
             "Overrides positional arguments and sets --min-index/--max-index automatically.",
    )
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR,
                        help="root directory containing event subdirs "
                             "(default: ../data/events relative to this script)")
    parser.add_argument("--min-freq", type=float, default=DEFAULT_MIN_FREQ,
                        help=f"keep only rows with uWaveFreq >= this value, in GHz "
                             f"(default: {DEFAULT_MIN_FREQ})")
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ,
                        help="keep only rows with uWaveFreq <= this value, in GHz "
                             "(default: no upper bound). Set equal to --min-freq "
                             "to select a single frequency.")
    parser.add_argument("--min-index", type=int, default=None,
                        help="keep only rows whose 0-based index (after the "
                             "QMeter + frequency filter) is >= this value. "
                             "When set, also overrides the nmr_signals cap "
                             "for slicing the RawSignal CSV.")
    parser.add_argument("--max-index", type=int, default=None,
                        help="keep only rows whose 0-based index is <= this "
                             "value (inclusive). When set, also overrides the "
                             "nmr_signals cap for slicing the RawSignal CSV.")
    parser.add_argument("--output-nmr", type=Path, default=None,
                        help="output PDF for NMR raw-signal pages, one per row of "
                             "<event_name>-RawSignal.csv "
                             "(default: <event_name>_NMR_Signal.pdf in cwd)")
    parser.add_argument("--no-nmr-pdf", action="store_true",
                        help="skip generating the NMR signal PDF")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Resolve --from-candidate, overriding positionals and index args
    if args.from_candidate is not None:
        csv_path, row_str = args.from_candidate
        try:
            row_number = int(row_str)
        except ValueError:
            logger.error("ROW_NUMBER must be an integer, got %r", row_str)
            return 1
        try:
            cand = _load_candidate_row(csv_path, row_number)
        except (FileNotFoundError, IndexError) as exc:
            logger.error("%s", exc)
            return 1
        args.qmeter_name = cand["qmeter_name"]
        args.event_name = cand["event_name"]
        args.min_index = int(cand["start_index"])
        args.max_index = int(cand["end_index"])
        logger.info(
            "Loaded candidate row %d: %s [%s] indices %s–%s",
            row_number, args.event_name, args.qmeter_name,
            args.min_index, args.max_index,
        )
    elif args.qmeter_name is None or args.event_name is None:
        logger.error(
            "qmeter_name and event_name are required unless --from-candidate is used."
        )
        return 1

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
                       args.qmeter_name, event_dir.name,
                       args.min_freq, args.max_freq,
                       args.min_index, args.max_index)
        return 2

    logger.info("loaded %d row(s) for %s in [%g, %g] GHz, index [%s, %s]",
                len(rows), event_dir.name,
                args.min_freq, args.max_freq,
                args.min_index, args.max_index)

    if args.min_freq == args.max_freq:
        title = f"{event_dir.name}  (f = {args.min_freq:g} GHz)"
    else:
        title = f"{event_dir.name}  ({args.min_freq:g} ≤ f ≤ {args.max_freq:g} GHz)"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    freq_fig = plot_pol_vs_freq(rows, title, args.qmeter_name)
    time_fig = plot_pol_vs_time(rows, title, args.qmeter_name)

    freq_path = plots_dir / f"{event_dir.name}_pol_vs_freq.png"
    time_path = plots_dir / f"{event_dir.name}_pol_vs_time.png"
    freq_fig.savefig(freq_path, dpi=150)
    time_fig.savefig(time_path, dpi=150)
    logger.info("saved %s", freq_path)
    logger.info("saved %s", time_path)

    if not args.no_nmr_pdf:
        nmr_pdf = args.output_nmr or plots_dir / f"{event_dir.name}_NMR_Signal.pdf"
        try:
            pages = build_nmr_pdf(event_dir, nmr_pdf,
                                  min_index=args.min_index,
                                  max_index=args.max_index)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping NMR PDF: %s", exc)
        else:
            if pages == 0:
                logger.warning("no NMR rows plotted — removing %s", nmr_pdf)
                nmr_pdf.unlink(missing_ok=True)
            else:
                logger.info("wrote %d NMR page(s) to %s", pages, nmr_pdf)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
