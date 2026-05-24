#!/usr/bin/env python3
"""Show polarization plots for a single event directory.

Loads '<events_dir>/<event_name>/<event_name>.csv', filters rows for the given
QMeterName and a tunable [min, max] uWaveFreq range, then saves to plots/:

  * <event>_pol_vs_freq.png  — Polarization vs uWaveFreq
  * <event>_pol_vs_time.png  — Polarization vs Eastern timestamp

Pass --nmr to also write a PDF of NMR raw-signal traces (±5 rows around the
peak polarization).  Accepts --min-time/--max-time in ISO format or the
compact 'YYYY-Mon-DD_HHMM' form (e.g. '2004-Apr-09_1300').

Pass --all-qmeters (with event_name only, no QMeterName) to plot every
QMeter in one event together. Emits two PNGs into plots/:

  * <event>_all_pol_vs_time.png  — Polarization vs time, with red vertical
    lines + rotated labels at each QMeter change
  * <event>_all_freq_vs_time.png — uWaveFreq vs time, same overlay

Examples:
    python plot_single_event.py "Top Proton" 2004-04-10_08h25m37s

    python plot_single_event.py "Top Proton" 2004-04-10_08h25m37s \\
        --min-time "2004-Apr-09_1300" --max-time "2004-Apr-09_1420" --nmr

    python plot_single_event.py --all-qmeters 2004-04-16_10h35m39s
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

from time_utils import EASTERN, parse_eastern, format_eastern

DEFAULT_MIN_FREQ = 138.0
DEFAULT_MAX_FREQ = float("inf")
nmr_signals = 100
nmr_peak_window = 5  # batch mode: ±N rows around the candidate's peak

DEFAULT_EVENTS_DIR = "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events"

logger = logging.getLogger("plot_single_event")


def load_event_rows(event_dir: Path, qmeter_name: str | None = None,
                    min_freq: float = DEFAULT_MIN_FREQ,
                    max_freq: float = DEFAULT_MAX_FREQ,
                    min_index: int | None = None,
                    max_index: int | None = None,
                    min_time_unix: int | None = None,
                    max_time_unix: int | None = None) -> pd.DataFrame:
    """Load rows from the event's CSV with numeric Polarization, uWaveFreq in
    [min_freq, max_freq], and a tz-aware Eastern Timestamp from EventNum.

    If *qmeter_name* is a string, rows are filtered to that QMeter and the
    returned frame does not include a QMeterName column (legacy behavior).
    If *qmeter_name* is None, rows from every QMeter are returned and
    QMeterName is included as a column so callers can detect transitions.

    Optionally restricted by row position [min_index, max_index] (inclusive,
    0-based after the QMeter + frequency filter) and/or by EventNum Unix
    seconds in [min_time_unix, max_time_unix]. When both kinds of filters
    are given they are intersected.

    Setting min_freq == max_freq selects a single frequency.
    """
    csv_path = event_dir / f"{event_dir.name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    # Line 1 is the header, so the first data row sits on line 2.
    df["csv_line"] = df.index + 2

    required = {"QMeterName", "Polarization", "uWaveFreq", "EventNum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    if qmeter_name is None:
        cols = ["uWaveFreq", "Polarization", "EventNum", "csv_line", "QMeterName"]
        matched = df.loc[:, cols].copy()
    else:
        cols = ["uWaveFreq", "Polarization", "EventNum", "csv_line"]
        matched = df.loc[df["QMeterName"] == qmeter_name, cols].copy()
    matched["uWaveFreq"] = pd.to_numeric(matched["uWaveFreq"], errors="coerce")
    matched["Polarization"] = pd.to_numeric(matched["Polarization"], errors="coerce")
    matched["EventNum"] = pd.to_numeric(matched["EventNum"], errors="coerce")
    matched = matched.dropna(subset=["uWaveFreq", "Polarization", "EventNum"])
    matched = matched[(matched["uWaveFreq"] >= min_freq) & (matched["uWaveFreq"] <= max_freq)]
    if min_time_unix is not None:
        matched = matched[matched["EventNum"] >= min_time_unix]
    if max_time_unix is not None:
        matched = matched[matched["EventNum"] <= max_time_unix]
    matched["Timestamp"] = pd.to_datetime(matched["EventNum"], unit="s", utc=True
                                          ).dt.tz_convert(EASTERN)
    matched = matched.reset_index(drop=True)

    if min_index is not None or max_index is not None:
        lo = 0 if min_index is None else min_index
        hi = len(matched) - 1 if max_index is None else max_index
        matched = matched.loc[(matched.index >= lo) & (matched.index <= hi)]
    return matched


def _raw_signal_index_for(event_dir: Path, target_event_num: int) -> int | None:
    """Return the 0-based row index in the RawSignal CSV whose event number (col 0) matches."""
    df = load_raw_signal_rows(event_dir)
    nums = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    hits = (nums == target_event_num).to_numpy().nonzero()[0]
    return int(hits[0]) if len(hits) else None


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
    eastern_str = format_eastern(event_number)
    ax.text(0.99, 0.01, f"Event: {event_number}  |  {eastern_str}  (n={len(channels)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def _write_nmr_pages(out: PdfPages, event_dir: Path,
                     min_index: int | None = None,
                     max_index: int | None = None,
                     limit: int | None = None,
                     title_prefix: str | None = None) -> int:
    """Append NMR pages from event_dir's RawSignal CSV to an open PdfPages.

    Returns pages written. `limit` (if not None) caps how many rows are taken
    from the sliced range.
    """
    df = load_raw_signal_rows(event_dir)
    if min_index is not None or max_index is not None:
        lo = 0 if min_index is None else max(0, min_index)
        hi = len(df) - 1 if max_index is None else max_index
        df = df.iloc[lo:hi + 1]
    if limit is not None and limit >= 0:
        df = df.head(limit)

    pages = 0
    for row in df.itertuples(index=True, name=None):
        row_index, event_number_raw, *channel_values = row
        try:
            event_number = int(float(event_number_raw))
        except (TypeError, ValueError):
            logger.debug("skipping raw row in %s: bad event number %r",
                         event_dir.name, event_number_raw)
            continue
        channels = pd.to_numeric(pd.Series(channel_values), errors="coerce")
        title = title_prefix or f"{event_dir.name}  (event {event_number})"
        fig = plot_raw_signal(channels, title, event_number, row_index=row_index)
        if fig is not None:
            out.savefig(fig)
            plt.close(fig)
            pages += 1
    return pages


def build_nmr_pdf(event_dir: Path, pdf_path: Path,
                  limit: int = nmr_signals,
                  min_index: int | None = None,
                  max_index: int | None = None) -> int:
    """Write NMR pages from the event's RawSignal CSV. Returns page count.

    If `min_index` or `max_index` is given, rows are sliced from the RawSignal
    CSV by 0-based row position (inclusive bounds) and `limit` is ignored.
    Otherwise the first `limit` rows are used.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    sliced = (min_index is not None or max_index is not None)
    effective_limit = None if sliced else limit
    with PdfPages(pdf_path) as out:
        pages = _write_nmr_pages(out, event_dir,
                                 min_index=min_index, max_index=max_index,
                                 limit=effective_limit)
        out.infodict()["Title"] = f"NMR raw signal — {event_dir.name}"
    return pages


def _add_index_axis(ax: plt.Axes, n: int, start: int = 0, end: int | None = None) -> None:
    """Attach a secondary x-axis at the bottom of *ax* showing CSV line numbers.

    *start* and *end* are the first and last line numbers of the data window.
    *end* defaults to ``start + n - 1`` when omitted.
    """
    hi = end if end is not None else start + n - 1
    ax2 = ax.twiny()
    ax2.set_xlim(start, max(hi, start + 1))
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 48))
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("Line", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8)


def plot_pol_vs_freq(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure:
    pairs = rows.sort_values("uWaveFreq")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(pairs["uWaveFreq"], pairs["Polarization"], marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("uWaveFreq (GHz)", labelpad=30)
    ax.set_ylabel("Polarization")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    start_ts = rows["Timestamp"].min()
    start_str = start_ts.strftime("%Y-%m-%d %H:%M %Z")
    ax.text(0.99, 0.01,
            f"QMeter: {qmeter_name}  |  Start date: {start_str}  (n={len(pairs)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    line_start = int(pairs["csv_line"].min()) if not pairs.empty else 0
    line_end   = int(pairs["csv_line"].max()) if not pairs.empty else 0
    _add_index_axis(ax, len(pairs), start=line_start, end=line_end)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


def _format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=EASTERN))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _overlay_qmeter_transitions(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """Draw a red vertical line + rotated label at each QMeter change.

    Requires *rows* to contain a 'QMeterName' column and a 'Timestamp'
    column, sorted in CSV order. The first row is treated as a transition
    so every segment gets a label. NaN QMeterName values are rendered as
    "(unknown)" so they don't trigger spurious transitions.
    """
    if "QMeterName" not in rows.columns or rows.empty:
        return
    names = rows["QMeterName"].fillna("(unknown)")
    is_change = names != names.shift()
    transitions = rows.loc[is_change, ["Timestamp"]].copy()
    transitions["QMeterName"] = names.loc[is_change]
    for ts, name in zip(transitions["Timestamp"], transitions["QMeterName"]):
        ax.axvline(ts, color="red", linewidth=0.8, alpha=0.7)
        ax.text(ts, 0.98, str(name),
                transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right",
                fontsize=8, color="red")


def plot_pol_vs_time(rows: pd.DataFrame, title: str, qmeter_name: str | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(rows["Timestamp"], rows["Polarization"], marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (Eastern)")
    ax.set_ylabel("Polarization")
    _format_time_axis(ax)
    ax.grid(True, alpha=0.3)
    start_ts = rows["Timestamp"].min()
    start_str = start_ts.strftime("%Y-%m-%d %H:%M %Z")
    if "QMeterName" in rows.columns:
        distinct = rows["QMeterName"].fillna("(unknown)").nunique()
        footer = f"QMeters: {distinct}  |  Start date: {start_str}  (n={len(rows)})"
    else:
        footer = f"QMeter: {qmeter_name}  |  Start date: {start_str}  (n={len(rows)})"
    ax.text(0.99, 0.01, footer,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    line_start = int(rows["csv_line"].min()) if not rows.empty else 0
    line_end   = int(rows["csv_line"].max()) if not rows.empty else 0
    _add_index_axis(ax, len(rows), start=line_start, end=line_end)
    _overlay_qmeter_transitions(ax, rows)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


def plot_freq_vs_time(rows: pd.DataFrame, title: str, qmeter_name: str | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(rows["Timestamp"], rows["uWaveFreq"], marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (Eastern)")
    ax.set_ylabel("uWaveFreq (GHz)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    _format_time_axis(ax)
    ax.grid(True, alpha=0.3)
    start_ts = rows["Timestamp"].min()
    start_str = start_ts.strftime("%Y-%m-%d %H:%M %Z")
    if "QMeterName" in rows.columns:
        distinct = rows["QMeterName"].fillna("(unknown)").nunique()
        footer = f"QMeters: {distinct}  |  Start date: {start_str}  (n={len(rows)})"
    else:
        footer = f"QMeter: {qmeter_name}  |  Start date: {start_str}  (n={len(rows)})"
    ax.text(0.99, 0.01, footer,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    line_start = int(rows["csv_line"].min()) if not rows.empty else 0
    line_end   = int(rows["csv_line"].max()) if not rows.empty else 0
    _add_index_axis(ax, len(rows), start=line_start, end=line_end)
    _overlay_qmeter_transitions(ax, rows)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


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
    """Read all data rows from a candidates.csv."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qmeter_name", nargs="?", default=None,
                        help='QMeterName to match exactly, e.g. "Top Proton". '
                             'Not required when --from-candidate or '
                             '--all-qmeters is used.')
    parser.add_argument("event_name", nargs="?", default=None,
                        help='Event directory name, e.g. "2004-04-10_08h25m37s". '
                             'Not required when --from-candidate is used. '
                             'The sole positional when --all-qmeters is used.')
    parser.add_argument(
        "--from-candidate",
        nargs="+",
        metavar="ARG",
        default=None,
        help="CSV_PATH [ROW_NUMBER] — read from a candidates.csv produced by "
             "scan_events.py (header = line 1, first data row = line 2). "
             "With ROW_NUMBER: plot that single candidate (overrides positional "
             "args and sets --min-index/--max-index automatically). "
             "Without ROW_NUMBER: plot every candidate and write two PDFs "
             "(pol_vs_freq and pol_vs_time) without showing the window.",
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
    parser.add_argument("--min-time", type=str, default=None,
                        help="keep only rows whose EventNum (Eastern time) is "
                             ">= this value. Accepted formats: "
                             "'YYYY-MM-DD HH:MM[:SS]', 'YYYY-MM-DDTHH:MM', "
                             "or 'YYYY-Mon-DD_HHMM' (e.g. '2004-Apr-09_1300'). "
                             "Interpreted as America/New_York (EST/EDT).")
    parser.add_argument("--max-time", type=str, default=None,
                        help="keep only rows whose EventNum (Eastern time) is "
                             "<= this value. Same formats as --min-time. "
                             "Intersected with --min-index/--max-index when both "
                             "are given.")
    parser.add_argument("--output-nmr", type=Path, default=None,
                        help="output PDF for NMR raw-signal pages, one per row of "
                             "<event_name>-RawSignal.csv "
                             "(default: <event_name>_NMR_Signal.pdf in cwd)")
    parser.add_argument("--nmr", action="store_true",
                        help="generate NMR signal PDF (±%d rows around peak polarization)" % nmr_peak_window)
    parser.add_argument("--all-qmeters", action="store_true",
                        help="plot every QMeter in the event (no QMeter filter). "
                             "Draws red vertical lines + rotated labels at each "
                             "QMeter change. Emits only the time-axis PNGs "
                             "(no pol_vs_freq, no NMR). Incompatible with a "
                             "positional qmeter_name, --from-candidate, and --nmr.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.all_qmeters:
        # argparse fills positionals left-to-right, so a single positional
        # lands in args.qmeter_name. Reinterpret it as event_name when
        # event_name is empty, so `--all-qmeters <event>` works as users expect.
        if args.event_name is None and args.qmeter_name is not None:
            args.event_name = args.qmeter_name
            args.qmeter_name = None
        if args.qmeter_name is not None:
            logger.error("--all-qmeters: supply only event_name, not both positionals")
            return 1
        if args.from_candidate is not None:
            logger.error("--all-qmeters cannot be combined with --from-candidate")
            return 1
        if args.nmr:
            logger.error("--all-qmeters cannot be combined with --nmr")
            return 1

    # Resolve --from-candidate, overriding positionals and index args
    if args.from_candidate is not None:
        if len(args.from_candidate) > 2:
            logger.error("--from-candidate takes 1 or 2 arguments: CSV_PATH [ROW_NUMBER]")
            return 1
        csv_path_str = args.from_candidate[0]

        if len(args.from_candidate) == 1:
            # batch mode: plot every candidate and save to two PDFs
            try:
                all_cands = _load_all_candidates(csv_path_str)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            if not all_cands:
                logger.error("no candidates found in %s", csv_path_str)
                return 2

            plots_dir = Path(__file__).parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            csv_stem = Path(csv_path_str).stem
            freq_pdf_path = plots_dir / f"{csv_stem}_pol_vs_freq.pdf"
            time_pdf_path = plots_dir / f"{csv_stem}_pol_vs_time.pdf"
            freq_time_pdf_path = plots_dir / f"{csv_stem}_freq_vs_time.pdf"
            nmr_pdf_path = plots_dir / f"{csv_stem}_nmr_signals.pdf"
            freq_pages = time_pages = freq_time_pages = nmr_pages = 0

            with contextlib.ExitStack() as stack:
                freq_out = stack.enter_context(PdfPages(freq_pdf_path))
                time_out = stack.enter_context(PdfPages(time_pdf_path))
                freq_time_out = stack.enter_context(PdfPages(freq_time_pdf_path))
                nmr_out = stack.enter_context(PdfPages(nmr_pdf_path)) if args.nmr else None

                for i, cand in enumerate(all_cands):
                    cand_qmeter = cand["qmeter_name"]
                    cand_event = cand["event_name"]
                    cand_start_line = int(cand["start_line"])
                    cand_end_line = int(cand["end_line"])
                    event_dir = args.events_dir / cand_event
                    if not event_dir.is_dir():
                        logger.warning("event directory not found: %s — skipping", event_dir)
                        continue
                    try:
                        all_filtered = load_event_rows(event_dir, cand_qmeter,
                                                       min_freq=args.min_freq, max_freq=args.max_freq)
                    except (FileNotFoundError, ValueError) as exc:
                        logger.warning("skipping %s: %s", cand_event, exc)
                        continue
                    hits_s = all_filtered.index[all_filtered["csv_line"] == cand_start_line]
                    hits_e = all_filtered.index[all_filtered["csv_line"] == cand_end_line]
                    if len(hits_s) == 0 or len(hits_e) == 0:
                        logger.warning("lines %d-%d not in %s [%s] — skipping",
                                       cand_start_line, cand_end_line, cand_event, cand_qmeter)
                        continue
                    cand_min_idx = int(hits_s[0])
                    cand_max_idx = int(hits_e[0])
                    rows = all_filtered.loc[(all_filtered.index >= cand_min_idx)
                                            & (all_filtered.index <= cand_max_idx)]
                    if rows.empty:
                        logger.warning("no matching rows for %s [%s] — skipping",
                                       cand_event, cand_qmeter)
                        continue
                    if args.min_freq == args.max_freq:
                        title = f"{cand_event}  (f = {args.min_freq:g} GHz)  [{cand_qmeter}]"
                    else:
                        title = (f"{cand_event}  ({args.min_freq:g} ≤ f ≤ {args.max_freq:g} GHz)"
                                 f"  [{cand_qmeter}]")
                    freq_fig = plot_pol_vs_freq(rows, title, cand_qmeter)
                    freq_out.savefig(freq_fig)
                    plt.close(freq_fig)
                    freq_pages += 1
                    time_fig = plot_pol_vs_time(rows, title, cand_qmeter)
                    time_out.savefig(time_fig)
                    plt.close(time_fig)
                    time_pages += 1
                    freq_time_fig = plot_freq_vs_time(rows, title, cand_qmeter)
                    freq_time_out.savefig(freq_time_fig)
                    plt.close(freq_time_fig)
                    freq_time_pages += 1
                    if args.nmr and nmr_out is not None:
                        try:
                            direction = int(cand.get("direction") or 0)
                        except ValueError:
                            direction = 0
                        if direction == -1:
                            peak_idx = int(rows["Polarization"].idxmin())
                        else:
                            peak_idx = int(rows["Polarization"].idxmax())
                        nmr_lo = peak_idx - nmr_peak_window
                        nmr_hi = peak_idx + nmr_peak_window
                        try:
                            written = _write_nmr_pages(
                                nmr_out, event_dir,
                                min_index=nmr_lo, max_index=nmr_hi,
                                limit=None,
                                title_prefix=f"{cand_event}  [{cand_qmeter}]  peak@{peak_idx}",
                            )
                            nmr_pages += written
                        except (FileNotFoundError, ValueError) as exc:
                            logger.warning("skipping NMR for %s: %s", cand_event, exc)
                    logger.debug("processed row %d: %s [%s]", i + 2, cand_event, cand_qmeter)
                freq_out.infodict()["Title"] = f"Polarization vs uWaveFreq — {csv_stem}"
                time_out.infodict()["Title"] = f"Polarization vs time — {csv_stem}"
                freq_time_out.infodict()["Title"] = f"uWaveFreq vs time — {csv_stem}"
                if nmr_out is not None:
                    nmr_out.infodict()["Title"] = f"NMR raw signal — {csv_stem}"

            summaries = [("pol_vs_freq", freq_pdf_path, freq_pages),
                         ("pol_vs_time", time_pdf_path, time_pages),
                         ("freq_vs_time", freq_time_pdf_path, freq_time_pages)]
            if args.nmr:
                summaries.append(("nmr_signals", nmr_pdf_path, nmr_pages))
            else:
                nmr_pdf_path.unlink(missing_ok=True)
            for label, path, pages in summaries:
                if pages == 0:
                    logger.warning("no pages for %s PDF — removing %s", label, path)
                    path.unlink(missing_ok=True)
                else:
                    logger.info("wrote %d page(s) to %s", pages, path)
            return 0 if (freq_pages or time_pages or freq_time_pages) else 2

        # single-row mode
        row_str = args.from_candidate[1]
        try:
            row_number = int(row_str)
        except ValueError:
            logger.error("ROW_NUMBER must be an integer, got %r", row_str)
            return 1
        try:
            cand = _load_candidate_row(csv_path_str, row_number)
        except (FileNotFoundError, IndexError) as exc:
            logger.error("%s", exc)
            return 1
        args.qmeter_name = cand["qmeter_name"]
        args.event_name = cand["event_name"]
        cand_start_line = int(cand["start_line"])
        cand_end_line = int(cand["end_line"])
        event_dir = args.events_dir / args.event_name
        try:
            prelim = load_event_rows(event_dir, args.qmeter_name,
                                     min_freq=args.min_freq, max_freq=args.max_freq)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        hits_s = prelim.index[prelim["csv_line"] == cand_start_line]
        hits_e = prelim.index[prelim["csv_line"] == cand_end_line]
        if len(hits_s) == 0 or len(hits_e) == 0:
            logger.error("candidate lines %d-%d not found in %s [%s] after filter",
                         cand_start_line, cand_end_line, args.event_name, args.qmeter_name)
            return 1
        args.min_index = int(hits_s[0])
        args.max_index = int(hits_e[0])
        logger.info(
            "Loaded candidate line %d: %s [%s] lines %d-%d (filtered indices %s-%s)",
            row_number, args.event_name, args.qmeter_name,
            cand_start_line, cand_end_line, args.min_index, args.max_index,
        )
    elif args.all_qmeters:
        if args.event_name is None:
            logger.error("event_name is required when --all-qmeters is used.")
            return 1
    elif args.qmeter_name is None or args.event_name is None:
        logger.error(
            "qmeter_name and event_name are required unless --from-candidate "
            "or --all-qmeters is used."
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

    min_time_unix: int | None = None
    max_time_unix: int | None = None
    try:
        if args.min_time is not None:
            min_time_unix = parse_eastern(args.min_time)
        if args.max_time is not None:
            max_time_unix = parse_eastern(args.max_time)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    if (min_time_unix is not None and max_time_unix is not None
            and min_time_unix > max_time_unix):
        logger.error("--min-time (%s) must not exceed --max-time (%s)",
                     args.min_time, args.max_time)
        return 1

    event_dir = args.events_dir / args.event_name
    if not event_dir.is_dir():
        logger.error("event directory not found: %s", event_dir)
        return 1

    loader_qmeter = None if args.all_qmeters else args.qmeter_name
    try:
        rows = load_event_rows(event_dir, loader_qmeter,
                               min_freq=args.min_freq, max_freq=args.max_freq,
                               min_index=args.min_index, max_index=args.max_index,
                               min_time_unix=min_time_unix,
                               max_time_unix=max_time_unix)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    time_lo_disp = format_eastern(min_time_unix) if min_time_unix is not None else None
    time_hi_disp = format_eastern(max_time_unix) if max_time_unix is not None else None

    if rows.empty:
        qmeter_disp = "<all>" if args.all_qmeters else repr(args.qmeter_name)
        logger.warning("no matching rows for QMeter=%s in %s with %g <= uWaveFreq <= %g, "
                       "index [%s, %s], time [%s, %s]",
                       qmeter_disp, event_dir.name,
                       args.min_freq, args.max_freq,
                       args.min_index, args.max_index,
                       time_lo_disp, time_hi_disp)
        return 2

    logger.info("loaded %d row(s) for %s in [%g, %g] GHz, index [%s, %s], time [%s, %s]",
                len(rows), event_dir.name,
                args.min_freq, args.max_freq,
                args.min_index, args.max_index,
                time_lo_disp, time_hi_disp)

    if args.all_qmeters:
        if args.min_freq == args.max_freq:
            title = f"{event_dir.name}  [all QMeters]  (f = {args.min_freq:g} GHz)"
        else:
            title = (f"{event_dir.name}  [all QMeters]  "
                     f"({args.min_freq:g} ≤ f ≤ {args.max_freq:g} GHz)")
    else:
        if args.min_freq == args.max_freq:
            title = f"{event_dir.name}  (f = {args.min_freq:g} GHz)"
        else:
            title = f"{event_dir.name}  ({args.min_freq:g} ≤ f ≤ {args.max_freq:g} GHz)"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if args.all_qmeters:
        pol_time_fig = plot_pol_vs_time(rows, title, None)
        freq_time_fig = plot_freq_vs_time(rows, title, None)
        pol_time_path = plots_dir / f"{event_dir.name}_all_pol_vs_time.png"
        freq_time_path = plots_dir / f"{event_dir.name}_all_freq_vs_time.png"
        pol_time_fig.savefig(pol_time_path, dpi=150)
        freq_time_fig.savefig(freq_time_path, dpi=150)
        logger.info("saved %s", pol_time_path)
        logger.info("saved %s", freq_time_path)
    else:
        freq_fig = plot_pol_vs_freq(rows, title, args.qmeter_name)
        time_fig = plot_pol_vs_time(rows, title, args.qmeter_name)
        freq_path = plots_dir / f"{event_dir.name}_pol_vs_freq.png"
        time_path = plots_dir / f"{event_dir.name}_pol_vs_time.png"
        freq_fig.savefig(freq_path, dpi=150)
        time_fig.savefig(time_path, dpi=150)
        logger.info("saved %s", freq_path)
        logger.info("saved %s", time_path)

    if args.nmr and not args.all_qmeters:
        nmr_pdf = args.output_nmr or plots_dir / f"{event_dir.name}_NMR_Signal.pdf"
        try:
            peak_event_num = int(rows.loc[rows["Polarization"].idxmax(), "EventNum"])
            raw_idx = _raw_signal_index_for(event_dir, peak_event_num)
            if raw_idx is None:
                logger.warning("peak event %d not found in RawSignal CSV — skipping NMR PDF",
                               peak_event_num)
            else:
                logger.info("NMR window: RawSignal rows %d–%d (peak event %d at row %d)",
                            raw_idx - nmr_peak_window, raw_idx + nmr_peak_window,
                            peak_event_num, raw_idx)
                pages = build_nmr_pdf(event_dir, nmr_pdf,
                                      min_index=raw_idx - nmr_peak_window,
                                      max_index=raw_idx + nmr_peak_window)
                if pages == 0:
                    logger.warning("no NMR rows plotted — removing %s", nmr_pdf)
                    nmr_pdf.unlink(missing_ok=True)
                else:
                    logger.info("wrote %d NMR page(s) to %s", pages, nmr_pdf)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping NMR PDF: %s", exc)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
