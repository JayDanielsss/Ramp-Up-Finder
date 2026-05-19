#!/usr/bin/env python3
"""Plot polarization data for a given QMeterName across event directories.

For each subdirectory of ../data/events/ named like 'YYYY-MM-DD_HHhMMmSSs'
(ignoring '*Base', '*RawSignal', '*PolySignal' variants), reads the matching
'<dir>/<dir>.csv' and writes three multi-page PDFs:

  * <qmeter>_pol_vs_freq.pdf  — Polarization vs uWaveFreq (one page per event)
  * <qmeter>_pol_vs_time.pdf  — Polarization vs row index (proxy for time)
  * <qmeter>_freq_vs_time.pdf — uWaveFreq vs row index (proxy for time)

All PDFs use the same row filter: numeric Polarization, numeric uWaveFreq,
and uWaveFreq strictly greater than MIN_FREQ (default 138 GHz).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

EVENT_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s$")
EXCLUDED_SUFFIXES = ("Base", "RawSignal", "PolySignal")
DEFAULT_MIN_FREQ = 138.0

DEFAULT_EVENTS_DIR = "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events"

logger = logging.getLogger("plot_events")


def find_event_dirs(events_dir: Path) -> list[Path]:
    """Return event subdirectories matching the timestamp pattern, sorted."""
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found: {events_dir}")

    dirs = [
        p for p in events_dir.iterdir()
        if p.is_dir()
        and EVENT_DIR_PATTERN.match(p.name)
        and not p.name.endswith(EXCLUDED_SUFFIXES)
    ]
    return sorted(dirs, key=lambda p: p.name)


def load_event_rows(event_dir: Path, qmeter_name: str,
                    min_freq: float = DEFAULT_MIN_FREQ) -> pd.DataFrame:
    """Load rows for qmeter_name with numeric Polarization and uWaveFreq > min_freq.

    Preserves original CSV row order. Returns an empty frame when the CSV is
    missing, unreadable, lacks required columns, or has no matching rows.
    """
    csv_path = event_dir / f"{event_dir.name}.csv"
    if not csv_path.is_file():
        return pd.DataFrame(columns=["uWaveFreq", "Polarization"])

    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        logger.warning("skipping %s: %s", csv_path.name, exc)
        return pd.DataFrame(columns=["uWaveFreq", "Polarization"])

    required = {"QMeterName", "Polarization", "uWaveFreq"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["uWaveFreq", "Polarization"])

    matched = df.loc[df["QMeterName"] == qmeter_name, ["uWaveFreq", "Polarization"]].copy()
    matched["uWaveFreq"] = pd.to_numeric(matched["uWaveFreq"], errors="coerce")
    matched["Polarization"] = pd.to_numeric(matched["Polarization"], errors="coerce")
    matched = matched.dropna(subset=["uWaveFreq", "Polarization"])
    matched = matched[matched["uWaveFreq"] > min_freq]
    return matched.reset_index(drop=True)


def plot_pol_vs_freq(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure | None:
    """Polarization vs uWaveFreq."""
    if rows.empty:
        return None
    pairs = rows.sort_values("uWaveFreq")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(pairs["uWaveFreq"], pairs["Polarization"], marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("uWaveFreq")
    ax.set_ylabel("Polarization")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"QMeter: {qmeter_name}  (n={len(pairs)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def plot_pol_vs_time(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure | None:
    """Polarization vs row index (time proxy)."""
    if rows.empty:
        return None
    series = rows["Polarization"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(series.index, series.values, marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Row index (time)")
    ax.set_ylabel("Polarization")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"QMeter: {qmeter_name}  (n={len(series)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def plot_freq_vs_time(rows: pd.DataFrame, title: str, qmeter_name: str) -> plt.Figure | None:
    """uWaveFreq vs row index (time proxy)."""
    if rows.empty:
        return None
    series = rows["uWaveFreq"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(series.index, series.values, marker="o", linestyle="-",
            markersize=3, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Row index (time)")
    ax.set_ylabel("uWaveFreq")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, f"QMeter: {qmeter_name}  (n={len(series)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    return fig


def build_pdfs(qmeter_name: str, events_dir: Path,
               freq_pdf: Path, time_pdf: Path, freq_time_pdf: Path,
               min_freq: float = DEFAULT_MIN_FREQ) -> tuple[int, int, int]:
    """Write all three PDFs. Returns (freq_pages, time_pages, freq_time_pages)."""
    event_dirs = find_event_dirs(events_dir)
    logger.info("scanning %d event directories under %s (uWaveFreq > %g)",
                len(event_dirs), events_dir, min_freq)

    for p in (freq_pdf, time_pdf, freq_time_pdf):
        p.parent.mkdir(parents=True, exist_ok=True)

    freq_pages = 0
    time_pages = 0
    freq_time_pages = 0

    with (PdfPages(freq_pdf) as freq_out,
          PdfPages(time_pdf) as time_out,
          PdfPages(freq_time_pdf) as freq_time_out):
        for event_dir in event_dirs:
            rows = load_event_rows(event_dir, qmeter_name, min_freq=min_freq)
            if rows.empty:
                continue

            freq_fig = plot_pol_vs_freq(rows, event_dir.name, qmeter_name)
            if freq_fig is not None:
                freq_out.savefig(freq_fig)
                plt.close(freq_fig)
                freq_pages += 1

            time_fig = plot_pol_vs_time(rows, event_dir.name, qmeter_name)
            if time_fig is not None:
                time_out.savefig(time_fig)
                plt.close(time_fig)
                time_pages += 1

            freq_time_fig = plot_freq_vs_time(rows, event_dir.name, qmeter_name)
            if freq_time_fig is not None:
                freq_time_out.savefig(freq_time_fig)
                plt.close(freq_time_fig)
                freq_time_pages += 1

            logger.debug("processed %s (pol_freq=%s, pol_time=%s, freq_time=%s)",
                         event_dir.name,
                         "yes" if freq_fig else "no",
                         "yes" if time_fig else "no",
                         "yes" if freq_time_fig else "no")

        freq_out.infodict()["Title"] = f"Polarization vs uWaveFreq — {qmeter_name}"
        time_out.infodict()["Title"] = f"Polarization vs time — {qmeter_name}"
        freq_time_out.infodict()["Title"] = f"uWaveFreq vs time — {qmeter_name}"

    return freq_pages, time_pages, freq_time_pages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("qmeter_name",
                        help='QMeterName to match exactly, e.g. "Top Proton"')
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR,
                        help="root directory containing event subdirs "
                             "(default: ../data/events relative to this script)")
    parser.add_argument("--output-freq", type=Path, default=None,
                        help="output PDF for polarization vs uWaveFreq "
                             "(default: <qmeter>_pol_vs_freq.pdf)")
    parser.add_argument("--output-time", type=Path, default=None,
                        help="output PDF for polarization vs time index "
                             "(default: <qmeter>_pol_vs_time.pdf)")
    parser.add_argument("--output-freq-time", type=Path, default=None,
                        help="output PDF for uWaveFreq vs time index "
                             "(default: <qmeter>_freq_vs_time.pdf)")
    parser.add_argument("--min-freq", type=float, default=DEFAULT_MIN_FREQ,
                        help=f"keep only rows with uWaveFreq strictly greater "
                             f"than this value (default: {DEFAULT_MIN_FREQ})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    slug = args.qmeter_name.replace(" ", "_")
    plots_dir = Path(__file__).parent / "plots"
    freq_pdf = args.output_freq or plots_dir / f"{slug}_pol_vs_freq.pdf"
    time_pdf = args.output_time or plots_dir / f"{slug}_pol_vs_time.pdf"
    freq_time_pdf = args.output_freq_time or plots_dir / f"{slug}_freq_vs_time.pdf"

    try:
        freq_pages, time_pages, freq_time_pages = build_pdfs(
            args.qmeter_name, args.events_dir, freq_pdf, time_pdf, freq_time_pdf,
            min_freq=args.min_freq,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    for label, path, pages in (("pol_vs_freq", freq_pdf, freq_pages),
                               ("pol_vs_time", time_pdf, time_pages),
                               ("freq_vs_time", freq_time_pdf, freq_time_pages)):
        if pages == 0:
            logger.warning("no matching rows for %s PDF — removing %s", label, path)
            path.unlink(missing_ok=True)
        else:
            logger.info("wrote %d page(s) to %s", pages, path)

    return 0 if (freq_pages or time_pages or freq_time_pages) else 2


if __name__ == "__main__":
    sys.exit(main())
