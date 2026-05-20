#!/usr/bin/env python3
"""Batch-scan all event directories for ramp-up candidates.

Iterates over every subdirectory in events_dir, applies the prominence-based
detector per QMeter, and appends new results to candidates.csv (skipping
duplicates by event_name + index range). Use --auto-extract to write slices
to rampUps/ during a scan, or --extract-from-csv to extract from an
already-generated candidates.csv without re-scanning.

Examples:
    python scan_events.py
    python scan_events.py --qmeter "Top Proton" --auto-extract
    python scan_events.py --extract-from-csv candidates.csv
    python scan_events.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from detector import detect_ramp_ups
from extract_event_slice import (
    DEFAULT_OUTPUT_DIR as DEFAULT_RAMPUPS_DIR,
    build_output_stem,
    load_event_rows,
    load_raw_signal_rows,
)

DEFAULT_EVENTS_DIR = Path(
    "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent
CANDIDATES_FILENAME = "candidates.csv"

QMETER_CONFIG: dict[str, dict] = {
    "Top Proton": {
        "prominence": 5,
        "min_swing": 10,
        "min_ramp_rows": 5,
        "monotonicity_fraction": 0.25,
    },
}

CANDIDATE_FIELDS = [
    "event_name",
    "qmeter_name",
    "start_index",
    "end_index",
    "direction",
    "start_polarization",
    "end_polarization",
    "swing",
    "max_polarization",
    "monotonicity_fraction",
    "accepted",  # N by default; set to Y to approve for extraction
]

logger = logging.getLogger("scan_events")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing_keys(csv_path: Path) -> set[tuple]:
    """Return (event_name, start_index, end_index) tuples already in csv_path."""
    if not csv_path.exists():
        return set()
    keys: set[tuple] = set()
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                keys.add((row["event_name"], int(row["start_index"]), int(row["end_index"])))
            except (KeyError, ValueError):
                pass
    return keys


def _extract_slice(row: dict, events_dir: Path, rampups_dir: Path) -> bool:
    """Extract main CSV + RawSignal for one candidate row. Returns True on success."""
    event_dir = events_dir / row["event_name"]
    stem = build_output_stem(
        row["event_name"],
        138.0, float("inf"),
        int(row["start_index"]), int(row["end_index"]),
    )
    try:
        df = load_event_rows(
            event_dir, row["qmeter_name"],
            min_index=int(row["start_index"]), max_index=int(row["end_index"]),
        )
        df.to_csv(rampups_dir / f"{stem}.csv", index=False)

        try:
            raw = load_raw_signal_rows(
                event_dir,
                min_index=int(row["start_index"]), max_index=int(row["end_index"]),
            )
            raw.to_csv(rampups_dir / f"{stem}-RawSignal.csv", index=False, header=False)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping RawSignal for %s: %s", stem, exc)

        logger.debug("extracted %s", stem)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to extract %s: %s", stem, exc)
        return False


def _qmeters_to_scan(qmeter_filter: str | None) -> dict[str, dict]:
    if qmeter_filter is not None:
        if qmeter_filter not in QMETER_CONFIG:
            logger.warning("QMeter %r not in QMETER_CONFIG — no candidates will be found", qmeter_filter)
            return {}
        return {qmeter_filter: QMETER_CONFIG[qmeter_filter]}
    return QMETER_CONFIG


def _scan_event(event_dir: Path, qmeters: dict[str, dict]) -> list[dict]:
    """Return candidate rows for one event directory. Skips bad CSVs."""
    rows: list[dict] = []
    for qmeter_name, config in qmeters.items():
        try:
            df = load_event_rows(event_dir, qmeter_name)
        except FileNotFoundError:
            logger.debug("%s: CSV not found — skipping", event_dir.name)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: could not load CSV (%s) — skipping", event_dir.name, exc)
            return []

        if df.empty:
            logger.debug("%s [%s]: no rows after filtering", event_dir.name, qmeter_name)
            continue

        candidates = detect_ramp_ups(df["Polarization"], config)
        logger.debug("%s [%s]: %d candidate(s) found", event_dir.name, qmeter_name, len(candidates))
        for c in candidates:
            rows.append({
                "event_name": event_dir.name,
                "qmeter_name": qmeter_name,
                "start_index": c.start_index,
                "end_index": c.end_index,
                "direction": c.direction,
                "start_polarization": c.start_polarization,
                "end_polarization": c.end_polarization,
                "swing": c.swing,
                "max_polarization": c.max_polarization,
                "monotonicity_fraction": c.monotonicity_fraction,
                "accepted": "N",
            })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=DEFAULT_EVENTS_DIR,
        help="root directory containing event subdirs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory to write candidates.csv (default: next to this script)",
    )
    parser.add_argument(
        "--qmeter",
        metavar="NAME",
        default=None,
        help="restrict scanning to a single QMeter name",
    )
    parser.add_argument(
        "--auto-extract",
        action="store_true",
        help="extract slices to rampUps/ for each candidate found during scanning",
    )
    parser.add_argument(
        "--extract-from-csv",
        type=Path,
        metavar="PATH",
        default=None,
        help="skip scanning — extract slices for every row in this candidates.csv",
    )
    parser.add_argument(
        "--rampups-dir",
        type=Path,
        default=DEFAULT_RAMPUPS_DIR,
        help="directory to write extracted slices (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print candidates to stdout without writing any files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable per-event debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # --- Mode: extract from existing CSV, no scanning ---
    if args.extract_from_csv is not None:
        if not args.extract_from_csv.is_file():
            logger.error("candidates CSV not found: %s", args.extract_from_csv)
            return 1
        with args.extract_from_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            logger.info("No rows in %s — nothing to extract.", args.extract_from_csv)
            return 0
        accepted = [r for r in rows if r.get("accepted", "").strip().upper() == "Y"]
        skipped = len(rows) - len(accepted)
        if not accepted:
            logger.info("No accepted rows in %s — mark the 'accepted' column to approve candidates for extraction.", args.extract_from_csv)
            return 0
        if args.dry_run:
            logger.info("dry run — would extract %d accepted slice(s), %d unaccepted skipped", len(accepted), skipped)
            return 0
        args.rampups_dir.mkdir(parents=True, exist_ok=True)
        n_extracted = sum(
            _extract_slice(row, args.events_dir, args.rampups_dir) for row in accepted
        )
        logger.info("%d/%d accepted slice(s) extracted to %s (%d unaccepted skipped)", n_extracted, len(accepted), args.rampups_dir, skipped)
        return 0

    # --- Mode: scan ---
    if not args.events_dir.is_dir():
        logger.error("events directory not found: %s", args.events_dir)
        return 1

    qmeters = _qmeters_to_scan(args.qmeter)
    if not qmeters:
        logger.error("No QMeters to scan. Check --qmeter argument or QMETER_CONFIG.")
        return 1

    out_path = args.output_dir / CANDIDATES_FILENAME
    existing = _existing_keys(out_path)
    if existing:
        logger.info("Skipping %d already-recorded candidate(s) from %s", len(existing), out_path)

    event_dirs = sorted(p for p in args.events_dir.iterdir() if p.is_dir())
    logger.info("Scanning %d event directories …", len(event_dirs))

    all_candidates: list[dict] = []
    events_scanned = 0
    n_dupes = 0

    for event_dir in event_dirs:
        rows = _scan_event(event_dir, qmeters)
        events_scanned += 1
        for row in rows:
            key = (row["event_name"], int(row["start_index"]), int(row["end_index"]))
            if key in existing:
                n_dupes += 1
            else:
                existing.add(key)
                all_candidates.append(row)

    n_new = len(all_candidates)

    if args.dry_run:
        logger.info(
            "%d events scanned, %d new candidate(s), %d duplicate(s) skipped (dry run)",
            events_scanned, n_new, n_dupes,
        )
        writer = csv.DictWriter(sys.stdout, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(all_candidates)
        return 0

    # Append new candidates
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_candidates)

    # Auto-extract
    n_extracted = 0
    if args.auto_extract and all_candidates:
        args.rampups_dir.mkdir(parents=True, exist_ok=True)
        n_extracted = sum(
            _extract_slice(row, args.events_dir, args.rampups_dir) for row in all_candidates
        )

    summary = f"{events_scanned} events scanned, {n_new} new candidate(s)"
    if n_dupes:
        summary += f", {n_dupes} duplicate(s) skipped"
    if args.auto_extract:
        summary += f", {n_extracted} slice(s) extracted"
    logger.info(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
