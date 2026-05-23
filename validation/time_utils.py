#!/usr/bin/env python3
"""Convert Unix timestamps to Eastern Time (America/New_York, DST-aware).

Used by plot_events.py and plot_single_event.py so EventNum (a Unix
timestamp column in each event CSV) can drive the time axis and the
--min-time/--max-time filters.

Standalone use:
    python time_utils.py 1047058035
    -> 2003-03-07 12:27:15 EST
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
_ISO_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",    "%Y-%m-%dT%H:%M",
    # abbreviated month, space separator: 2004-Apr-09 13:00[:00]
    "%Y-%b-%d %H:%M:%S", "%Y-%b-%d %H:%M",
    # abbreviated month, underscore + compact HHMM: 2004-Apr-09_1300[00]
    "%Y-%b-%d_%H%M%S",   "%Y-%b-%d_%H%M",
)


def unix_to_eastern(unix_seconds: int | float | str) -> datetime:
    """Return a tz-aware datetime in America/New_York for the given Unix seconds."""
    return datetime.fromtimestamp(float(unix_seconds), tz=EASTERN)


def format_eastern(unix_seconds: int | float | str) -> str:
    """Return 'YYYY-MM-DD HH:MM:SS TZ' (TZ is EST or EDT depending on date)."""
    return unix_to_eastern(unix_seconds).strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_eastern(text: str) -> int:
    """Parse 'YYYY-MM-DD[ T]HH:MM[:SS]' as Eastern wall-clock; return Unix seconds.

    DST is resolved by zoneinfo. Ambiguous fall-back hours resolve to the
    earlier (pre-fold) occurrence; nonexistent spring-forward hours are
    shifted forward, matching Python's default fold=0 semantics.
    """
    for fmt in _ISO_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return int(naive.replace(tzinfo=EASTERN).timestamp())
    raise ValueError(
        f"could not parse {text!r} as Eastern time; expected e.g. "
        f"'2003-03-07 12:25:00', '2003-03-07T12:25', or '2003-Mar-07_1225'"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("unix_seconds",
                        help="Unix timestamp (seconds since epoch)")
    args = parser.parse_args(argv)
    try:
        print(format_eastern(args.unix_seconds))
    except (TypeError, ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
