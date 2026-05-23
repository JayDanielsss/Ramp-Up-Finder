This is a collection of scripts to process B28 polarized target data at UVA.

The directory structure is:

```
detector.py
features.py
nmr_gaussian.py
scan_events.py
extract_event_slice.py
candidates.csv
rampUps/
tests/
    test_detector.py
    test_features.py
    test_nmr_gaussian.py
validation/
    plots/
    plot_events.py
    plot_single_event.py
    time_utils.py
```

---

## Ramp-Up Discovery Pipeline

### Phase 1 — Rule-Based Detection

**Step 1: Scan all events for ramp-up candidates**
```
python scan_events.py --qmeter "Top Proton"
```
Writes new candidates to `candidates.csv` (duplicates skipped automatically).

**Step 2: Review candidates**
```
# polarization plots only (fast)
python validation/plot_single_event.py --from-candidate candidates.csv 2
python validation/plot_single_event.py --from-candidate candidates.csv 3
# ... repeat for each row (header = line 1, first data row = line 2)

# include NMR raw-signal PDF (±5 rows around peak polarization)
python validation/plot_single_event.py --from-candidate candidates.csv 2 --nmr
```

**Step 3: Mark approved candidates**

Open `candidates.csv` and set the `accepted` column to `Y` for candidates that look like valid ramp-ups. Default is `N`.

**Step 4: Extract approved slices**
```
python scan_events.py --extract-from-csv candidates.csv
```
Writes only `Y`-accepted rows to `rampUps/` (main CSV + RawSignal CSV per candidate).

Or combine scan + extract in one shot:
```
python scan_events.py --qmeter "Top Proton" --auto-extract
```

---

## scan_events.py

Batch-scans all event directories for ramp-up candidates using the prominence-based segmenter. Appends new results to `candidates.csv`, skipping duplicates by `(event_name, start_line, end_line)`. Supports extracting approved slices to `rampUps/` without re-scanning.

**candidates.csv schema:**
```
event_name, qmeter_name, start_line, end_line, direction,
start_polarization, end_polarization, swing,
max_polarization, monotonicity_fraction, accepted
```
`start_line`/`end_line` are 1-based line numbers in the source event CSV (header = line 1).

`accepted` defaults to `N`; set to `Y` to approve a candidate for extraction.

**Usage:**
```
python scan_events.py [options]

options:
  --qmeter NAME            restrict scanning to a single QMeter name
  --events-dir DIR         root directory containing event subdirs
                           (default: data/events/)
  --output-dir DIR         directory to write candidates.csv
                           (default: next to this script)
  --auto-extract           extract slices to rampUps/ for each new candidate found
  --extract-from-csv PATH  skip scanning — extract Y-accepted rows from this
                           candidates.csv into rampUps/
  --rampups-dir DIR        directory to write extracted slices (default: rampUps/)
  --dry-run                report what would happen without writing any files
  -v, --verbose            enable per-event debug logging
```

**Examples:**
```
python scan_events.py --qmeter "Top Proton"
python scan_events.py --qmeter "Top Proton" --auto-extract
python scan_events.py --extract-from-csv candidates.csv
python scan_events.py --dry-run

# Preview candidates for a single event without writing any files:
python scan_events.py --dry-run \
    --events-dir /path/to/data/events/2004-08-03_10h29m47s/..
```

---

## detector.py

Core ramp-up detection module. Implements a two-pass prominence-based segmenter as a pure function with no file I/O.

**Algorithm:**
1. **Pass 1 — find boundaries.** `scipy.signal.find_peaks` locates prominent maxima and minima; series endpoints are always included.
2. **Pass 2 — segment and filter.** One candidate per adjacent boundary pair, kept if it meets `min_ramp_rows`, `|swing| ≥ min_swing`, and `monotonicity_fraction` thresholds.

**Interface:**
```python
from detector import detect_ramp_ups, Candidate

candidates = detect_ramp_ups(polarization_series, config)
# returns list[Candidate], each with:
#   start_index, end_index, direction (+1/-1),
#   start_polarization, end_polarization, swing,
#   max_polarization, monotonicity_fraction
```

**Config dict** (per QMeter, from `QMETER_CONFIG` in `scan_events.py`):
```python
{
    "prominence": 5,               # min peak/trough prominence to use as a boundary
    "min_swing": 10,               # |end_pol - start_pol| must exceed this
    "min_ramp_rows": 5,            # minimum segment length in rows
    "monotonicity_fraction": 0.25, # fraction of steps in the ramp direction
}
```

Importable from notebooks. Tests: `pytest tests/test_detector.py`

---

## features.py

Feature extraction module for Phase 2 ML. Computes a fixed-length numeric vector for any polarization window.

**Interface:**
```python
from features import extract_features

f = extract_features(polarization_series, raw_signal_df=None)
# returns dict with keys:
#   start_pol, end_pol, max_pol, net_slope,
#   monotonicity_fraction, gradient_std, nmr_gaussian_r2
# nmr_gaussian_r2 is NaN if raw_signal_df is not provided
```

Importable from notebooks. Tests: `pytest tests/test_features.py`

---

## nmr_gaussian.py

Fits a 1-D Gaussian to a mean NMR voltage sweep and returns the R² goodness-of-fit. Used as a feature in `features.py` and as a standalone validation tool.

**Interface:**
```python
from nmr_gaussian import fit_gaussian_r2

r2 = fit_gaussian_r2(signal_array)  # float; nan if fit fails
```

R² > 0.95 indicates a clean Gaussian shape (expected for valid proton ramp-ups).
R² < 0.5 indicates noise or a calibration artifact.

Tests: `pytest tests/test_nmr_gaussian.py`

---

## Running tests

```
pytest tests/
```

20 tests covering the detector, feature extractor, and Gaussian scorer.

---

## validation/plot_events.py

Scans all event directories under `data/events/` and writes two multi-page PDFs summarising polarization for a given QMeterName across every event.

**Output:**
- `<qmeter>_pol_vs_freq.pdf` — Polarization vs uWaveFreq, one page per event
- `<qmeter>_pol_vs_time.pdf` — Polarization vs Eastern timestamp (from `EventNum`), one page per event
- `<qmeter>_freq_vs_time.pdf` — uWaveFreq vs Eastern timestamp (from `EventNum`), one page per event

**Usage:**
```
python validation/plot_events.py <QMeterName> [options]

positional:
  QMeterName          exact QMeterName to match, e.g. "Top Proton"

options:
  --events-dir DIR    root directory containing event subdirs
  --output-freq PATH  output PDF for polarization vs uWaveFreq
  --output-time PATH  output PDF for polarization vs time index
  --min-freq FLOAT    keep rows with uWaveFreq > this value in GHz (default: 138.0)
  -v, --verbose       enable debug logging
```

**Example:**
```
python validation/plot_events.py "Top Proton" --min-freq 140.0
```

---

## validation/plot_single_event.py

Displays polarization plots for a single event and optionally writes a multi-page PDF of NMR raw-signal traces.

**Output (saved to `validation/plots/`):**
- `<event>_pol_vs_freq.png` — Polarization vs uWaveFreq
- `<event>_pol_vs_time.png` — Polarization vs Eastern timestamp (from `EventNum`)
- `<event>_NMR_Signal.pdf` — one NMR raw-signal plot per row; only written when `--nmr` is passed

**Usage:**
```
python validation/plot_single_event.py <QMeterName> <event_name> [options]

positional:
  QMeterName               exact QMeterName to match, e.g. "Top Proton"
                           (not required when --from-candidate is used)
  event_name               event directory name, e.g. "2004-04-10_08h25m37s"
                           (not required when --from-candidate is used)

options:
  --from-candidate PATH [N]  load event_name, qmeter_name, --min-index, and
                             --max-index from line N of a candidates.csv
                             (header = line 1, first data row = line 2).
                             Omit N to batch-plot every row.
  --events-dir DIR         root directory containing event subdirs
  --min-freq FLOAT         keep rows with uWaveFreq >= this value in GHz (default: 138.0)
  --max-freq FLOAT         keep rows with uWaveFreq <= this value in GHz
  --min-index INT          lower bound on 0-based row index
  --max-index INT          upper bound on 0-based row index (inclusive)
  --min-time STR           lower bound on EventNum as Eastern wall-clock time.
                           Accepted formats:
                             "YYYY-MM-DD HH:MM[:SS]"  e.g. "2004-04-09 13:00"
                             "YYYY-Mon-DD_HHMM"       e.g. "2004-Apr-09_1300"
                           Intersected with --min-index when both are given.
  --max-time STR           upper bound on EventNum (inclusive). Same formats.
                           Intersected with --max-index when both are given.
  --output-nmr PATH        output path for NMR signal PDF
  --nmr                    generate NMR signal PDF (±5 rows around peak polarization)
  -v, --verbose            enable debug logging
```

**Examples:**
```
# polarization plots only for a specific event
python validation/plot_single_event.py "Top Proton" 2004-04-10_08h25m37s

# narrow by Eastern time using the compact date format; include NMR PDF
python validation/plot_single_event.py "Top Proton" 2004-04-10_08h25m37s \
    --min-time "2004-Apr-09_1300" --max-time "2004-Apr-09_1420" \
    --nmr

# same range with the ISO format
python validation/plot_single_event.py "Top Proton" 2004-04-10_08h25m37s \
    --min-time "2004-04-09 13:00" --max-time "2004-04-09 14:20" \
    --nmr

# narrow by row index directly
python validation/plot_single_event.py "Top Proton" 2004-08-03_10h29m47s \
    --min-index 1028 --max-index 1212

# review a single candidate from candidates.csv (line 2 = first data row)
python validation/plot_single_event.py --from-candidate candidates.csv 2

# review with NMR PDF
python validation/plot_single_event.py --from-candidate candidates.csv 2 --nmr

# batch-plot all candidates and write four PDFs to plots/
python validation/plot_single_event.py --from-candidate candidates.csv --nmr
```

---

## validation/time_utils.py

Helpers for converting between Unix timestamps and Eastern Time (`America/New_York`,
DST-aware). Used by the plot scripts to derive a timestamp axis from the `EventNum`
column and to parse `--min-time`/`--max-time` arguments.

**Interface:**
```python
from time_utils import unix_to_eastern, format_eastern, parse_eastern

unix_to_eastern(1047058035)            # -> tz-aware datetime
format_eastern(1047058035)             # -> "2003-03-07 12:27:15 EST"
parse_eastern("2003-03-07 12:27:15")   # -> 1047058035  (ISO format)
parse_eastern("2003-Mar-07_1227")      # -> 1047058020  (compact event format)
```

`parse_eastern` accepts any of these formats (all interpreted as `America/New_York`):

| Format | Example |
|---|---|
| `YYYY-MM-DD HH:MM:SS` | `2004-04-09 13:00:00` |
| `YYYY-MM-DDTHH:MM` | `2004-04-09T13:00` |
| `YYYY-MM-DD HH:MM` | `2004-04-09 13:00` |
| `YYYY-Mon-DD HH:MM:SS` | `2004-Apr-09 13:00:00` |
| `YYYY-Mon-DD HH:MM` | `2004-Apr-09 13:00` |
| `YYYY-Mon-DD_HHMM` | `2004-Apr-09_1300` |
| `YYYY-Mon-DD_HHMMSS` | `2004-Apr-09_130045` |

**Standalone CLI** — convert a Unix timestamp to an Eastern Time stamp:
```
python validation/time_utils.py 1047058035
# -> 2003-03-07 12:27:15 EST
```

---

## extract_event_slice.py

Extracts a filtered slice of event data to CSV files without plotting. Applies QMeterName, frequency, and index filters to the main event CSV and its RawSignal CSV, then writes the matching rows to `rampUps/`.

**Output filenames** encode the slice bounds, e.g.:
```
rampUps/2005-03-03_10h41m57s-800-end.csv
rampUps/2005-03-03_10h41m57s-800-end-RawSignal.csv
```

**Usage:**
```
python extract_event_slice.py <QMeterName> <event_name> [options]

positional:
  QMeterName           exact QMeterName to match, e.g. "Top Proton"
  event_name           event directory name, e.g. "2005-03-03_10h41m57s"

options:
  --events-dir DIR     root directory containing event subdirs
  --output-dir DIR     directory to write output CSVs (default: rampUps/)
  --min-freq FLOAT     keep rows with uWaveFreq >= this value in GHz (default: 138.0)
  --max-freq FLOAT     keep rows with uWaveFreq <= this value in GHz
  --min-index INT      first 0-based row index to include
  --max-index INT      last 0-based row index to include (inclusive)
  --no-raw-signal      skip extracting the RawSignal CSV
  -v, --verbose        enable debug logging
```

**Example:**
```
python extract_event_slice.py "Top Proton" 2005-03-03_10h41m57s \
    --min-index 800 --max-index 1200
```
