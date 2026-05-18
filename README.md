This is a collection of scripts to process B28 polarized target data at UVA.

The directory structure is:

```
rampUps/
validation/
    plots/
    plot_events.py
    plot_single_event.py
extract_event_slice.py
```

---

## validation/plot_events.py

Scans all event directories under `data/events/` and writes two multi-page PDFs summarising polarization for a given QMeterName across every event.

**Output:**
- `<qmeter>_pol_vs_freq.pdf` — Polarization vs uWaveFreq, one page per event
- `<qmeter>_pol_vs_time.pdf` — Polarization vs row index (time proxy), one page per event

**Usage:**
```
python validation/plot_events.py <QMeterName> [options]

positional:
  QMeterName          exact QMeterName to match, e.g. "Top Proton"

options:
  --events-dir DIR    root directory containing event subdirs
                      (default: data/events/)
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
- `<event>_pol_vs_time.png` — Polarization vs row index (time proxy)
- `<event>_NMR_Signal.pdf` — one NMR raw-signal plot per row of `<event>-RawSignal.csv` (skipped with `--no-nmr-pdf`)

**Usage:**
```
python validation/plot_single_event.py <QMeterName> <event_name> [options]

positional:
  QMeterName          exact QMeterName to match, e.g. "Top Proton"
  event_name          event directory name, e.g. "2004-04-10_08h25m37s"

options:
  --events-dir DIR    root directory containing event subdirs
                      (default: data/events/)
  --min-freq FLOAT    keep rows with uWaveFreq >= this value in GHz (default: 138.0)
  --max-freq FLOAT    keep rows with uWaveFreq <= this value in GHz (default: no bound)
                      set equal to --min-freq to select a single frequency
  --min-index INT     lower bound on 0-based row index after QMeter+freq filter
  --max-index INT     upper bound on 0-based row index (inclusive)
  --output-nmr PATH   output path for NMR signal PDF
  --no-nmr-pdf        skip generating the NMR signal PDF
  -v, --verbose       enable debug logging
```

**Example:**
```
python validation/plot_single_event.py "Top Proton" 2004-04-10_08h25m37s \
    --min-freq 140.0 --max-freq 140.3
```

```
python plot_single_event.py "Top Proton" 2004-08-03_10h29m47s --min-index 4817 --max-index 5050
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
                       (default: data/events/)
  --output-dir DIR     directory to write output CSVs (default: rampUps/)
  --min-freq FLOAT     keep rows with uWaveFreq >= this value in GHz (default: 138.0)
  --max-freq FLOAT     keep rows with uWaveFreq <= this value in GHz (default: no bound)
  --min-index INT      first 0-based row index to include (after QMeter + freq filter)
  --max-index INT      last 0-based row index to include (inclusive)
  --no-raw-signal      skip extracting the RawSignal CSV
  -v, --verbose        enable debug logging
```

**Example:**
```
python extract_event_slice.py "Top Proton" 2005-03-03_10h41m57s \
    --min-index 800 --max-index 1200
```

