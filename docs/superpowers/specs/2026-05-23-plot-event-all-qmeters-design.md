# Design: `--all-qmeters` mode for `plot_single_event.py`

## Motivation

Today, `plot_single_event.py` requires a `QMeterName` and filters the event
CSV to that one QMeter. When inspecting an event we sometimes want to see
*every* row across *all* QMeters in CSV order, with visual markers showing
where the active QMeter changes. That makes it easy to spot whether a
ramp/transition aligns with a QMeter switch.

## Scope

Add an `--all-qmeters` mode to the existing `validation/plot_single_event.py`.
No new script. No changes to `plot_events.py` or `plot_single_event.py`'s
default (single-QMeter) behavior.

## CLI surface

- `qmeter_name` positional becomes optional.
- New flag `--all-qmeters`.
- When `--all-qmeters` is set:
  - `qmeter_name` must be omitted; passing both is a usage error.
  - `event_name` is still required.
  - Combining with `--from-candidate` is a usage error (candidates are
    QMeter-specific).
  - Combining with `--nmr` is a usage error (peak-window NMR needs a single
    QMeter series).
- `--min-freq` (default 138), `--max-freq`, `--min-time`, `--max-time`,
  `--min-index`, `--max-index` all still apply. Index bounds are 0-based row
  positions *after* the frequency/time filter, just like single-QMeter mode.

## Data loading

Extend `load_event_rows`:

- Signature gains `qmeter_name: str | None = None` (was required `str`).
- When `qmeter_name is None`, skip the `df["QMeterName"] == qmeter_name`
  filter and include `QMeterName` in the returned columns. Row order remains
  CSV order (already true today).
- When `qmeter_name` is a string, behavior is unchanged. `QMeterName` is not
  added to the returned columns in that path, preserving existing callers.

## Plots emitted

Two PNGs to `plots/`, matching the existing naming convention:

- `<event>_all_pol_vs_time.png` — Polarization vs Eastern time
- `<event>_all_freq_vs_time.png` — uWaveFreq vs Eastern time

No `pol_vs_freq` plot in this mode — mixing QMeters on a frequency axis
isn't meaningful.

## QMeter change overlay

Inside `plot_pol_vs_time` and `plot_freq_vs_time`:

- If the rows include a `QMeterName` column, compute transition rows as
  `rows[rows["QMeterName"] != rows["QMeterName"].shift()]`. This naturally
  includes the first row.
- For each transition row, draw `ax.axvline(timestamp, color="red",
  linewidth=0.8, alpha=0.7)`.
- Label each line with the *new* QMeter name using
  `ax.text(timestamp, 0.98, name, transform=ax.get_xaxis_transform(),
  rotation=90, va="top", ha="right", fontsize=8, color="red")`.
- Swap the footer text from `QMeter: <name>  |  ...` to
  `QMeters: <n distinct>  |  ...` so it stays informative.

When the `QMeterName` column is absent (single-QMeter path), the existing
behavior is preserved exactly.

## Function changes

- `load_event_rows`: optional `qmeter_name`; include `QMeterName` column
  when None.
- `plot_pol_vs_time`, `plot_freq_vs_time`: detect `QMeterName` column and
  overlay transitions + labels; swap footer.
- `plot_pol_vs_freq`: unchanged. Only called in single-QMeter mode.
- `main`: branch on `args.all_qmeters` to:
  - Validate mutual exclusions (`qmeter_name`, `--from-candidate`, `--nmr`).
  - Call `load_event_rows(event_dir, None, ...)`.
  - Skip the pol-vs-freq plot and the NMR pipeline.
  - Use `_all_` infix in output filenames.
- `parse_args`: make `qmeter_name` optional in the all-qmeters case; add
  `--all-qmeters` flag and update help text on `qmeter_name`.

## Title and labels

- Plot title: `<event_name>  [all QMeters]  (138 ≤ f ...)`. Drops the
  per-QMeter bracket present in the candidate batch mode.
- Y/X axis labels unchanged from the existing functions.

## Error handling

- All new usage errors print via `logger.error(...)` and return exit code 1,
  matching existing patterns in `main`.
- If the filtered frame is empty, behavior matches single-QMeter empty: log
  a warning and return exit code 2.

## Out of scope

- No changes to `plot_events.py`.
- No new tests (the existing scripts have no tests; this design follows the
  same posture and is verified manually per the project's current practice).
- No color-coded segments per QMeter — only red transition lines.
