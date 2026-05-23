# `--all-qmeters` Mode for `plot_single_event.py` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--all-qmeters` mode to `validation/plot_single_event.py` that plots an entire event's polarization and uWaveFreq vs time across every QMeter, with red vertical lines and rotated labels marking each QMeter transition.

**Architecture:** Extend `load_event_rows` so it can return every QMeter's rows (keeping `QMeterName` as a column). Augment the two time-axis plotting functions to draw a transition overlay when that column is present. Add a `--all-qmeters` CLI flag and a branch in `main()` that skips the QMeter required-arg check, the pol-vs-freq plot, and the NMR pipeline.

**Tech Stack:** Python 3, pandas, matplotlib. No new dependencies. Tests are not added — `plot_single_event.py` has no test suite and verification is manual, per the spec.

**Spec:** `docs/superpowers/specs/2026-05-23-plot-event-all-qmeters-design.md`

---

## File Structure

Only one file is modified:

- **Modify:** `validation/plot_single_event.py`
  - `load_event_rows` — gain optional `qmeter_name`.
  - `plot_pol_vs_time`, `plot_freq_vs_time` — overlay QMeter transitions when the input frame includes a `QMeterName` column; swap the footer text in that case.
  - Add private helper `_overlay_qmeter_transitions(ax, rows)`.
  - `parse_args` — make `qmeter_name` optional in all-qmeters mode; add `--all-qmeters` flag.
  - `main` — branch on `args.all_qmeters` to validate mutually exclusive flags, call the loader with `qmeter_name=None`, skip the freq plot and NMR, and use `_all_` in output filenames.

No other files are touched.

---

## Task 1: Extend `load_event_rows` to optionally skip the QMeter filter

**Files:**
- Modify: `validation/plot_single_event.py:48-97`

- [ ] **Step 1: Change the function signature so `qmeter_name` is optional**

Replace the signature at line 48:

```python
def load_event_rows(event_dir: Path, qmeter_name: str | None = None,
                    min_freq: float = DEFAULT_MIN_FREQ,
                    max_freq: float = DEFAULT_MAX_FREQ,
                    min_index: int | None = None,
                    max_index: int | None = None,
                    min_time_unix: int | None = None,
                    max_time_unix: int | None = None) -> pd.DataFrame:
```

Update the docstring's first paragraph to read:

```
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
```

- [ ] **Step 2: Swap the filter + column selection to branch on `qmeter_name`**

Replace the block currently at lines 78-83 (the `cols = [...]` and `matched = df.loc[...]` lines through the three `pd.to_numeric(...)` calls) with:

```python
    if qmeter_name is None:
        cols = ["uWaveFreq", "Polarization", "EventNum", "csv_line", "QMeterName"]
        matched = df.loc[:, cols].copy()
    else:
        cols = ["uWaveFreq", "Polarization", "EventNum", "csv_line"]
        matched = df.loc[df["QMeterName"] == qmeter_name, cols].copy()
    matched["uWaveFreq"] = pd.to_numeric(matched["uWaveFreq"], errors="coerce")
    matched["Polarization"] = pd.to_numeric(matched["Polarization"], errors="coerce")
    matched["EventNum"] = pd.to_numeric(matched["EventNum"], errors="coerce")
```

The downstream `.dropna(...)`, frequency window filter, time-window filter, timestamp conversion, `.reset_index(drop=True)`, and index-window filter all remain unchanged.

- [ ] **Step 3: Sanity check the change compiles**

Run from the repo root:

```bash
python -c "import importlib.util, sys; \
spec = importlib.util.spec_from_file_location('p', 'validation/plot_single_event.py'); \
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
print('ok')"
```

Expected: prints `ok`. (Just verifies the module imports cleanly.)

- [ ] **Step 4: Verify single-QMeter callers still work end-to-end**

Run an existing single-QMeter invocation against a real event directory:

```bash
cd validation && python plot_single_event.py "Top Proton" 2002-06-21_13h34m30s
```

Expected: writes `validation/plots/2002-06-21_13h34m30s_pol_vs_freq.png` and `..._pol_vs_time.png` and opens the matplotlib window (close it). No tracebacks. If the QMeter has no matching rows for that event, try another known-good QMeter — the goal is just to confirm the existing path still runs.

- [ ] **Step 5: Commit**

```bash
git add validation/plot_single_event.py
git commit -m "Allow load_event_rows to return rows across all QMeters"
```

---

## Task 2: Add a QMeter transition overlay to the two time-axis plots

**Files:**
- Modify: `validation/plot_single_event.py` (functions `plot_pol_vs_time` lines 252-272 and `plot_freq_vs_time` lines 275-296; add a private helper above them)

- [ ] **Step 1: Add the transition overlay helper**

Insert this function immediately above `plot_pol_vs_time` (i.e. between `_format_time_axis` and `plot_pol_vs_time`, around line 250):

```python
def _overlay_qmeter_transitions(ax: plt.Axes, rows: pd.DataFrame) -> None:
    """Draw a red vertical line + rotated label at each QMeter change.

    Requires *rows* to contain a 'QMeterName' column and a 'Timestamp'
    column, sorted in CSV order. The first row is treated as a transition
    so every segment gets a label.
    """
    if "QMeterName" not in rows.columns or rows.empty:
        return
    names = rows["QMeterName"]
    is_change = names != names.shift()
    transitions = rows.loc[is_change, ["Timestamp", "QMeterName"]]
    for ts, name in zip(transitions["Timestamp"], transitions["QMeterName"]):
        ax.axvline(ts, color="red", linewidth=0.8, alpha=0.7)
        ax.text(ts, 0.98, str(name),
                transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right",
                fontsize=8, color="red")
```

- [ ] **Step 2: Update `plot_pol_vs_time` to use the overlay and swap the footer when multi-QMeter**

Replace the body of `plot_pol_vs_time` (the function starting around line 252) with:

```python
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
        distinct = rows["QMeterName"].nunique()
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
```

Note: signature now accepts `qmeter_name: str | None`. Existing callers always pass a string, so behavior is unchanged on those paths.

- [ ] **Step 3: Update `plot_freq_vs_time` the same way**

Replace the body of `plot_freq_vs_time` (around line 275) with:

```python
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
        distinct = rows["QMeterName"].nunique()
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
```

- [ ] **Step 4: Verify the single-QMeter path still works (footer should still say `QMeter: ...`)**

```bash
cd validation && python plot_single_event.py "Top Proton" 2002-06-21_13h34m30s
```

Expected: PNGs written under `validation/plots/`, no tracebacks. Open one of the time PNGs and confirm the footer still reads `QMeter: Top Proton  |  Start date: ...  (n=...)`. If "Top Proton" has no rows for this event, swap in any QMeter you know is present.

- [ ] **Step 5: Commit**

```bash
git add validation/plot_single_event.py
git commit -m "Overlay QMeter transition markers on time-axis plots"
```

---

## Task 3: Wire up the `--all-qmeters` CLI flag and `main()` branch

**Files:**
- Modify: `validation/plot_single_event.py` (functions `parse_args` lines 320-378 and `main` lines 381-653)

- [ ] **Step 1: Add the `--all-qmeters` flag and update help on the positional**

In `parse_args`, change the help on the `qmeter_name` positional to mention the new flag, and add the flag itself. Find the existing `qmeter_name` argument definition (around line 322):

```python
    parser.add_argument("qmeter_name", nargs="?", default=None,
                        help='QMeterName to match exactly, e.g. "Top Proton". '
                             'Not required when --from-candidate is used.')
```

Replace with:

```python
    parser.add_argument("qmeter_name", nargs="?", default=None,
                        help='QMeterName to match exactly, e.g. "Top Proton". '
                             'Not required when --from-candidate or '
                             '--all-qmeters is used.')
```

Then add this new argument right after the existing `--nmr` flag (around line 374-375), before the `-v/--verbose` argument:

```python
    parser.add_argument("--all-qmeters", action="store_true",
                        help="plot every QMeter in the event (no QMeter filter). "
                             "Draws red vertical lines + rotated labels at each "
                             "QMeter change. Emits only the time-axis PNGs "
                             "(no pol_vs_freq, no NMR). Incompatible with a "
                             "positional qmeter_name, --from-candidate, and --nmr.")
```

- [ ] **Step 2: Add mutual-exclusion validation at the top of `main()`**

Insert this block at the very top of `main()`, immediately after the `logging.basicConfig(...)` call (around line 387):

```python
    if args.all_qmeters:
        if args.qmeter_name is not None:
            logger.error("--all-qmeters cannot be combined with a positional qmeter_name")
            return 1
        if args.from_candidate is not None:
            logger.error("--all-qmeters cannot be combined with --from-candidate")
            return 1
        if args.nmr:
            logger.error("--all-qmeters cannot be combined with --nmr")
            return 1
```

- [ ] **Step 3: Skip the existing required-arg check when `--all-qmeters` is set**

The current `elif args.qmeter_name is None or args.event_name is None:` check (around line 546) needs to allow `qmeter_name=None` in all-qmeters mode. Replace that elif block:

```python
    elif args.qmeter_name is None or args.event_name is None:
        logger.error(
            "qmeter_name and event_name are required unless --from-candidate is used."
        )
        return 1
```

with:

```python
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
```

- [ ] **Step 4: Branch the plotting block to skip pol-vs-freq + NMR in all-qmeters mode**

Find the block that builds the title and saves the two PNGs (around lines 612-627):

```python
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
```

Replace with:

```python
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
```

- [ ] **Step 5: Skip the NMR block in all-qmeters mode**

Find the `if args.nmr:` block (around line 629). Replace it with:

```python
    if args.nmr and not args.all_qmeters:
```

The body is unchanged. (The earlier mutual-exclusion check already returns 1 if both flags are set, so this guard is defense-in-depth and keeps the intent obvious.)

- [ ] **Step 6: Call `load_event_rows` with the right `qmeter_name`**

Find the loader call in `main` (around line 585):

```python
    try:
        rows = load_event_rows(event_dir, args.qmeter_name,
                               min_freq=args.min_freq, max_freq=args.max_freq,
                               min_index=args.min_index, max_index=args.max_index,
                               min_time_unix=min_time_unix,
                               max_time_unix=max_time_unix)
```

Replace with:

```python
    loader_qmeter = None if args.all_qmeters else args.qmeter_name
    try:
        rows = load_event_rows(event_dir, loader_qmeter,
                               min_freq=args.min_freq, max_freq=args.max_freq,
                               min_index=args.min_index, max_index=args.max_index,
                               min_time_unix=min_time_unix,
                               max_time_unix=max_time_unix)
```

Also update the "no matching rows" warning (the `logger.warning` call around line 598) so it doesn't print a stale `QMeter=...` value in all-qmeters mode. Replace:

```python
    if rows.empty:
        logger.warning("no matching rows for QMeter=%r in %s with %g <= uWaveFreq <= %g, "
                       "index [%s, %s], time [%s, %s]",
                       args.qmeter_name, event_dir.name,
                       args.min_freq, args.max_freq,
                       args.min_index, args.max_index,
                       time_lo_disp, time_hi_disp)
        return 2
```

with:

```python
    if rows.empty:
        qmeter_disp = "<all>" if args.all_qmeters else repr(args.qmeter_name)
        logger.warning("no matching rows for QMeter=%s in %s with %g <= uWaveFreq <= %g, "
                       "index [%s, %s], time [%s, %s]",
                       qmeter_disp, event_dir.name,
                       args.min_freq, args.max_freq,
                       args.min_index, args.max_index,
                       time_lo_disp, time_hi_disp)
        return 2
```

- [ ] **Step 7: Run the help text to confirm the flag is registered**

```bash
cd validation && python plot_single_event.py --help
```

Expected: `--all-qmeters` appears in the options list with the help text from Step 1.

- [ ] **Step 8: Commit**

```bash
git add validation/plot_single_event.py
git commit -m "Add --all-qmeters CLI mode to plot_single_event"
```

---

## Task 4: Manual verification on a real event

**Files:** none modified unless a defect surfaces.

- [ ] **Step 1: Pick an event with at least two QMeters**

List candidate event directories and pick one. The CSVs include a `QMeterName` column; pick any event with multiple distinct values:

```bash
cd "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/data/events" && \
  for d in 2002-06-21_13h34m30s 2004-04-10_08h25m37s 2004-04-16_10h35m39s; do
    if [ -f "$d/$d.csv" ]; then
      echo "=== $d ==="
      awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="QMeterName") c=i; next} c{print $c}' \
        "$d/$d.csv" | sort -u
    fi
  done
```

Expected: at least one event prints two or more distinct QMeter names. Note which event you'll use below — examples below assume `2004-04-16_10h35m39s`.

- [ ] **Step 2: Run the new mode**

```bash
cd "/Users/jay/Desktop/Papers_For_PHD/Microwave paper/code/Ramp-Up-Finder/validation" && \
  python plot_single_event.py --all-qmeters 2004-04-16_10h35m39s
```

Expected log lines: `saved ...plots/2004-04-16_10h35m39s_all_pol_vs_time.png` and `..._all_freq_vs_time.png`. A matplotlib window opens — close it.

- [ ] **Step 3: Visually verify the two PNGs**

Open both files (e.g. `open validation/plots/2004-04-16_10h35m39s_all_pol_vs_time.png` from the repo root). Check:
- Title includes `[all QMeters]` and the frequency window.
- Footer reads `QMeters: <n>  |  Start date: ...  (n=...)`.
- Red vertical lines appear at every QMeter change, including the first row.
- Each red line has a rotated label with the QMeter name, anchored near the top of the plot.
- No matplotlib warnings about the overlay's transform or text placement.

- [ ] **Step 4: Verify the misuse paths fail cleanly**

```bash
cd validation && \
  python plot_single_event.py --all-qmeters "Top Proton" 2004-04-16_10h35m39s; \
  echo "exit=$?"; \
  python plot_single_event.py --all-qmeters --nmr 2004-04-16_10h35m39s; \
  echo "exit=$?"
```

Expected: both invocations print an `ERROR` and exit with code 1. No PNGs are written from these runs.

- [ ] **Step 5: Confirm single-QMeter mode still works**

```bash
cd validation && \
  python plot_single_event.py "Top Proton" 2004-04-16_10h35m39s
```

Expected: the pre-existing `..._pol_vs_freq.png` and `..._pol_vs_time.png` are written; the time PNG's footer still reads `QMeter: Top Proton  |  ...` (single-QMeter footer); no red overlay appears.

- [ ] **Step 6: If anything failed in Steps 3-5, fix inline and commit the fix**

Diagnose with `git diff` against the last commit. If the fix is small (e.g. a label position tweak), make it and commit:

```bash
git add validation/plot_single_event.py
git commit -m "Fix <specific issue> in --all-qmeters output"
```

If no fixes are needed, no commit is required for this task.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Optional `qmeter_name` + `--all-qmeters` flag | Task 3 Step 1 |
| Reject `qmeter_name` + `--all-qmeters` together | Task 3 Step 2 |
| Reject `--from-candidate` + `--all-qmeters` | Task 3 Step 2 |
| Reject `--nmr` + `--all-qmeters` | Task 3 Step 2 (and Step 5 defense) |
| `--min/max-freq`, `--min/max-time`, `--min/max-index` still apply | Task 1 (loader untouched on those filters) |
| `load_event_rows(qmeter_name=None)` includes `QMeterName` column | Task 1 Step 2 |
| Single-QMeter path unchanged | Task 1 Step 2 (else branch), Task 2 Steps 2-3 (footer fallback), Task 4 Step 5 |
| Two PNGs only: `_all_pol_vs_time`, `_all_freq_vs_time` | Task 3 Step 4 |
| Red `axvline` per transition + rotated label at top | Task 2 Step 1 |
| First row counts as a transition | Task 2 Step 1 (`names != names.shift()` makes the first row True) |
| Footer becomes `QMeters: <n distinct>` | Task 2 Steps 2-3 |
| Title gains `[all QMeters]` | Task 3 Step 4 |
| Errors use `logger.error` + exit 1; empty frame logs warning + exit 2 | Task 3 Steps 2, 3, 6 |

**Placeholder scan:** no TBD / TODO / "add appropriate X" / vague "similar to" steps. Every code-changing step includes the actual code.

**Type/name consistency:**
- `_overlay_qmeter_transitions(ax, rows)` — defined Task 2 Step 1, called by exact same name in Steps 2 and 3.
- `load_event_rows` — `qmeter_name: str | None = None` in Task 1 Step 1; called positionally with `loader_qmeter` in Task 3 Step 6.
- `args.all_qmeters` — argparse converts `--all-qmeters` to that attribute (hyphens → underscores), referenced consistently in `main`.
- Output filenames use `_all_pol_vs_time.png` and `_all_freq_vs_time.png` in Task 3 Step 4 and Task 4 Steps 2-3.
