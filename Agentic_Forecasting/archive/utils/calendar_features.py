"""Deterministic calendar / holiday feature synthesis.

This module injects country-specific holiday and seasonal indicator columns
into the source dataframe BEFORE feature availability detection runs. It
closes the data-engineering gap where upstream ETLs forget to populate
calendar columns for the forecast horizon, leaving flags like
``holiday_christmas_day`` set to 0 for every future week even though
Christmas obviously falls inside those weeks.

Because calendar events are deterministic, we can compute them perfectly
from the ``year_week`` / ``year_month`` column. This gives the classifier
downstream fully populated calendar features in both history and future,
which is exactly what the benchmark forecast has access to.

Design notes
------------

* The module is country-aware: the primary engine is the ``holidays``
  package. Columns emitted by the library are snake-cased automatically
  (``"Christmas Day"`` → ``"holiday_christmas_day"``), which matches the
  existing UK source-data conventions.
* Country-specific supplementary events that are NOT part of the standard
  bank-holiday library (Shrove Tuesday, Black Friday, one-off jubilees,
  etc.) live in ``COUNTRY_CUSTOM_EVENTS``. Each entry is either a list of
  ISO date strings or a callable that receives a year and returns dates.
* All outputs are integer columns with values in {0, 1}. A ``holiday_flag``
  "any holiday this period" column and a ``holiday_total_holidays`` count
  column are always emitted for convenience.
* Pre- and post-holiday lead/lag windows are emitted as
  ``is_pre_holiday_<N>w`` / ``is_post_holiday_<N>w`` flags for configurable
  ``N``.
* A ``weeks_to_next_holiday`` and ``weeks_from_last_holiday`` integer
  distance feature is always produced (clipped at 52 for weekly data,
  12 for monthly).

Usage
-----

>>> from utils.calendar_features import inject_calendar_features
>>> df = inject_calendar_features(
...     df, date_col="year_week", time_format="year_week",
...     country="UK", subdivision="ENG",
...     overwrite_mode="always",
... )

Author: FEU-Agentic-Forecasting Demand Forecasting System
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except Exception as _e:  # pragma: no cover — library may be missing on some nodes
    _holidays_lib = None
    _HOLIDAYS_AVAILABLE = False
    logger.warning("holidays library unavailable (%s) — calendar synthesis will fall back to seasonal Fourier only.", _e)


# =============================================================================
# Country / subdivision mapping
# =============================================================================

# Map the pipeline's country label to an ISO code accepted by the holidays lib.
COUNTRY_CODE_MAP: Dict[str, str] = {
    "UK": "GB", "GB": "GB", "GBR": "GB",
    "DE": "DE", "DEU": "DE", "GERMANY": "DE",
    "TH": "TH", "THA": "TH", "THAILAND": "TH",
    "US": "US", "USA": "US",
    "IN": "IN", "IND": "IN", "INDIA": "IN",
    "FR": "FR", "FRA": "FR",
    "NL": "NL", "NLD": "NL", "NETHERLANDS": "NL",
    "JP": "JP", "JPN": "JP", "JAPAN": "JP",
    "AT": "AT", "CH": "CH",
    "IT": "IT", "ES": "ES", "PT": "PT",
    "IE": "IE",
    "AU": "AU", "NZ": "NZ",
    "BR": "BR", "MX": "MX", "CA": "CA",
    "CN": "CN", "SG": "SG", "MY": "MY",
    "ZA": "ZA",
    # Sub-regions / regional aliases
    "DACH": "DE",  # use DE as primary; users should run per-country for accuracy
}


# =============================================================================
# Custom country-specific events that are NOT in the holidays library
# =============================================================================

def _easter_sunday(year: int) -> date:
    """Compute Western Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _shrove_tuesday(year: int) -> date:
    """Shrove Tuesday / Pancake Day = Easter Sunday − 47 days."""
    return _easter_sunday(year) - timedelta(days=47)


def _last_monday_of(year: int, month: int) -> date:
    """Return the last Monday of ``year``/``month`` (for UK bank holidays)."""
    last_day = date(year, month, 28)
    # Walk forward to the end of the month
    while (last_day + timedelta(days=1)).month == month:
        last_day += timedelta(days=1)
    # Walk back to Monday (weekday 0)
    while last_day.weekday() != 0:
        last_day -= timedelta(days=1)
    return last_day


def _nth_weekday_of(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence (1-indexed) of ``weekday`` in ``month`` of ``year``."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _black_friday(year: int) -> date:
    """Black Friday = day after US Thanksgiving = 4th Thursday of November + 1."""
    thanksgiving = _nth_weekday_of(year, 11, weekday=3, n=4)
    return thanksgiving + timedelta(days=1)


# Mapping: country → { output_column_name: callable(year)->date | list[str] | list[date] }
# Callable receives a year and returns a single date (or list of dates).
COUNTRY_CUSTOM_EVENTS: Dict[str, Dict[str, Any]] = {
    "UK": {
        "pancake_holiday_flag": _shrove_tuesday,
        # UK calls it "Late Summer Bank Holiday" — the CSV column is august_bank_holiday_flag.
        "august_bank_holiday_flag": lambda y: _last_monday_of(y, 8),
        # One-off UK jubilees. Add more as they occur.
        "holiday_queens_platinum_jubilee": ["2022-06-02", "2022-06-03"],
        # Additional UK retail events that business tracks but aren't bank holidays.
        "black_friday_flag": _black_friday,
    },
    "US": {
        "black_friday_flag": _black_friday,
        "super_bowl_flag": lambda y: _nth_weekday_of(y, 2, weekday=6, n=1),  # 1st Sunday of Feb
    },
    "DE": {
        # Germany's Karfreitag / Ostermontag are bank holidays already.
        # Fasching / Rosenmontag is 48 days before Easter.
        "rosenmontag_flag": lambda y: _easter_sunday(y) - timedelta(days=48),
    },
    "IN": {
        # India's major retail events that are non-bank-holiday:
        "diwali_flag": {  # approximations — business may override with custom_events
            2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20",
            2026: "2026-11-08", 2027: "2027-10-29", 2028: "2028-11-17",
        },
    },
    "TH": {
        # Songkran water festival is a multi-day event; bank holidays library covers the days
        # but we can add a festival-week flag that fires the week of April 13.
        "songkran_flag": lambda y: date(y, 4, 13),
    },
    "JP": {
        "golden_week_flag": lambda y: date(y, 5, 3),  # constitution day anchors golden week
    },
}


# =============================================================================
# Period → date helpers
# =============================================================================

_NUMERIC_WEEK_RE = re.compile(r"^\s*(\d{4})[-]?(\d{1,2})\s*$")


def _parse_period(period: Any, time_format: str) -> Tuple[int, int]:
    """Parse a YYYY-WW / YYYYWW / YYYY-MM / YYYYMM period into (year, sub)."""
    s = str(period).strip()
    m = _NUMERIC_WEEK_RE.match(s.replace("_", "-"))
    if m:
        return int(m.group(1)), int(m.group(2))
    # Fallback: try integer like 202509
    try:
        v = int(float(s))
        year = v // 100
        sub = v % 100
        return year, sub
    except Exception:
        raise ValueError(f"Cannot parse period {period!r}")


def _period_date_range(period: Any, time_format: str) -> Tuple[date, date]:
    """Return inclusive ``(start, end)`` dates covered by ``period``.

    For ``year_week`` returns Monday..Sunday of that ISO week.
    For ``year_month`` returns the 1st through last day of the month.
    """
    year, sub = _parse_period(period, time_format)
    if time_format == "year_month":
        start = date(year, sub, 1)
        if sub == 12:
            nxt = date(year + 1, 1, 1)
        else:
            nxt = date(year, sub + 1, 1)
        return start, nxt - timedelta(days=1)

    # year_week (ISO)
    try:
        start = date.fromisocalendar(year, sub, 1)  # Monday
    except ValueError as exc:
        # ISO week 53 in a year that only has 52 → fall back to last day of year.
        logger.debug("Non-existent ISO week (%s, %s): %s", year, sub, exc)
        start = date(year, 12, 28)
    end = start + timedelta(days=6)  # Sunday
    return start, end


# =============================================================================
# Name sanitisation
# =============================================================================

_SNAKE_PUNCT_RE = re.compile(r"[^0-9a-z]+")


def _snake(name: str) -> str:
    """Lower-case + snake-case a holiday name (e.g. 'May Day' → 'may_day')."""
    s = name.lower()
    s = s.replace("&", "and")
    s = _SNAKE_PUNCT_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _holiday_column(name: str) -> str:
    """Map a holiday library name to a snake-cased column name, avoiding double prefixes."""
    snake = _snake(name)
    # Strip the "(observed)" suffix that the holidays library emits — observed
    # holidays already fall in the same week, so collapsing makes the column stable.
    snake = snake.replace("_observed", "")
    # Normalise very common UK variant names to the CSV canonical names.
    canon = {
        "new_year_s_day": "new_years_day",
        "easter_monday": "easter_monday",
        "may_day": "early_may_bank_holiday",
        "spring_bank_holiday": "spring_bank_holiday",
        "late_summer_bank_holiday": "late_summer_bank_holiday",
    }
    snake = canon.get(snake, snake)
    if snake.startswith("holiday_"):
        return snake
    return f"holiday_{snake}"


# =============================================================================
# Core holiday-set computation
# =============================================================================

def _build_holidays_for_years(
    country: str,
    subdivision: Optional[str],
    years: Iterable[int],
    custom_events: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[date, List[str]], Set[str]]:
    """Return ``(date → holiday-column-names, set-of-all-column-names)``.

    Merges the ``holidays`` library output with country-specific custom events
    and any run-level overrides passed in via ``custom_events``.
    """
    country_code = COUNTRY_CODE_MAP.get(country.upper(), country.upper())
    years = list(years)
    date_to_cols: Dict[date, List[str]] = {}
    all_cols: Set[str] = set()

    # --- Library-provided bank holidays -----------------------------------
    if _HOLIDAYS_AVAILABLE:
        try:
            if subdivision:
                lib_holidays = _holidays_lib.country_holidays(
                    country_code, subdiv=subdivision, years=years
                )
            else:
                lib_holidays = _holidays_lib.country_holidays(country_code, years=years)
            for d, n in lib_holidays.items():
                col = _holiday_column(n)
                date_to_cols.setdefault(d, []).append(col)
                all_cols.add(col)
        except Exception as exc:
            logger.warning(
                "holidays library failed for %s (subdiv=%s): %s — continuing with custom events only.",
                country_code, subdivision, exc,
            )

    # --- Country-level custom events --------------------------------------
    country_custom = COUNTRY_CUSTOM_EVENTS.get(country.upper(), {})
    for col, spec in country_custom.items():
        for d in _resolve_event_dates(spec, years):
            date_to_cols.setdefault(d, []).append(col)
            all_cols.add(col)

    # --- Run-level custom events (from config) ----------------------------
    if custom_events:
        for col, spec in custom_events.items():
            for d in _resolve_event_dates(spec, years):
                date_to_cols.setdefault(d, []).append(col)
                all_cols.add(col)

    # --- UK-specific: Easter / Good Friday span flag ----------------------
    # The CSV has holiday_easter_flag — mark the week containing Easter Sunday.
    # We attach it to the Easter Sunday date; the per-period roll-up will lift
    # the flag to the full week.
    if country.upper() in ("UK", "GB"):
        for y in years:
            es = _easter_sunday(y)
            date_to_cols.setdefault(es, []).append("holiday_easter_flag")
            all_cols.add("holiday_easter_flag")
            # UK substitute bank holidays for Xmas/Boxing when they land on weekends
            xmas = date(y, 12, 25)
            if xmas.weekday() >= 5:
                sub = xmas + timedelta(days=(7 - xmas.weekday()))  # move to next Monday
                date_to_cols.setdefault(sub, []).append(
                    "holiday_substitute_bank_holiday_for_christmas_day"
                )
                all_cols.add("holiday_substitute_bank_holiday_for_christmas_day")
            boxing = date(y, 12, 26)
            if boxing.weekday() >= 5:
                sub = boxing + timedelta(days=(7 - boxing.weekday()))
                date_to_cols.setdefault(sub, []).append(
                    "holiday_substitute_bank_holiday_for_boxing_day"
                )
                all_cols.add("holiday_substitute_bank_holiday_for_boxing_day")

    return date_to_cols, all_cols


def _resolve_event_dates(spec: Any, years: List[int]) -> List[date]:
    """Resolve a ``COUNTRY_CUSTOM_EVENTS`` spec into a concrete list of dates.

    Accepts:
      - callable(year) → date or list of dates
      - dict(year → iso-date string)
      - list/tuple of iso-date strings (one-off events)
      - list of ``date`` objects
    """
    out: List[date] = []
    if callable(spec):
        for y in years:
            try:
                r = spec(y)
            except Exception as exc:
                logger.debug("custom event callable failed for year %s: %s", y, exc)
                continue
            if isinstance(r, date):
                out.append(r)
            elif isinstance(r, (list, tuple)):
                for item in r:
                    out.append(item if isinstance(item, date) else date.fromisoformat(str(item)))
        return out

    if isinstance(spec, dict):
        for y in years:
            if y in spec:
                v = spec[y]
                out.append(v if isinstance(v, date) else date.fromisoformat(str(v)))
        return out

    if isinstance(spec, (list, tuple)):
        for item in spec:
            if isinstance(item, date):
                out.append(item)
            else:
                try:
                    out.append(date.fromisoformat(str(item)))
                except Exception:
                    logger.debug("cannot parse custom event date %r", item)
        return out

    logger.warning("Unrecognised custom-event spec: %r", spec)
    return out


# =============================================================================
# Per-period aggregation (date-level holidays → period-level flags)
# =============================================================================

def _compute_period_flags(
    unique_periods: List[Any],
    time_format: str,
    date_to_cols: Dict[date, List[str]],
    all_cols: Set[str],
    lead_lag_windows: List[int],
) -> pd.DataFrame:
    """Build a dataframe with one row per unique period and one column per flag.

    Every column is int8 with values in {0, 1}. Adds:
      * ``holiday_flag`` — any holiday fires in the period
      * ``holiday_total_holidays`` — count of distinct holidays in the period
      * ``weeks_to_next_holiday`` / ``weeks_from_last_holiday`` — integer
        distance to the nearest holiday period (clipped at 52 / 12).
      * ``is_pre_holiday_<N>w`` / ``is_post_holiday_<N>w`` — lead / lag flags.
      * Cyclical calendar features: ``iso_week``, ``month``, ``quarter``,
        ``is_month_end``, ``is_quarter_end``, ``is_year_end``, plus
        Fourier-basis sin/cos pairs (orders 1..3).
    """
    rows: List[Dict[str, Any]] = []
    # Deterministic ordering so tests are stable
    sorted_cols = sorted(all_cols)

    max_dist = 52 if time_format == "year_week" else 12
    period_starts: Dict[Any, date] = {}

    for p in unique_periods:
        start, end = _period_date_range(p, time_format)
        period_starts[p] = start

        flags: Dict[str, int] = {c: 0 for c in sorted_cols}
        hits: List[str] = []
        cursor = start
        while cursor <= end:
            if cursor in date_to_cols:
                for col in date_to_cols[cursor]:
                    if flags.get(col, 0) == 0:
                        flags[col] = 1
                        hits.append(col)
            cursor += timedelta(days=1)

        row: Dict[str, Any] = {"__period__": p, **flags}
        row["holiday_flag"] = int(len(hits) > 0)
        row["holiday_total_holidays"] = int(len(hits))

        # Cyclical calendar (year_week: ISO week / month derived from Monday)
        if time_format == "year_month":
            row["month"] = start.month
            row["quarter"] = (start.month - 1) // 3 + 1
            row["iso_week"] = int(start.isocalendar().week)
        else:
            iw = int(start.isocalendar().week)
            row["iso_week"] = iw
            row["month"] = start.month
            row["quarter"] = (start.month - 1) // 3 + 1
        row["is_month_end"] = int(end.month != (end + timedelta(days=1)).month)
        row["is_quarter_end"] = int(row["is_month_end"] and row["month"] in (3, 6, 9, 12))
        row["is_year_end"] = int(row["is_month_end"] and row["month"] == 12)

        # Fourier orders 1..3 on the seasonal period
        period_of_year = row["iso_week"] if time_format == "year_week" else row["month"]
        period_length = 52.0 if time_format == "year_week" else 12.0
        for k in (1, 2, 3):
            theta = 2 * np.pi * k * period_of_year / period_length
            row[f"fourier_sin_{k}"] = float(np.sin(theta))
            row[f"fourier_cos_{k}"] = float(np.cos(theta))

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Distance features ---
    # Periods ordered chronologically by their start date.
    df_sorted = df.sort_values("__period__", kind="stable").reset_index(drop=True)
    holiday_mask = df_sorted["holiday_flag"].values.astype(bool)
    n = len(df_sorted)
    fwd = np.full(n, max_dist, dtype=np.int32)
    bwd = np.full(n, max_dist, dtype=np.int32)

    last = -max_dist - 1
    for i in range(n):
        if holiday_mask[i]:
            last = i
        bwd[i] = min(max_dist, i - last) if last >= 0 else max_dist

    nxt = n + max_dist + 1
    for i in range(n - 1, -1, -1):
        if holiday_mask[i]:
            nxt = i
        fwd[i] = min(max_dist, nxt - i) if nxt < n + max_dist else max_dist

    df_sorted["weeks_from_last_holiday"] = bwd
    df_sorted["weeks_to_next_holiday"] = fwd

    # --- Lead / lag windows ---
    for N in lead_lag_windows:
        df_sorted[f"is_pre_holiday_{N}w"] = (fwd <= N).astype(np.int8)
        df_sorted[f"is_post_holiday_{N}w"] = (bwd <= N).astype(np.int8)

    # Restore original period order for the join downstream
    df_sorted = df_sorted.set_index("__period__")

    # Ensure integer dtypes for flag columns (object → int8 to save memory)
    for c in sorted_cols:
        df_sorted[c] = df_sorted[c].astype(np.int8)
    return df_sorted


# =============================================================================
# Public entry point
# =============================================================================

def inject_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    time_format: str,
    country: str,
    subdivision: Optional[str] = None,
    overwrite_mode: str = "always",
    history_cutoff: Optional[Any] = None,
    custom_events: Optional[Dict[str, Any]] = None,
    lead_lag_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Synthesize country-specific calendar features into ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe with a date column. Returned unchanged if ``country``
        is empty or the module cannot resolve any events.
    date_col : str
        Period column name (e.g. ``'year_week'`` or ``'year_month'``).
    time_format : str
        Either ``'year_week'`` or ``'year_month'``. Any other value returns
        ``df`` unchanged with a warning (calendar synthesis needs a
        regular period grid).
    country : str
        Country label — see :data:`COUNTRY_CODE_MAP`.
    subdivision : str, optional
        Sub-division code as expected by the ``holidays`` library
        (``'ENG'`` / ``'SCT'`` / ``'WLS'`` / ``'NIR'`` for the UK).
    overwrite_mode : str
        ``'never'`` → only create columns that don't already exist.
        ``'future_only'`` → overwrite only rows where ``date_col >
        history_cutoff``. ``'always'`` (default) → overwrite every row.
    history_cutoff : str, optional
        Period string used for ``overwrite_mode == 'future_only'``. Ignored
        for the other modes.
    custom_events : dict, optional
        Run-level overrides / additions. Same structure as
        :data:`COUNTRY_CUSTOM_EVENTS` entries.
    lead_lag_windows : list[int], optional
        Window sizes for ``is_pre_holiday_<N>`` / ``is_post_holiday_<N>``
        flags. Defaults to ``[1, 2, 4]``.

    Returns
    -------
    pd.DataFrame
        Input dataframe with calendar columns added / updated according to
        ``overwrite_mode``.
    """
    if not country:
        logger.info("calendar_features: country is empty — skipping injection.")
        return df
    if time_format not in ("year_week", "year_month"):
        logger.info(
            "calendar_features: time_format=%r unsupported — skipping injection.",
            time_format,
        )
        return df
    if date_col not in df.columns:
        logger.warning("calendar_features: date_col %r not in df — skipping.", date_col)
        return df
    if overwrite_mode not in ("never", "future_only", "always"):
        raise ValueError(f"overwrite_mode must be one of never|future_only|always, got {overwrite_mode!r}")
    if lead_lag_windows is None:
        lead_lag_windows = [1, 2, 4]

    from utils.period_utils import normalise_period, normalise_period_column

    # Normalise periods in a copy so comparisons downstream are stable.
    work = normalise_period_column(df, date_col) if df is not None else df
    unique_periods = list(dict.fromkeys(work[date_col].astype(str).tolist()))
    if not unique_periods:
        return df

    years: Set[int] = set()
    for p in unique_periods:
        try:
            y, _ = _parse_period(p, time_format)
            years.add(y)
        except Exception:
            logger.debug("calendar_features: cannot parse period %r", p)
    if not years:
        logger.warning("calendar_features: no valid years extracted — skipping.")
        return df

    date_to_cols, all_cols = _build_holidays_for_years(
        country=country,
        subdivision=subdivision,
        years=sorted(years),
        custom_events=custom_events,
    )

    flags = _compute_period_flags(
        unique_periods=unique_periods,
        time_format=time_format,
        date_to_cols=date_to_cols,
        all_cols=all_cols,
        lead_lag_windows=lead_lag_windows,
    )

    # --- Decide which columns to write/overwrite --------------------------
    out_cols = list(flags.columns)
    out = df.copy()

    if overwrite_mode == "future_only" and history_cutoff is not None:
        cutoff_norm = normalise_period(history_cutoff)
        period_norm = out[date_col].astype(str).map(normalise_period)
        future_mask = period_norm > cutoff_norm
    elif overwrite_mode == "future_only":
        logger.warning(
            "calendar_features: overwrite_mode=future_only but history_cutoff "
            "is None — falling back to 'always'."
        )
        overwrite_mode = "always"
        future_mask = pd.Series(True, index=out.index)
    else:
        future_mask = pd.Series(True, index=out.index)

    # Map (period → row index in flags) for fast lookup
    period_series = out[date_col].astype(str)
    normalised_period = period_series.map(lambda x: x)  # already normalised via the column
    flag_index = flags.index.astype(str)
    flags.index = flag_index

    added, updated = 0, 0
    for col in out_cols:
        col_values = flags[col].reindex(normalised_period.values).values
        src_dtype = flags[col].dtype

        if col not in out.columns:
            # Create the column in the correct dtype so partial assignments
            # with 'future_only' don't trigger object/float coercion warnings.
            if overwrite_mode == "never" or overwrite_mode == "always":
                out[col] = col_values
            else:  # future_only
                if np.issubdtype(src_dtype, np.floating):
                    default = np.float64(0.0)
                else:
                    default = np.int8(0)
                out[col] = default
                out[col] = out[col].astype(src_dtype, copy=False)
                mask_arr = future_mask.to_numpy()
                out.loc[future_mask, col] = col_values[mask_arr]
            added += 1
        else:
            if overwrite_mode == "never":
                continue
            if overwrite_mode == "always":
                out[col] = col_values
            else:  # future_only
                # Promote the existing column dtype if needed so we don't
                # clip float Fourier terms into an int64 holder.
                if np.issubdtype(src_dtype, np.floating) and not np.issubdtype(out[col].dtype, np.floating):
                    out[col] = out[col].astype(np.float64, copy=False)
                mask_arr = future_mask.to_numpy()
                out.loc[future_mask, col] = col_values[mask_arr]
            updated += 1

    logger.info(
        "calendar_features: country=%s subdiv=%s time_format=%s years=%s — "
        "added=%d updated=%d (overwrite_mode=%s, periods=%d)",
        country, subdivision, time_format, sorted(years),
        added, updated, overwrite_mode, len(unique_periods),
    )
    return out


__all__ = [
    "inject_calendar_features",
    "COUNTRY_CODE_MAP",
    "COUNTRY_CUSTOM_EVENTS",
]
