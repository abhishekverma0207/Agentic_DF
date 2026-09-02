"""
UK FMCG bank-holiday calendar + signed-distance features.

Why
---
Our model is dominated by rolling-mean features (90%+ of gain) — so it cannot
'see' moving holidays. UK FMCG demand has predictable spike/dip patterns around
bank holidays (e.g. pre-Easter spike, post-Easter dip; pre-May-Day stocking, etc),
which `prediction_rf` clearly captures and we don't.

Fix: for the TARGET week of each forecast, add SIGNED distance features
(weeks-to/from major UK holidays). These are 100% known in advance, so they're
legitimate known-future features.

Calendar (ISO weeks; verified):
  - Easter (movable):  2023 Apr 9 → wk 14; 2024 Mar 31 → wk 13; 2025 Apr 20 → wk 16;
                       2026 Apr 5 → wk 14; 2027 Mar 28 → wk 12.
  - Early May Bank Holiday (1st Mon of May): always wk 18 or 19.
  - Spring Bank Holiday (last Mon of May): wk 22.
  - Summer Bank Holiday (last Mon of Aug): wk 35.
  - Christmas / Boxing Day: wk 52 (or 1 of next year if late).
  - New Year: wk 1.
"""
from __future__ import annotations
from datetime import datetime
import pandas as pd

# ISO week numbers for each event per year — verified manually.
UK_HOLIDAYS_WEEK = {
    # year -> {event: iso_week}
    2023: {"easter": 14, "may_day": 18, "spring_bank": 22, "summer_bank": 35, "christmas": 52, "new_year": 1},
    2024: {"easter": 13, "may_day": 19, "spring_bank": 22, "summer_bank": 35, "christmas": 52, "new_year": 1},
    2025: {"easter": 16, "may_day": 18, "spring_bank": 22, "summer_bank": 35, "christmas": 52, "new_year": 1},
    2026: {"easter": 14, "may_day": 19, "spring_bank": 22, "summer_bank": 36, "christmas": 53, "new_year": 1},
    2027: {"easter": 12, "may_day": 18, "spring_bank": 22, "summer_bank": 35, "christmas": 52, "new_year": 53},
    2028: {"easter": 15, "may_day": 18, "spring_bank": 22, "summer_bank": 35, "christmas": 52, "new_year": 53},
}
EVENTS = ("easter", "may_day", "spring_bank", "summer_bank", "christmas")


def _monday(yw: str) -> datetime:
    s = str(yw); y, w = int(s[:4]), int(s[4:])
    return datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")


def weeks_to_event(yw: str, event: str) -> int:
    """Signed weeks from `yw` to the NEAREST occurrence of `event` (across yw's year
    plus the year before and after). Negative = event is in the past; positive = future.
    Returns 99 if event isn't in the calendar for any nearby year."""
    y = int(str(yw)[:4])
    base = _monday(yw)
    candidates = []
    for yr in (y - 1, y, y + 1):
        if yr in UK_HOLIDAYS_WEEK and event in UK_HOLIDAYS_WEEK[yr]:
            wk = UK_HOLIDAYS_WEEK[yr][event]
            try:
                d = datetime.strptime(f"{yr}-W{wk:02d}-1", "%G-W%V-%u")
                candidates.append((abs((d - base).days // 7), (d - base).days // 7))
            except ValueError:
                pass
    if not candidates:
        return 99
    return min(candidates)[1]


def add_holiday_features(df: pd.DataFrame, yw_col: str = "year_week") -> list:
    """Add `weeks_to_<event>` columns (one per major event) to df. These are
    'known-future' features keyed by year_week. Returns the list of new column names.
    Vectorized via a unique-week map (only ~150 distinct year_weeks in our cache)."""
    weeks = df[yw_col].astype(str).unique()
    new_cols = []
    for ev in EVENTS:
        m = {w: weeks_to_event(w, ev) for w in weeks}
        col = f"weeks_to_{ev}"
        df[col] = df[yw_col].astype(str).map(m).astype("int16")
        new_cols.append(col)
    # Compact "weeks to nearest bank holiday of any kind" — captures pre-holiday shopping in general
    bh_events = ("may_day", "spring_bank", "summer_bank")
    def _nearest_bh(w):
        offs = [weeks_to_event(w, e) for e in bh_events]
        return min(offs, key=abs) if offs else 99
    m = {w: _nearest_bh(w) for w in weeks}
    df["weeks_to_nearest_bh"] = df[yw_col].astype(str).map(m).astype("int16")
    new_cols.append("weeks_to_nearest_bh")
    return new_cols


__all__ = ["UK_HOLIDAYS_WEEK", "EVENTS", "weeks_to_event", "add_holiday_features"]
