"""Unit tests for utils.calendar_features."""

import pandas as pd

from utils.calendar_features import inject_calendar_features


def _mini_df(weeks, extra_cols=None):
    base = {"year_week": weeks,
            "key": ["A"] * len(weeks),
            "actual_sales": [1] * len(weeks)}
    if extra_cols:
        base.update(extra_cols)
    return pd.DataFrame(base)


def test_uk_christmas_week_is_flagged():
    # ISO week 52 of 2025 is 2025-12-22..12-28 -> contains Xmas
    df = _mini_df(["202549", "202552", "202614"])
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="always")
    row = out.loc[out["year_week"] == "202552"].iloc[0]
    assert row["holiday_christmas_day"] == 1
    assert row["holiday_boxing_day"] == 1
    assert row["holiday_flag"] == 1
    assert row["holiday_total_holidays"] >= 2


def test_pancake_day_2026_fires_on_right_iso_week():
    # Shrove Tuesday 2026 = 2026-02-17 → ISO week 8 → 202608
    df = _mini_df(["202606", "202607", "202608", "202609"])
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="always")
    flags = dict(zip(out["year_week"], out["pancake_holiday_flag"]))
    assert flags["202608"] == 1
    assert flags["202607"] == 0
    assert flags["202606"] == 0


def test_de_christmas_week_is_flagged():
    df = _mini_df(["202552"])
    out = inject_calendar_features(df, "year_week", "year_week", "DE",
                                   overwrite_mode="always")
    # DE library names Christmas Day "Erster Weihnachtstag"
    assert out.iloc[0]["holiday_erster_weihnachtstag"] == 1
    assert out.iloc[0]["holiday_flag"] == 1


def test_future_only_preserves_history_for_existing_col():
    # Pre-existing column — history_cutoff at 202614; only future rows overwritten
    df = _mini_df(
        ["202552", "202614", "202652", "202713"],
        extra_cols={"holiday_christmas_day": [0, 0, 0, 0]},
    )
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="future_only",
                                   history_cutoff="202614")
    # Future rows overwritten from 0 → actual flag
    assert out.loc[out["year_week"] == "202652", "holiday_christmas_day"].iloc[0] == 1
    # History kept as-is (still 0 because caller pre-seeded zeros)
    assert out.loc[out["year_week"] == "202552", "holiday_christmas_day"].iloc[0] == 0


def test_always_overwrites_every_row():
    df = _mini_df(
        ["202552", "202614"],
        extra_cols={"holiday_christmas_day": [0, 0]},
    )
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="always")
    # Week 202552 contains Christmas → overwritten to 1
    assert out.loc[out["year_week"] == "202552", "holiday_christmas_day"].iloc[0] == 1


def test_never_preserves_existing_column_but_adds_new_ones():
    df = _mini_df(
        ["202552"],
        extra_cols={"holiday_christmas_day": [0]},
    )
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="never")
    # Existing column preserved (still 0)
    assert out.loc[out["year_week"] == "202552", "holiday_christmas_day"].iloc[0] == 0
    # New column like holiday_flag added
    assert "holiday_flag" in out.columns


def test_empty_country_short_circuits():
    df = _mini_df(["202552"])
    out = inject_calendar_features(df, "year_week", "year_week", "",
                                   overwrite_mode="always")
    assert list(out.columns) == list(df.columns)


def test_unsupported_time_format_short_circuits():
    df = _mini_df(["202552"])
    out = inject_calendar_features(df, "year_week", "date", "UK",
                                   overwrite_mode="always")
    assert list(out.columns) == list(df.columns)


def test_year_month_mode_flags_december():
    # Dec 2026 contains Xmas and Boxing Day
    df = pd.DataFrame({"year_month": ["202611", "202612", "202701"],
                       "key": ["A"] * 3,
                       "actual_sales": [1, 1, 1]})
    out = inject_calendar_features(df, "year_month", "year_month", "UK", "ENG",
                                   overwrite_mode="always")
    row = out.loc[out["year_month"] == "202612"].iloc[0]
    assert row["holiday_christmas_day"] == 1
    assert row["holiday_boxing_day"] == 1
    # January contains New Year's Day
    row_jan = out.loc[out["year_month"] == "202701"].iloc[0]
    assert row_jan["holiday_new_years_day"] == 1


def test_lead_lag_windows():
    # 202551 = week before Xmas → should have is_pre_holiday_1w = 1
    df = _mini_df(["202550", "202551", "202552", "202601", "202602"])
    out = inject_calendar_features(df, "year_week", "year_week", "UK", "ENG",
                                   overwrite_mode="always",
                                   lead_lag_windows=[1, 2])
    flags = out.set_index("year_week")
    # New Year's Day 2026 falls in week 202601. 202551 is 2w before → pre_holiday_2w=1
    # (202551 is also 1w before Xmas which is in 202552)
    assert flags.loc["202551", "is_pre_holiday_1w"] == 1
