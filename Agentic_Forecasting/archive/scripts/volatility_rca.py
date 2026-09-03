"""Volatility root-cause-analysis on the actuals (smooth_sales).

Question: has the intrinsic volatility of demand changed across 2024-2026 YTD,
and could that explain the forecast-accuracy degradation?

Inputs
------
A PySpark DataFrame with at minimum:
    key          : str   - series identifier (SKU / store-SKU / ...)
    year_week    : int   - ISO year*100 + week, e.g. 202613 = 2026 ISO W13
    smooth_sales : float - actuals (the column the forecast is scored against)

Outputs (all Spark DataFrames)
------------------------------
volatility_by_fy        : per (key, calendar year) volatility profile
volatility_by_ytd       : per (key, year) volatility on the SAME W01..W{ytd} window
                          - makes 2026 YTD comparable to prior YTD-equivalents
portfolio_by_fy         : portfolio (all keys pooled) per year
rolling_cv_per_key      : per (key, year_week) rolling CV at each window size
rolling_cv_portfolio    : per year_week pooled-portfolio rolling CV
yoy_delta               : per key, ΔCV / ΔCV²_nz / ΔADI / Δclass year-over-year
top_keys_recent         : top-N keys by sum(sales) over the last `recent_window_weeks`
volatility_top_keys_fy  : volatility_by_fy filtered to the top-N keys (for show-and-tell)
class_counts_by_year    : count of keys in each Syntetos-Boylan class per year
high_vol_counts_by_year : count of keys above the high-volatility CV threshold per year
volatility_by_month       : per (key, year_month) volatility (calendar-month buckets)
class_counts_by_month     : count of keys per Syntetos-Boylan class per month (TREND)
high_vol_counts_by_month  : count of keys above CV threshold per month (TREND)
vol_top_keys_monthly      : volatility_by_month restricted to the top-N recent keys

Metric glossary
---------------
CV          std / mean over the full series (zeros included).  Sparsity dominates.
CV²_nz      (std/mean)² computed on NON-zero weeks only.  Pure dispersion.
ADI         Average Demand Interval = N_weeks / N_nonzero_weeks.  Sparsity proxy.
sb_class    Syntetos-Boylan class (smooth | intermittent | erratic | lumpy)
            using ADI=1.32 and CV²_nz=0.49 cutoffs.
pct_zero    Fraction of weeks with zero demand.
spike_pct   Fraction of weeks above mean + 3σ.

Usage
-----
    from pyspark.sql import SparkSession
    from notebooks.volatility_rca import run_volatility_rca

    spark = SparkSession.getActiveSession()
    sdf   = spark.read.table("uc.forecast.feature_panel")  # or wherever it lives
    out   = run_volatility_rca(sdf, key_col="key",
                               year_week_col="year_week",
                               sales_col="smooth_sales",
                               max_year_week=202618,    # everything up to 2026-W18
                               ytd_cutoff_week=18,
                               rolling_windows=(4, 8, 13))
    out["rolling_cv_portfolio"].show(truncate=False)
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


# --------------------------------------------------------------------------- #
# constants                                                                   #
# --------------------------------------------------------------------------- #
SB_ADI_CUT      = 1.32
SB_CV2_CUT      = 0.49
DEFAULT_WINDOWS = (4, 8, 13)   # rolling CV window sizes, in weeks
MIN_OBS_FRAC    = 0.60         # need this fraction of window present to emit a CV


def _safe_div(num, den):
    """Division that returns NULL when den is null or <= 0 — safe under ANSI mode."""
    return F.when((den.isNotNull()) & (den > 0), num/den)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _add_calendar_cols(sdf: DataFrame, year_week_col: str) -> DataFrame:
    """Split YYYYWW int into year, week, ISO-Monday date, and calendar year_month."""
    yw = F.col(year_week_col).cast("int")
    sdf = (sdf
           .withColumn("_year", (yw / F.lit(100)).cast("int"))
           .withColumn("_week", (yw % F.lit(100)).cast("int")))
    # ISO Monday: jump to Jan 4 of the ISO year (always in W1), back up to its
    # Monday, then add (week-1)*7 days.
    sdf = sdf.withColumn(
        "_iso_date",
        F.date_add(
            F.date_sub(F.to_date(F.format_string("%d-01-04", F.col("_year"))),
                       F.dayofweek(F.to_date(F.format_string("%d-01-04", F.col("_year"))))-2),
            (F.col("_week")-1)*7,
        ),
    )
    # Calendar month bucket as YYYYMM int (driven by the ISO-Monday date — so
    # ISO weeks straddling a month boundary fall in the month of their Monday).
    sdf = (sdf
           .withColumn("_year_month",
                       F.year("_iso_date")*F.lit(100) + F.month("_iso_date"))
           .withColumn("_month_start", F.trunc("_iso_date", "month")))
    return sdf


def _sb_class_expr(adi_col: str, cv2_nz_col: str):
    """Return a Column expression for the Syntetos-Boylan class label."""
    a, c = F.col(adi_col), F.col(cv2_nz_col)
    return (F.when(a.isNull() | c.isNull(), F.lit("noclass"))
             .when((a <  SB_ADI_CUT) & (c <  SB_CV2_CUT), F.lit("smooth"))
             .when((a >= SB_ADI_CUT) & (c <  SB_CV2_CUT), F.lit("intermittent"))
             .when((a <  SB_ADI_CUT) & (c >= SB_CV2_CUT), F.lit("erratic"))
             .otherwise(F.lit("lumpy")))


def _metric_agg(sales_col: str):
    """Return the list of agg expressions that produce the volatility block."""
    s = F.col(sales_col).cast("double")
    nz = F.when(s > 0, s)
    return [
        F.count(F.lit(1)).alias("weeks"),
        F.sum(s).alias("total"),
        F.mean(s).alias("mean"),
        F.stddev_samp(s).alias("std"),
        F.mean(nz).alias("nz_mean"),
        F.stddev_samp(nz).alias("nz_std"),
        F.sum(F.when(s == 0, 1).otherwise(0)).alias("zero_weeks"),
        F.sum(F.when(s > 0, 1).otherwise(0)).alias("nonzero_weeks"),
        F.max(s).alias("max"),
    ]


def _add_derived(sdf: DataFrame) -> DataFrame:
    """Tack CV, CV²_nz, ADI, pct_zero, sb_class onto an already-aggregated frame."""
    sdf = (sdf
           .withColumn("cv",        F.when(F.col("mean")>0, F.col("std")/F.col("mean")))
           .withColumn("cv2_nz",    F.when(F.col("nz_mean")>0,
                                           F.pow(F.col("nz_std")/F.col("nz_mean"), F.lit(2))))
           .withColumn("adi",       F.when(F.col("nonzero_weeks")>0,
                                           F.col("weeks")/F.col("nonzero_weeks")))
           .withColumn("pct_zero",  F.when(F.col("weeks")>0,
                                           F.col("zero_weeks")/F.col("weeks")))
           .withColumn("sb_class",  _sb_class_expr("adi","cv2_nz")))
    return sdf


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def _rolling_cv_cols(sdf: DataFrame, *, key_cols: Sequence[str], sales_col: str,
                     order_col: str, windows: Iterable[int]) -> DataFrame:
    """Add one `rolling_cv_{w}w` column per window size, partitioned by key_cols."""
    out = sdf
    for win in windows:
        min_obs = max(2, ceil(MIN_OBS_FRAC * win))
        w = (Window.partitionBy(*key_cols)
                   .orderBy(order_col)
                   .rowsBetween(-(win-1), 0))
        out = (out
               .withColumn(f"_n_{win}",  F.count(sales_col).over(w))
               .withColumn(f"_m_{win}",  F.mean(sales_col).over(w))
               .withColumn(f"_s_{win}",  F.stddev_samp(sales_col).over(w))
               .withColumn(f"rolling_cv_{win}w",
                           F.when((F.col(f"_n_{win}") >= min_obs) &
                                  (F.col(f"_m_{win}") > 0),
                                  F.col(f"_s_{win}") / F.col(f"_m_{win}")))
               .drop(f"_n_{win}", f"_m_{win}", f"_s_{win}"))
    return out


@dataclass
class VolatilityRCAResult:
    volatility_by_fy:        DataFrame
    volatility_by_ytd:       DataFrame
    portfolio_by_fy:         DataFrame
    rolling_cv_per_key:      DataFrame   # week-by-week, one row per (key, year_week)
    rolling_cv_portfolio:    DataFrame   # week-by-week, one row per year_week
    yoy_delta:               DataFrame
    top_keys_recent:         DataFrame   # top-N keys by recent sales
    volatility_top_keys_fy:  DataFrame   # volatility_by_fy filtered to top keys
    class_counts_by_year:    DataFrame   # year × sb_class count
    high_vol_counts_by_year: DataFrame   # year × count of keys with CV > threshold
    volatility_by_month:     DataFrame   # per (key, year_month) volatility
    class_counts_by_month:   DataFrame   # year_month × sb_class count (TREND)
    high_vol_counts_by_month:DataFrame   # year_month × high-vol counts (TREND)
    vol_top_keys_monthly:    DataFrame   # volatility_by_month restricted to top keys
    monthly_trailing_snapshot: DataFrame # per (key, year_month) trailing-13w stats
    class_counts_trailing_monthly: DataFrame  # smoother class trend (trailing window)
    high_vol_counts_trailing_monthly: DataFrame  # smoother high-vol trend
    monthly_volatility_summary: DataFrame  # ONE ROW PER MONTH, includes sparse keys
    monthly_actuals_distribution: DataFrame  # distribution-of-actuals + key count by month


def _monthly_trailing_snapshot(base: DataFrame, *, key_col: str, sales_col: str,
                               window_weeks: int) -> DataFrame:
    """For each (key, year_month), compute volatility metrics over the trailing
    `window_weeks` weeks ending at the LAST observed week of that month.

    Returns one row per (key, year_month) with smoothed trend-friendly metrics:
        r_cv, r_cv2_nz, r_adi, r_sb_class.

    Use this for *monthly trend* charts where calendar-month buckets (4-5 obs
    per cell) are too noisy.
    """
    s   = F.col(sales_col)
    nzs = F.when(s > 0, s)
    w   = (Window.partitionBy(key_col)
                 .orderBy("_iso_date")
                 .rowsBetween(-(window_weeks-1), 0))
    min_obs = max(2, ceil(MIN_OBS_FRAC * window_weeks))

    rolled = (base
        .withColumn("_w_count",    F.count(s).over(w))
        .withColumn("_w_mean",     F.mean(s).over(w))
        .withColumn("_w_std",      F.stddev_samp(s).over(w))
        .withColumn("_w_nz_count", F.count(nzs).over(w))
        .withColumn("_w_nz_mean",  F.mean(nzs).over(w))
        .withColumn("_w_nz_std",   F.stddev_samp(nzs).over(w))
        .withColumn("r_cv",
                    F.when((F.col("_w_count") >= min_obs) & (F.col("_w_mean") > 0),
                           F.col("_w_std") / F.col("_w_mean")))
        .withColumn("r_cv2_nz",
                    F.when((F.col("_w_nz_count") >= 2) & (F.col("_w_nz_mean") > 0),
                           F.pow(F.col("_w_nz_std")/F.col("_w_nz_mean"), F.lit(2))))
        .withColumn("r_adi",
                    F.when(F.col("_w_nz_count") > 0,
                           F.col("_w_count") / F.col("_w_nz_count")))
        .withColumn("r_sb_class", _sb_class_expr("r_adi", "r_cv2_nz")))

    pick = Window.partitionBy(key_col, "_year_month").orderBy(F.col("_iso_date").desc())
    snap = (rolled.withColumn("_rk", F.row_number().over(pick))
                  .filter("_rk = 1"))
    return snap.select(
        key_col,
        F.col("_year_month").alias("year_month"),
        F.col("_month_start").alias("month_start"),
        F.col("_iso_date").alias("as_of_week_date"),
        F.col("_w_count").alias("trailing_weeks"),
        F.col("r_cv").alias("trailing_cv"),
        F.col("r_cv2_nz").alias("trailing_cv2_nz"),
        F.col("r_adi").alias("trailing_adi"),
        F.col("r_sb_class").alias("trailing_sb_class"))


def _top_keys_by_recent_sales(base: DataFrame, *, key_col: str, sales_col: str,
                              year_week_col: str, last_n_weeks: int,
                              top_n: int) -> DataFrame:
    """Return top-N keys by sum(sales) over the most recent `last_n_weeks` year_weeks
    that actually contain any non-zero data (so trailing-zero feed gaps don't dominate)."""
    # Use the last N distinct year_weeks present in the panel (skipping all-zero tail).
    nonzero_weeks = (base.filter(F.col(sales_col) > 0)
                          .select(year_week_col).distinct()
                          .orderBy(F.col(year_week_col).desc())
                          .limit(last_n_weeks))
    recent = base.join(nonzero_weeks, year_week_col, "inner")
    ranked = (recent.groupBy(key_col)
                    .agg(F.sum(sales_col).alias("recent_sales"),
                         F.min(year_week_col).alias("recent_window_start_yw"),
                         F.max(year_week_col).alias("recent_window_end_yw"))
                    .orderBy(F.col("recent_sales").desc())
                    .limit(top_n)
                    .withColumn("rank", F.row_number().over(
                        Window.orderBy(F.col("recent_sales").desc()))))
    return ranked


def run_volatility_rca(
    sdf: DataFrame,
    *,
    key_col: str = "key",
    year_week_col: str = "year_week",
    sales_col: str = "smooth_sales",
    ytd_cutoff_week: int = 21,
    max_actual_year: int = 2026,
    min_actual_year: int = 2023,
    max_year_week: int | None = None,
    rolling_windows: Sequence[int] = DEFAULT_WINDOWS,
    top_n_recent_keys: int = 20,
    recent_window_weeks: int = 13,
    high_vol_cv_threshold: float = 1.0,
    min_weeks_per_month: int = 3,
) -> Dict[str, DataFrame]:
    """Compute the full volatility-RCA panel.

    Parameters
    ----------
    ytd_cutoff_week : ISO week the YTD-comparable table uses for every year.
    max_actual_year : drop rows beyond this year (forecast horizon has no actuals).
    min_actual_year : lower bound on the analysis window.
    max_year_week   : optional fine-grained upper bound (YYYYWW int). Takes precedence
                      over max_actual_year if set — e.g. 202618 means "everything up to
                      and including 2026-W18".
    rolling_windows : window sizes for rolling CV (in weeks).
    top_n_recent_keys     : how many top sellers (by recent_window_weeks sum) to spotlight.
    recent_window_weeks   : window (in weeks of non-zero data) used to rank top sellers.
                            Trailing zero weeks are skipped so feed-lag gaps don't bias.
    high_vol_cv_threshold : keys with CV strictly greater than this are counted as
                            "high volatility" for the year-over-year (and month-over-
                            month) count charts.
    min_weeks_per_month   : drop (key, month) cells with fewer than this many observed
                            weeks — avoids spurious CV/class flips from sparse buckets.
    """
    base = _add_calendar_cols(sdf, year_week_col)
    base = base.filter(F.col("_year").between(min_actual_year, max_actual_year))
    if max_year_week is not None:
        base = base.filter(F.col(year_week_col).cast("int") <= int(max_year_week))
    base = base.select(key_col, year_week_col, "_year", "_week", "_iso_date",
                       "_year_month", "_month_start",
                       F.col(sales_col).cast("double").alias(sales_col))

    # ------------------------------------------------------------------ FY
    fy = (base.groupBy(key_col, F.col("_year").alias("year"))
              .agg(*_metric_agg(sales_col)))
    fy = _add_derived(fy)

    # ----------------------------------------------------------- YTD slice
    ytd_slice = base.filter(F.col("_week") <= ytd_cutoff_week)
    ytd = (ytd_slice.groupBy(key_col, F.col("_year").alias("year"))
                    .agg(*_metric_agg(sales_col)))
    ytd = _add_derived(ytd).withColumn("ytd_cutoff_week", F.lit(ytd_cutoff_week))

    # ----------------------------------------------------------- portfolio
    port = (base.groupBy(F.col("_year").alias("year"))
                .agg(*_metric_agg(sales_col)))
    port = _add_derived(port)

    # --------------------------------------------- rolling CV (per-key, all windows)
    roll_key = _rolling_cv_cols(base,
                                key_cols=[key_col],
                                sales_col=sales_col,
                                order_col="_iso_date",
                                windows=rolling_windows)
    roll_key = roll_key.select(
        key_col, year_week_col, "_year", "_week", "_iso_date", sales_col,
        *[f"rolling_cv_{w}w" for w in rolling_windows],
    )

    # ---------------------------------------- rolling CV (portfolio, pooled by week)
    port_weekly = (base.groupBy(year_week_col, "_year", "_week", "_iso_date")
                       .agg(F.sum(sales_col).alias(sales_col)))
    roll_port = _rolling_cv_cols(port_weekly.withColumn("_grp", F.lit(1)),
                                 key_cols=["_grp"],
                                 sales_col=sales_col,
                                 order_col="_iso_date",
                                 windows=rolling_windows).drop("_grp")
    roll_port = roll_port.select(
        year_week_col, "_year", "_week", "_iso_date", sales_col,
        *[f"rolling_cv_{w}w" for w in rolling_windows],
    )

    # ----------------------------------------------------- YoY delta block
    cur, prev = fy.alias("cur"), fy.alias("prev")
    yoy = (cur.join(prev,
                    (F.col(f"cur.{key_col}") == F.col(f"prev.{key_col}")) &
                    (F.col("cur.year") == F.col("prev.year")+1),
                    "left")
              .select(F.col(f"cur.{key_col}").alias(key_col),
                      F.col("cur.year").alias("year"),
                      F.col("cur.cv").alias("cv"),
                      F.col("prev.cv").alias("cv_prev"),
                      (F.col("cur.cv")-F.col("prev.cv")).alias("delta_cv"),
                      F.col("cur.cv2_nz").alias("cv2_nz"),
                      F.col("prev.cv2_nz").alias("cv2_nz_prev"),
                      (F.col("cur.cv2_nz")-F.col("prev.cv2_nz")).alias("delta_cv2_nz"),
                      F.col("cur.adi").alias("adi"),
                      F.col("prev.adi").alias("adi_prev"),
                      (F.col("cur.adi")-F.col("prev.adi")).alias("delta_adi"),
                      F.col("cur.sb_class").alias("sb_class"),
                      F.col("prev.sb_class").alias("sb_class_prev"),
                      F.when(F.col("cur.sb_class") != F.col("prev.sb_class"),
                             F.lit(True)).otherwise(F.lit(False)).alias("class_changed")))

    # ----------------------------------- top sellers in latest weeks (key-level focus)
    top_keys = _top_keys_by_recent_sales(base,
                                         key_col=key_col,
                                         sales_col=sales_col,
                                         year_week_col=year_week_col,
                                         last_n_weeks=recent_window_weeks,
                                         top_n=top_n_recent_keys)

    # volatility_by_fy restricted to the top sellers — useful for "show me these keys'
    # CV moved" tables/heatmaps without dragging the whole panel along.
    vol_top = (fy.join(top_keys.select(key_col, "recent_sales", "rank"), key_col, "inner")
                 .orderBy("rank", "year"))

    # ------------------------------------------ Syntetos-Boylan class counts per year
    class_counts = (fy.groupBy("year")
                      .pivot("sb_class",
                             ["smooth", "intermittent", "erratic", "lumpy", "noclass"])
                      .agg(F.countDistinct(key_col))
                      .na.fill(0)
                      .withColumn("total_keys",
                                  F.col("smooth") + F.col("intermittent")
                                  + F.col("erratic") + F.col("lumpy") + F.col("noclass"))
                      .orderBy("year"))

    # ----------------------------------------- high-volatility key counts per year
    high_vol = (fy
                .withColumn("is_high_vol",
                            F.when(F.col("cv") > F.lit(high_vol_cv_threshold), 1).otherwise(0))
                .withColumn("is_lumpy_or_erratic",
                            F.when(F.col("sb_class").isin("lumpy", "erratic"), 1).otherwise(0))
                .groupBy("year")
                .agg(F.countDistinct(key_col).alias("total_keys_with_data"),
                     F.sum("is_high_vol").alias("keys_cv_gt_threshold"),
                     F.sum("is_lumpy_or_erratic").alias("keys_lumpy_or_erratic"),
                     F.lit(float(high_vol_cv_threshold)).alias("cv_threshold"))
                .withColumn("pct_cv_gt_threshold",
                            _safe_div(F.col("keys_cv_gt_threshold"), F.col("total_keys_with_data")))
                .withColumn("pct_lumpy_or_erratic",
                            _safe_div(F.col("keys_lumpy_or_erratic"), F.col("total_keys_with_data")))
                .orderBy("year"))

    # =====================================================================
    # MONTHLY block — same recipe as the FY block but on calendar-month buckets
    # =====================================================================
    month_raw = (base.groupBy(key_col, "_year_month", "_month_start")
                     .agg(*_metric_agg(sales_col)))
    month_raw = (_add_derived(month_raw)
                 .withColumnRenamed("_year_month", "year_month")
                 .withColumnRenamed("_month_start", "month_start"))

    # Mask cells with too few weeks — avoids 1-week months distorting trend.
    mask_thin = F.col("weeks") < F.lit(int(min_weeks_per_month))
    month_vol = (month_raw
        .withColumn("cv",        F.when(mask_thin, F.lit(None)).otherwise(F.col("cv")))
        .withColumn("cv2_nz",    F.when(mask_thin, F.lit(None)).otherwise(F.col("cv2_nz")))
        .withColumn("adi",       F.when(mask_thin, F.lit(None)).otherwise(F.col("adi")))
        .withColumn("sb_class",  F.when(mask_thin, F.lit("noclass"))
                                  .otherwise(F.col("sb_class")))
        .withColumn("enough_weeks", ~mask_thin))

    # SB-class counts per month (only counts cells with enough_weeks=True)
    class_counts_m = (month_vol.filter("enough_weeks")
        .groupBy("year_month", "month_start")
        .pivot("sb_class", ["smooth", "intermittent", "erratic", "lumpy"])
        .agg(F.countDistinct(key_col))
        .na.fill(0)
        .withColumn("total_keys",
                    F.col("smooth") + F.col("intermittent")
                    + F.col("erratic") + F.col("lumpy"))
        .orderBy("year_month"))

    # High-vol key counts per month
    high_vol_m = (month_vol.filter("enough_weeks")
        .withColumn("is_high_vol",
                    F.when(F.col("cv") > F.lit(high_vol_cv_threshold), 1).otherwise(0))
        .withColumn("is_lumpy_or_erratic",
                    F.when(F.col("sb_class").isin("lumpy", "erratic"), 1).otherwise(0))
        .groupBy("year_month", "month_start")
        .agg(F.countDistinct(key_col).alias("total_keys_with_data"),
             F.sum("is_high_vol").alias("keys_cv_gt_threshold"),
             F.sum("is_lumpy_or_erratic").alias("keys_lumpy_or_erratic"),
             F.lit(float(high_vol_cv_threshold)).alias("cv_threshold"))
        .withColumn("pct_cv_gt_threshold",
                    _safe_div(F.col("keys_cv_gt_threshold"), F.col("total_keys_with_data")))
        .withColumn("pct_lumpy_or_erratic",
                    _safe_div(F.col("keys_lumpy_or_erratic"), F.col("total_keys_with_data")))
        .orderBy("year_month"))

    # Monthly volatility restricted to the top recent keys (for line chart)
    vol_top_m = (month_vol.join(top_keys.select(key_col, "recent_sales", "rank"),
                                key_col, "inner")
                          .orderBy("rank", "year_month"))

    # ----------- Trailing-window monthly snapshot (smoother trend view) ----------
    trail_win = max(rolling_windows)   # use the longest configured window (default 13)
    monthly_trailing = _monthly_trailing_snapshot(
        base, key_col=key_col, sales_col=sales_col, window_weeks=trail_win)

    # Class counts per month — based on TRAILING-window classification
    class_counts_tr = (monthly_trailing
        .filter(F.col("trailing_sb_class") != "noclass")
        .groupBy("year_month", "month_start")
        .pivot("trailing_sb_class", ["smooth", "intermittent", "erratic", "lumpy"])
        .agg(F.countDistinct(key_col))
        .na.fill(0)
        .withColumn("total_keys",
                    F.col("smooth") + F.col("intermittent")
                    + F.col("erratic") + F.col("lumpy"))
        .withColumn("trailing_window_weeks", F.lit(int(trail_win)))
        .orderBy("year_month"))

    # ============================================================================
    # MONTHLY VOLATILITY SUMMARY — one row per month, ALL keys included
    # ============================================================================
    # This is the simple "actuals volatility, month by month" view.  Unlike the
    # class-counts table above (which excludes keys with too-sparse trailing data),
    # this rollup includes every key that had any sales in the trailing window.
    # Use this for the headline "volatility is climbing" chart and the WAPE-style
    # numbers you'd quote to the business.
    monthly_summary = (monthly_trailing
        .groupBy("year_month", "month_start")
        .agg(
            # Coverage
            F.countDistinct(key_col).alias("keys_in_panel"),
            F.count(F.when(F.col("trailing_cv").isNotNull(), 1)).alias("keys_with_cv"),
            # CV distribution across keys (use percentile_approx for scalable percentiles)
            F.mean("trailing_cv").alias("avg_cv"),
            F.expr("percentile_approx(trailing_cv, 0.50)").alias("median_cv"),
            F.expr("percentile_approx(trailing_cv, 0.75)").alias("p75_cv"),
            F.expr("percentile_approx(trailing_cv, 0.90)").alias("p90_cv"),
            F.expr("percentile_approx(trailing_cv, 0.95)").alias("p95_cv"),
            # Count of keys above CV thresholds — multiple cuts so the threshold
            # discussion is data-driven, not eyeballed.
            F.sum(F.when(F.col("trailing_cv") > 0.5, 1).otherwise(0))
                .alias("n_keys_cv_gt_0_5"),
            F.sum(F.when(F.col("trailing_cv") > 1.0, 1).otherwise(0))
                .alias("n_keys_cv_gt_1_0"),
            F.sum(F.when(F.col("trailing_cv") > 1.5, 1).otherwise(0))
                .alias("n_keys_cv_gt_1_5"),
            # ADI / sparsity distribution
            F.mean("trailing_adi").alias("avg_adi"),
            F.expr("percentile_approx(trailing_adi, 0.50)").alias("median_adi"),
            F.lit(int(trail_win)).alias("trailing_window_weeks"),
        )
        # Convert raw counts to percentages — easier to read at scale
        .withColumn("pct_cv_gt_0_5",
                    _safe_div(F.col("n_keys_cv_gt_0_5"), F.col("keys_with_cv")))
        .withColumn("pct_cv_gt_1_0",
                    _safe_div(F.col("n_keys_cv_gt_1_0"), F.col("keys_with_cv")))
        .withColumn("pct_cv_gt_1_5",
                    _safe_div(F.col("n_keys_cv_gt_1_5"), F.col("keys_with_cv")))
        .orderBy("year_month"))

    # High-vol counts per month — trailing-window definitions
    high_vol_tr = (monthly_trailing
        .filter(F.col("trailing_sb_class") != "noclass")
        .withColumn("is_high_vol",
                    F.when(F.col("trailing_cv") > F.lit(high_vol_cv_threshold), 1)
                     .otherwise(0))
        .withColumn("is_lumpy_or_erratic",
                    F.when(F.col("trailing_sb_class").isin("lumpy", "erratic"), 1)
                     .otherwise(0))
        .groupBy("year_month", "month_start")
        .agg(F.countDistinct(key_col).alias("total_keys_with_data"),
             F.sum("is_high_vol").alias("keys_cv_gt_threshold"),
             F.sum("is_lumpy_or_erratic").alias("keys_lumpy_or_erratic"),
             F.lit(float(high_vol_cv_threshold)).alias("cv_threshold"),
             F.lit(int(trail_win)).alias("trailing_window_weeks"))
        .withColumn("pct_cv_gt_threshold",
                    _safe_div(F.col("keys_cv_gt_threshold"), F.col("total_keys_with_data")))
        .withColumn("pct_lumpy_or_erratic",
                    _safe_div(F.col("keys_lumpy_or_erratic"), F.col("total_keys_with_data")))
        .orderBy("year_month"))

    # ============================================================================
    # MONTHLY ACTUALS DISTRIBUTION — the simple "actuals are getting more volatile,
    # but the panel size is stable" view.  Per month it returns:
    #   - distinct_keys      (panel size — should be roughly flat)
    #   - mean / std / CV of weekly smooth_sales across the whole month
    #   - percentiles  (p10 .. p99) of weekly smooth_sales
    #   - same percentiles restricted to non-zero observations
    # If the percentile fan widens over time while distinct_keys stays flat,
    # that IS the volatility-of-actuals story, with no SB classification needed.
    # ============================================================================
    s_all = F.col(sales_col)
    s_nz  = F.when(s_all > 0, s_all)
    monthly_dist = (base.groupBy("_year_month", "_month_start")
        .agg(
            F.countDistinct(key_col).alias("distinct_keys"),
            F.count(F.lit(1)).alias("weekly_obs"),
            F.sum(F.when(s_all > 0, 1).otherwise(0)).alias("nonzero_obs"),
            F.sum(s_all).alias("portfolio_total"),
            F.mean(s_all).alias("mean_weekly"),
            F.stddev_samp(s_all).alias("std_weekly"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.10)").alias("p10_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.25)").alias("p25_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.50)").alias("p50_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.75)").alias("p75_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.90)").alias("p90_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.95)").alias("p95_all"),
            F.expr(f"percentile_approx(`{sales_col}`, 0.99)").alias("p99_all"),
            F.max(s_all).alias("max_weekly"),
            # Non-zero distribution — useful for intermittent panels where p25_all = 0
            F.mean(s_nz).alias("mean_nz"),
            F.stddev_samp(s_nz).alias("std_nz"),
            F.expr(f"percentile_approx(case when `{sales_col}` > 0 then `{sales_col}` end, 0.50)").alias("p50_nz"),
            F.expr(f"percentile_approx(case when `{sales_col}` > 0 then `{sales_col}` end, 0.75)").alias("p75_nz"),
            F.expr(f"percentile_approx(case when `{sales_col}` > 0 then `{sales_col}` end, 0.90)").alias("p90_nz"),
            F.expr(f"percentile_approx(case when `{sales_col}` > 0 then `{sales_col}` end, 0.95)").alias("p95_nz"),
        )
        .withColumn("cv_weekly",  _safe_div(F.col("std_weekly"), F.col("mean_weekly")))
        .withColumn("cv_nz",      _safe_div(F.col("std_nz"),     F.col("mean_nz")))
        .withColumn("iqr_all",    F.col("p75_all") - F.col("p25_all"))
        .withColumn("p10_p90",    F.col("p90_all") - F.col("p10_all"))
        .withColumnRenamed("_year_month", "year_month")
        .withColumnRenamed("_month_start", "month_start")
        .orderBy("year_month"))

    return {
        "volatility_by_fy":        fy.orderBy(key_col, "year"),
        "volatility_by_ytd":       ytd.orderBy(key_col, "year"),
        "portfolio_by_fy":         port.orderBy("year"),
        "rolling_cv_per_key":      roll_key.orderBy(key_col, "_iso_date"),
        "rolling_cv_portfolio":    roll_port.orderBy("_iso_date"),
        "yoy_delta":               yoy.orderBy(key_col, "year"),
        "top_keys_recent":         top_keys,
        "volatility_top_keys_fy":  vol_top,
        "class_counts_by_year":    class_counts,
        "high_vol_counts_by_year": high_vol,
        "volatility_by_month":     month_vol.orderBy(key_col, "year_month"),
        "class_counts_by_month":   class_counts_m,
        "high_vol_counts_by_month":high_vol_m,
        "vol_top_keys_monthly":    vol_top_m,
        "monthly_trailing_snapshot":        monthly_trailing.orderBy(key_col, "year_month"),
        "class_counts_trailing_monthly":    class_counts_tr,
        "high_vol_counts_trailing_monthly": high_vol_tr,
        "monthly_volatility_summary":       monthly_summary,
        "monthly_actuals_distribution":     monthly_dist,
    }


# --------------------------------------------------------------------------- #
# Multi-group helper: run RCA at TOTAL + per Product Cluster (PC)             #
# --------------------------------------------------------------------------- #
def run_volatility_rca_by_pc(
    sdf: DataFrame,
    *,
    pc_mapping: Dict[str, str],
    category_col: str,
    include_total: bool = True,
    unmapped_label: str = "UNMAPPED",
    include_unmapped_in_total: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, DataFrame]]:
    """Run `run_volatility_rca` once at TOTAL level and once per PC bucket.

    Parameters
    ----------
    pc_mapping     : dict mapping category_name -> PC, e.g.
                     {"CONDIMENT": "FOODS", "HAIR CARE": "BPC", ...}
    category_col   : column in `sdf` holding the category names (keys of pc_mapping)
    include_total  : if True, also run on the whole panel and key the result as "TOTAL"
    unmapped_label : label to use for any category not present in the mapping
    include_unmapped_in_total : keep unmapped categories in the TOTAL run (default True)
    **kwargs       : forwarded to `run_volatility_rca` (e.g. max_year_week=202614)

    Returns
    -------
    Dict[str, Dict[str, DataFrame]] keyed by "TOTAL" (if requested) plus each unique
    PC value.  Each entry is the full result dict from run_volatility_rca.

    Example
    -------
        PC_MAPPING = {
            "CONDIMENT":  "FOODS",   "MINI MEAL":     "FOODS",
            "HOME & HYGIENE": "HOME CARE",
            "HAIR CARE":  "BPC",     "ORAL CARE":     "BPC",
            ...
        }
        results = run_volatility_rca_by_pc(
            df, pc_mapping=PC_MAPPING, category_col="category_name",
            max_year_week=202614, ytd_cutoff_week=14,
        )
        results["TOTAL"]["monthly_actuals_distribution"].display()
        results["FOODS"]["monthly_actuals_distribution"].display()
        results["BPC"]["monthly_actuals_distribution"].display()
    """
    spark = sdf.sparkSession
    mapping_df = spark.createDataFrame(
        list(pc_mapping.items()), ["__cat__", "__pc__"])

    sdf_pc = (sdf.join(mapping_df,
                       sdf[category_col] == mapping_df["__cat__"],
                       "left")
                 .withColumn("__pc__",
                             F.coalesce(F.col("__pc__"), F.lit(unmapped_label)))
                 .drop("__cat__"))

    results: Dict[str, Dict[str, DataFrame]] = {}

    if include_total:
        total_sdf = (sdf_pc if include_unmapped_in_total
                     else sdf_pc.filter(F.col("__pc__") != unmapped_label))
        results["TOTAL"] = run_volatility_rca(total_sdf.drop("__pc__"), **kwargs)

    pcs = sorted({v for v in pc_mapping.values()})
    for pc in pcs:
        sub = sdf_pc.filter(F.col("__pc__") == pc).drop("__pc__")
        results[pc] = run_volatility_rca(sub, **kwargs)

    # Also include UNMAPPED if any rows landed there (so the user can audit)
    unmapped_count = (sdf_pc.filter(F.col("__pc__") == unmapped_label)
                            .limit(1).count())
    if unmapped_count > 0 and unmapped_label not in results:
        sub = sdf_pc.filter(F.col("__pc__") == unmapped_label).drop("__pc__")
        results[unmapped_label] = run_volatility_rca(sub, **kwargs)

    return results


def stack_output(results: Dict[str, Dict[str, DataFrame]],
                 output_key: str,
                 group_col: str = "pc_group") -> DataFrame:
    """Stack one named output across the PC groups produced by run_volatility_rca_by_pc.

    Example:
        combined = stack_output(results, "monthly_actuals_distribution")
        combined.display()   # one row per (pc_group, year_month)
    """
    combined = None
    for pc, res in results.items():
        sub = res[output_key].withColumn(group_col, F.lit(pc))
        combined = sub if combined is None else combined.unionByName(sub)
    return combined


# --------------------------------------------------------------------------- #
# Databricks/CLI entry point                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, sys
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser(description="Volatility RCA")
    p.add_argument("--input",  required=True, help="Spark table or CSV path")
    p.add_argument("--output_dir", required=True, help="Folder for parquet outputs")
    p.add_argument("--key_col",       default="key")
    p.add_argument("--year_week_col", default="year_week")
    p.add_argument("--sales_col",     default="smooth_sales")
    p.add_argument("--ytd_cutoff_week", type=int, default=21)
    p.add_argument("--max_actual_year", type=int, default=2026)
    p.add_argument("--min_actual_year", type=int, default=2023)
    p.add_argument("--max_year_week",   type=int, default=None,
                   help="optional fine-grained upper bound, e.g. 202618")
    p.add_argument("--rolling_windows", type=int, nargs="+", default=[4, 8, 13],
                   help="rolling CV window sizes in weeks")
    p.add_argument("--top_n_recent_keys",     type=int,   default=20)
    p.add_argument("--recent_window_weeks",   type=int,   default=13)
    p.add_argument("--high_vol_cv_threshold", type=float, default=1.0)
    p.add_argument("--min_weeks_per_month",   type=int,   default=3)
    args = p.parse_args()

    spark = (SparkSession.builder.appName("volatility_rca").getOrCreate())
    if args.input.endswith(".csv"):
        sdf = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    else:
        sdf = spark.read.table(args.input)

    out = run_volatility_rca(sdf,
                             key_col=args.key_col,
                             year_week_col=args.year_week_col,
                             sales_col=args.sales_col,
                             ytd_cutoff_week=args.ytd_cutoff_week,
                             max_actual_year=args.max_actual_year,
                             min_actual_year=args.min_actual_year,
                             max_year_week=args.max_year_week,
                             rolling_windows=tuple(args.rolling_windows),
                             top_n_recent_keys=args.top_n_recent_keys,
                             recent_window_weeks=args.recent_window_weeks,
                             high_vol_cv_threshold=args.high_vol_cv_threshold,
                             min_weeks_per_month=args.min_weeks_per_month)

    for name, df in out.items():
        df.write.mode("overwrite").parquet(f"{args.output_dir}/{name}")
        print(f"wrote {args.output_dir}/{name} (rows={df.count()})", file=sys.stderr)
