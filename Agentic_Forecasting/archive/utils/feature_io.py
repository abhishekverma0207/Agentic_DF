"""
Format-agnostic I/O helpers for the train/val/test_features intermediate files.

Background
----------
Feature engineering used to write `train_features.csv`, `val_features.csv`,
and `test_features.csv` directly with `pd.to_csv`.  On a UK-scale frame
(~5M rows x ~300 cols) this is glacial:

  * to_csv:        ~3-5 minutes per file (text serialisation, no compression)
  * read_csv:      ~1-2 minutes per file (text parsing, dtype guessing)
  * disk size:     ~5-10 GB per file
  * dtype loss:    int year_week becomes string '202401' on round-trip,
                   forcing every consumer to re-coerce

Switching to parquet wins on every dimension:

  * to_parquet:    ~10-30 seconds (10-30x faster)
  * read_parquet:  ~5-15 seconds (10-30x faster)
  * disk size:     ~1-3 GB (3-5x smaller, snappy-compressed)
  * dtype preserved: year_week stays int, float32 stays float32

Migration strategy
------------------
**Format-agnostic with parquet-first writes.**

* Writers (`write_features_intermediate`) save parquet by default and remove
  any stale `.csv` at the same base name to prevent confusion.  If
  pyarrow is unavailable they fall back to CSV transparently.
* Readers (`read_features_intermediate`) try `.parquet` first then
  fall back to `.csv`.  This means:
    - Fresh runs use parquet end-to-end (the fast path).
    - Old runs that already have `.csv` files keep working.
    - Mixed-format scenarios (e.g. one stage writes parquet, an older
      tool re-emits CSV) keep working.

This file is the single source of truth for these conversions.  Every
consumer that historically did `pd.read_csv(os.path.join(feat_dir,
"train_features.csv"))` should call `read_features_intermediate(feat_dir,
"train_features")` instead — note the BASE NAME (no extension) is passed
so the helper can pick the right one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# The three canonical base names.  Other modules can use any base name,
# but these are the ones we know are written by feature engineering.
FEATURE_BASE_NAMES = ("train_features", "val_features", "test_features")

# Suffixes we recognise.  Order matters: we prefer parquet because it's
# faster and dtype-preserving.
_RECOGNISED_SUFFIXES = (".parquet", ".pq", ".csv")
_PARQUET_SUFFIXES = (".parquet", ".pq")


def _strip_known_suffix(name: str) -> str:
    """Accept either bare 'train_features' or 'train_features.csv' /
    'train_features.parquet' for caller convenience."""
    for s in _RECOGNISED_SUFFIXES:
        if name.lower().endswith(s):
            return name[: -len(s)]
    return name


def features_intermediate_path(
    feat_dir: Union[str, os.PathLike],
    base_name: str,
) -> Optional[Path]:
    """Return the actual on-disk path for ``base_name`` in ``feat_dir``,
    or None if neither parquet nor CSV exists.

    Parquet is preferred when both exist (it's the fast path).
    """
    base_name = _strip_known_suffix(base_name)
    feat_dir = Path(feat_dir)
    for suffix in _RECOGNISED_SUFFIXES:
        p = feat_dir / f"{base_name}{suffix}"
        if p.exists():
            return p
    return None


def features_intermediate_exists(
    feat_dir: Union[str, os.PathLike],
    base_name: str,
) -> bool:
    """True if ``{feat_dir}/{base_name}.parquet`` OR
    ``{feat_dir}/{base_name}.csv`` exists."""
    return features_intermediate_path(feat_dir, base_name) is not None


def read_features_intermediate(
    feat_dir: Union[str, os.PathLike],
    base_name: str,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    **csv_kwargs,
) -> pd.DataFrame:
    """Read ``base_name`` from ``feat_dir`` regardless of format.

    Tries parquet first (fast path).  Falls back to CSV.

    Parameters
    ----------
    feat_dir : str | Path
        Directory holding the intermediate features file.
    base_name : str
        Either ``'train_features'`` or ``'train_features.csv'`` /
        ``'train_features.parquet'`` — the suffix is stripped before lookup.
    columns : List[str], optional
        Column projection (works for both formats).
    nrows : int, optional
        Row limit.  For CSV this is passed through; for parquet we
        materialise then truncate (parquet has no native nrows).
    **csv_kwargs
        Extra kwargs forwarded to ``pd.read_csv`` ONLY (silently ignored
        for parquet — they're CSV-specific and meaningless for a
        dtype-preserving format).  The most common ones are
        ``low_memory=False`` and ``dtype={...}``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError if neither format exists at ``feat_dir/{base_name}``.
    """
    p = features_intermediate_path(feat_dir, base_name)
    if p is None:
        candidates = ", ".join(
            f"{base_name}{s}" for s in _RECOGNISED_SUFFIXES
        )
        raise FileNotFoundError(
            f"None of [{candidates}] found in {feat_dir!s}. "
            "Has feature engineering run for this category?"
        )

    if p.suffix.lower() in _PARQUET_SUFFIXES:
        df = pd.read_parquet(str(p), columns=columns)
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)
        return df

    # CSV path — forward the kwargs verbatim.
    return pd.read_csv(
        str(p),
        usecols=columns,
        nrows=nrows,
        **csv_kwargs,
    )


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the all-null padding rows the feature pipeline can append (key/period
    None) and force a numeric ``year_week`` to ONE clean YYYYWW string type.

    This is the single fix for a recurring bug on TH data: the panels accrue rows
    with ``key=None, year_week=None`` and mixed str/float ``year_week`` (CSV round-trip
    makes it numeric, parquet keeps it str), which crashes any downstream
    ``sorted(year_week)`` / ``year_week < cutoff`` with
    ``'<' not supported between 'str' and 'float'`` (DMH augmenters, recursive
    inference, validation calibration, reconciliation, parquet-write). Clean panels
    fix all of them. No-op for already-clean UK/DE panels.
    """
    if "key" in df.columns:
        df = df[df["key"].notna()]
    if "year_week" in df.columns:
        df = df[df["year_week"].notna()].copy()
        yw = pd.to_numeric(df["year_week"], errors="coerce")
        if len(df) and yw.notna().mean() > 0.99:   # numeric period (YYYYWW / YYYYMM)
            df = df[yw.notna()]
            df["year_week"] = pd.to_numeric(df["year_week"]).astype("int64").astype(str)
    return df


def write_features_intermediate(
    df: pd.DataFrame,
    feat_dir: Union[str, os.PathLike],
    base_name: str,
    *,
    prefer_parquet: bool = True,
) -> Path:
    """Write ``df`` to ``feat_dir`` using the chosen format.

    Default: parquet (fast + small + dtype-preserving).  If pyarrow /
    fastparquet is unavailable — a real possibility on a stripped-down
    cluster — silently falls back to CSV so the pipeline keeps running.

    Cleans up any STALE alternate-format file at the same base name so
    a future read doesn't pick up data from a previous run.  Without
    this cleanup, a category re-run that switches from CSV to parquet
    would leave the old CSV next to the new parquet, and a downstream
    reader picking the wrong one would silently train on stale data.

    Parameters
    ----------
    df : pd.DataFrame
    feat_dir : str | Path
        Output directory (created if missing).
    base_name : str
        Bare name without extension, e.g. ``'train_features'``.  Trailing
        .csv / .parquet are tolerated for caller convenience.
    prefer_parquet : bool
        If False, write CSV directly without trying parquet.  Useful for
        compat tests where downstream legacy tooling expects CSV.

    Returns
    -------
    Path
        The actual file written.
    """
    base_name = _strip_known_suffix(base_name)
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    df = _sanitize_features(df)   # drop null padding rows + clean year_week dtype

    parquet_path = feat_dir / f"{base_name}.parquet"
    csv_path = feat_dir / f"{base_name}.csv"

    if prefer_parquet:
        try:
            df.to_parquet(parquet_path, index=False)
            # Stale-format cleanup: remove the .csv from a prior run.
            if csv_path.exists():
                try:
                    csv_path.unlink()
                    logger.info(
                        "[feature_io] removed stale CSV %s after writing parquet",
                        csv_path.name,
                    )
                except OSError as exc:
                    logger.warning(
                        "[feature_io] could not remove stale CSV %s: %s",
                        csv_path.name, exc,
                    )
            return parquet_path
        except Exception as exc:
            logger.warning(
                "[feature_io] parquet write failed (%s); falling back to CSV",
                exc,
            )

    df.to_csv(csv_path, index=False)
    # Stale-format cleanup the other direction too.
    if parquet_path.exists():
        try:
            parquet_path.unlink()
            logger.info(
                "[feature_io] removed stale parquet %s after writing CSV",
                parquet_path.name,
            )
        except OSError:
            pass
    return csv_path


__all__ = [
    "FEATURE_BASE_NAMES",
    "features_intermediate_path",
    "features_intermediate_exists",
    "read_features_intermediate",
    "write_features_intermediate",
]
