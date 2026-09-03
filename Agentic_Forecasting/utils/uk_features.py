"""
Feature engineering helpers for the full 294-column input — collapses the families
of one-hot promo flags into proper categorical columns, and pulls signal out of the
list-string fields (`promo_list_of_*_shipped`).

Why: LightGBM finds signal in a categorical split (one decision over a 9-way enum)
far better than across 9 sparse 0/1 columns. The original input ships both forms —
the one-hots AND the actual string list — but the trimmed cache only kept the one-hots.

Mapping
-------
- promo_primary_mechanic_*    (9 one-hots)  -> promo_mechanic         (categorical)
- promo_secondary_mechanic_*  (~25 one-hots)-> promo_secondary        (categorical)
- promo_feature_*             (~20 one-hots)-> promo_feature          (categorical)
- promo_primary_objective_*   (7 one-hots)  -> promo_objective        (categorical)
- promo_secondary_objective_* (3 one-hots)  -> promo_sec_objective    (categorical)
- promo_instore_primary_mechanic_* + family -> promo_instore_mechanic / feature / objective
- promo_list_of_*_shipped (string)          -> used DIRECTLY where populated; the one-hot
                                                argmax is the fallback.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _onehot_argmax(df: pd.DataFrame, prefix: str, strip_prefix: bool = True) -> pd.Series:
    """For a family of one-hot columns sharing a prefix, return a Series naming the
    column with the largest value per row (or 'NONE' if all zero). Multi-positive
    rows are resolved by argmax (sum-stable across runs since one-hots are 0/1)."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(["NONE"] * len(df), index=df.index)
    block = df[cols].fillna(0).to_numpy(dtype="float32")
    any_pos = block.sum(axis=1) > 0
    idx = block.argmax(axis=1)                    # ties -> lowest index (deterministic)
    out = np.where(any_pos, np.array(cols)[idx], "NONE")
    if strip_prefix:
        out = np.where(any_pos, np.char.replace(out.astype(str), prefix, ""), "NONE")
    return pd.Series(out, index=df.index, dtype="object")


def _first_listed(s: pd.Series, sep: str = ",") -> pd.Series:
    """For a comma/list string column, return the first element (uppercased, trimmed).
    Empty / NaN / 'NONE' / '0' (the upstream no-promo sentinel) -> 'NONE'."""
    x = s.fillna("").astype(str).str.strip()
    head = x.str.split(sep, n=1).str[0].str.strip().str.upper()
    head = head.where(~head.isin(["", "0", "NONE", "NAN"]), "NONE")
    return head


def _prefer_string(df: pd.DataFrame, string_col: str, onehot_prefix: str) -> pd.Series:
    """Where the string-list field is populated, use it; else fall back to one-hot argmax.
    The list fields ship the ACTUAL human-readable mechanic/feature (e.g. 'MultiBuy'),
    so they're cleaner than the prefix-stripped one-hot tag — when present."""
    fallback = _onehot_argmax(df, onehot_prefix)
    if string_col not in df.columns:
        return fallback
    s = _first_listed(df[string_col])
    return s.where(s != "NONE", fallback)


COLLAPSE_GROUPS: List[Tuple[str, Optional[str], str]] = [
    # (output_col,          string_list_col_if_any,                              onehot_prefix)
    ("promo_mechanic",       "promo_list_of_primary_mechanics_shipped",          "promo_primary_mechanic_"),
    ("promo_secondary",      None,                                               "promo_secondary_mechanic_"),
    ("promo_feature",        "promo_list_of_promo_features_shipped",             "promo_feature_"),
    ("promo_objective",      None,                                               "promo_primary_objective_"),
    ("promo_sec_objective",  None,                                               "promo_secondary_objective_"),
    # in-store promo mirror — same idea
    ("promo_instore_mechanic",   None, "promo_instore_primary_mechanic_"),
    ("promo_instore_secondary",  None, "promo_instore_secondary_mechanic_"),
    ("promo_instore_feature",    None, "promo_instore_feature_"),
    ("promo_instore_objective",  None, "promo_instore_primary_objective_"),
]


def collapse_promo_groups(df: pd.DataFrame, drop_onehots: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Add collapsed categorical columns to df; optionally drop the source one-hots
    (default True — they're now redundant). Returns (df, list_of_new_cat_cols).

    Idempotent: if the categorical already exists, it's overwritten."""
    new_cols: List[str] = []
    drops: List[str] = []
    for out_col, str_col, prefix in COLLAPSE_GROUPS:
        df[out_col] = _prefer_string(df, str_col, prefix) if str_col else _onehot_argmax(df, prefix)
        new_cols.append(out_col)
        if drop_onehots:
            drops += [c for c in df.columns if c.startswith(prefix)]
            if str_col and str_col in df.columns:
                drops.append(str_col)
    if drops:
        df = df.drop(columns=list(set(drops)), errors="ignore")
    return df, new_cols


__all__ = ["collapse_promo_groups", "COLLAPSE_GROUPS"]
