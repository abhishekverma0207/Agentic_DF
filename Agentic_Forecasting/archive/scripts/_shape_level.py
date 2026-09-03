"""Combine SHAPE from an accurate forecast x LEVEL from a growth-capturing forecast: keep each key's
weekly shape from `shape_f` (high per-week accuracy) but rescale its total to match `level_f` (which
captures the YoY growth -> low bias). Per-key total scaling => accurate shape AT the right level, the
decomposition that beats the acc<->bias frontier on growth-heavy categories.
Usage: _shape_level.py <shape.parquet> <level.parquet> <out.parquet> [cap=4.0]"""
import sys, numpy as np, pandas as pd

shape_f, level_f, out_f = sys.argv[1], sys.argv[2], sys.argv[3]
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
shape = pd.read_parquet(shape_f); shape["key"] = shape["key"].astype(str); shape["year_week"] = shape["year_week"].astype(int)
level = pd.read_parquet(level_f); level["key"] = level["key"].astype(str)
st = shape.groupby("key")["predicted"].sum()
lt = level.groupby("key")["predicted"].sum()
fac = (lt.reindex(st.index) / st.replace(0, np.nan)).clip(1.0 / cap, cap).fillna(1.0)
out = shape.copy(); out["predicted"] = out["predicted"] * out["key"].map(fac).fillna(1.0)
out[["key", "year_week", "predicted"]].to_parquet(out_f, index=False)
print(f"shape={shape_f.split('/')[-1]} x level={level_f.split('/')[-1]} -> {out_f.split('/')[-1]}; "
      f"keys {len(fac)}, median factor {fac.median():.2f}, p90 {fac.quantile(0.9):.2f}")
