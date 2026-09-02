"""Split the 4.5 GB Thailand all-categories CSV into per-category Parquet files.

- Models ALL accounts (62) per category — the production deliverable. E1016 is just
  the LEGO eval subset.
- Excludes ICE CREAM (per project instruction).
- key = Model_Hierarchy (= CS_BARCODE_<account>; for E1016 == LEGO key).
- Keeps ALL weeks incl. future placeholder rows (needed for known-future features).
- Streams via pyarrow ParquetWriter with a FIXED schema → low memory, consistent dtypes.

Iteration knob: --nonE1016-frac F keeps a deterministic hash-sampled fraction F of
non-E1016 keys (E1016 keys ALWAYS kept) to shrink big categories for fast local runs.

Usage:
  .venv/bin/python scripts/prep_th_categories.py [--nonE1016-frac 1.0] [--out DIR]
"""
import argparse
import hashlib
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TH = "sourcedata/THAILAND/thailand_all_categories.csv"
DEFAULT_OUT = "sourcedata/THAILAND/by_category"
EXCLUDE_CATEGORIES = {"ICE CREAM"}

# Columns kept as strings; everything else coerced to float64 ('null' -> NaN).
# On output: Model_Hierarchy -> "key", Shipment_Week -> "year_week" (pipeline convention).
OUT_RENAME = {"Model_Hierarchy": "key", "Shipment_Week": "year_week"}
STRING_COLS = {
    "year_week", "key", "CS_BARCODE",
    "ACTPM_Planning_Account_Code", "Category", "Snapshot_Week", "PromoID",
}


def cat_filename(cat: str) -> str:
    s = cat.upper()
    for a, b in [("&", "AND"), ("/", "_"), (" ", "_")]:
        s = s.replace(a, b)
    while "__" in s:
        s = s.replace("__", "_")
    return s


def keep_key(mh: str, frac: float) -> bool:
    """Deterministic hash-sample. E1016 keys always kept."""
    if frac >= 1.0:
        return True
    if isinstance(mh, str) and mh.endswith("_E1016"):
        return True
    if not isinstance(mh, str):
        return False
    h = int(hashlib.md5(mh.encode()).hexdigest()[:8], 16)
    return (h % 1000) < int(frac * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonE1016-frac", type=float, default=1.0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chunksize", type=int, default=300_000)
    # Drop far-future placeholder rows (source runs to 202819). We forecast to
    # 202615; keep a lead-feature buffer to 202620. Avoids bloat + period bugs.
    ap.add_argument("--max-week", default="202620", help="keep year_week <= this (YYYYWW)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hdr = pd.read_csv(TH, nrows=0).columns.tolist()
    out_hdr = [OUT_RENAME.get(c, c) for c in hdr]  # rename on output
    schema = pa.schema(
        [(c, pa.string() if c in STRING_COLS else pa.float64()) for c in out_hdr]
    )
    num_cols = [c for c in out_hdr if c not in STRING_COLS]

    writers = {}   # cat -> ParquetWriter
    paths = {}     # cat -> path
    counts = {}    # cat -> rows
    n_in = n_out = 0

    reader = pd.read_csv(TH, dtype=str, chunksize=args.chunksize)
    for i, ch in enumerate(reader):
        n_in += len(ch)
        ch = ch[~ch["Category"].isin(EXCLUDE_CATEGORIES)]
        if ch.empty:
            continue
        ch = ch.rename(columns=OUT_RENAME)
        # canonical pipeline period format: YYYYWW (no dash), DMH parses via int()
        ch["year_week"] = ch["year_week"].astype(str).str.replace("-", "", regex=False)
        ch = ch[ch["year_week"] <= args.max_week]
        if ch.empty:
            continue
        if args.nonE1016_frac < 1.0:
            mask = ch["key"].map(lambda m: keep_key(m, args.nonE1016_frac))
            ch = ch[mask]
        if ch.empty:
            continue
        for c in num_cols:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        for cat, sub in ch.groupby("Category"):
            tbl = pa.Table.from_pandas(sub[out_hdr], schema=schema, preserve_index=False)
            if cat not in writers:
                path = os.path.join(args.out, f"{cat_filename(cat)}.parquet")
                writers[cat] = pq.ParquetWriter(path, schema, compression="snappy")
                paths[cat] = path
                counts[cat] = 0
            writers[cat].write_table(tbl)
            counts[cat] += len(sub)
            n_out += len(sub)
        if i % 10 == 0:
            print(f"  chunk {i}: in={n_in:,} out={n_out:,}", flush=True)

    for w in writers.values():
        w.close()

    print("\n=== PREP COMPLETE ===", flush=True)
    print(f"rows read: {n_in:,} | rows written: {n_out:,} | nonE1016_frac={args.nonE1016_frac}")
    for cat in sorted(counts, key=lambda c: -counts[c]):
        print(f"  {cat:28s} {counts[cat]:>10,}  -> {paths[cat]}")


if __name__ == "__main__":
    main()
