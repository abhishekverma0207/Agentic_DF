"""Client deliverable: Excel workbook comparing Actual vs LEGO vs Our Model units at the
LEGO eval grain (CS_BARCODE x week, account E1016), for the forecast window 2026-03..15.
Sheet 1 'Detail' = one row per key x week. Sheet 2 'Summary' = per-category accuracy/bias
vs LEGO (t+4&5 and 13-week). Usage: _export_excel.py [final_forecast.parquet] [out.xlsx]"""
import sys, glob
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

FC = sys.argv[1] if len(sys.argv) > 1 else "artifacts_th_local/final_forecast.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "notebooks/rca_outputs/TH_Forecast_vs_LEGO.xlsx"


def _segments():
    """lego_segment per key from the LEGO benchmark (a product attribute), if available."""
    for f in glob.glob("Benchmark_Thailand/*.csv"):
        lb = pd.read_csv(f, usecols=lambda c: c in ("key", "lego_segment"))
        if "lego_segment" in lb.columns:
            return lb.dropna(subset=["key"]).drop_duplicates("key").set_index("key")["lego_segment"]
    return pd.Series(dtype=object)


def main():
    base = lego_eval.load_eval_base()
    fc = pd.read_parquet(FC)
    p = lego_eval.attach_forecast(base, fc, pred_col="predicted")  # cs x Shipment_Week, E1016 overlap
    p = p[p["Category"] != "ICE CREAM"].copy()  # ICE CREAM is excluded from scope
    # horizon label t+1..t+13 (t already on the frame from attach_forecast)
    p["key"] = p["cs"].astype(str) + "_E1016"
    seg = _segments()
    detail = pd.DataFrame({
        "Category": p["Category"],
        "CS_BARCODE": p["cs"].astype(str),
        "LEGO_Segment": p["key"].map(seg),
        "key": p["key"],
        "Week": p["Shipment_Week"].astype(str),
        "Horizon_t+": p["t"].astype("Int64"),
        "Actual": p["actual"].round(1),
        "LEGO_Forecast": p["lego"].round(1),
        "DIQ_Forecast": p["ours"].fillna(0).round(1),
    }).sort_values(["Category", "CS_BARCODE", "Week"]).reset_index(drop=True)
    detail["DIQ_vs_LEGO_AbsErr_gain"] = (
        (detail["LEGO_Forecast"] - detail["Actual"]).abs()
        - (detail["DIQ_Forecast"] - detail["Actual"]).abs()).round(1)  # +ve = DIQ beats LEGO on that cell

    # Summary per category (t45 + 13wk: 1-WAPE accuracy & bias, ours vs LEGO)
    def ab(s, col):
        a = s["actual"].values; f = s[col].fillna(0).values if col == "ours" else s[col].values
        t = a.sum()
        return (round(1 - np.abs(f - a).sum() / t, 3), round((f - a).sum() / t, 3)) if t else (np.nan, np.nan)
    rows = []
    for c in sorted(p["Category"].unique()):
        pc = p[p["Category"] == c]; p45 = pc[pc.t.isin([4, 5])]
        oa, ob = ab(p45, "ours"); la, lb = ab(p45, "lego")
        oa13, ob13 = ab(pc, "ours"); la13, lb13 = ab(pc, "lego")
        rows.append({"Category": c, "E1016_Actual_Units": int(pc["actual"].sum()),
                     "DIQ_t45_Acc": oa, "LEGO_t45_Acc": la, "t45_Winner": "DIQ" if oa > la else "LEGO",
                     "DIQ_t45_Bias": ob, "LEGO_t45_Bias": lb,
                     "DIQ_13wk_Acc": oa13, "LEGO_13wk_Acc": la13, "13wk_Winner": "DIQ" if oa13 > la13 else "LEGO"})
    summ = pd.DataFrame(rows)
    vol = summ["E1016_Actual_Units"]; w = vol / vol.sum()
    summ.loc["TOTAL"] = {"Category": "SALES-WEIGHTED", "E1016_Actual_Units": int(vol.sum()),
                         "DIQ_t45_Acc": round((summ["DIQ_t45_Acc"] * w).sum(), 3),
                         "LEGO_t45_Acc": round((summ["LEGO_t45_Acc"] * w).sum(), 3),
                         "t45_Winner": "", "DIQ_t45_Bias": "", "LEGO_t45_Bias": "",
                         "DIQ_13wk_Acc": round((summ["DIQ_13wk_Acc"] * w).sum(), 3),
                         "LEGO_13wk_Acc": round((summ["LEGO_13wk_Acc"] * w).sum(), 3), "13wk_Winner": ""}

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        summ.to_excel(xl, sheet_name="Summary", index=False)
        detail.to_excel(xl, sheet_name="Detail", index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                wmax = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(wmax + 2, 22)
    print(f"wrote {OUT}: Detail {len(detail)} rows ({detail['key'].nunique()} keys x weeks), "
          f"Summary {len(summ)-1} categories", flush=True)


if __name__ == "__main__":
    main()
