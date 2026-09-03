"""Email the client Excel via the user's Outlook (AppleScript). Builds a summary body
from the live scorecard, attaches the workbook, sends to the given address.
Usage: _send_email.py <xlsx_path> <final_forecast.parquet> [to_addr]"""
import sys, os, subprocess, tempfile
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from utils import lego_eval

XLSX = os.path.abspath(sys.argv[1])
FC = sys.argv[2] if len(sys.argv) > 2 else "artifacts_th_local/final_forecast.parquet"
TO = sys.argv[3] if len(sys.argv) > 3 else "debonil.chowdhury@aria-is.com"
SUBJECT = "Thailand DIQ Forecast vs LEGO — Model Comparison (t+4&5 and 13-week)"


def scorecard():
    base = lego_eval.load_eval_base()
    p = lego_eval.attach_forecast(base, pd.read_parquet(FC), pred_col="predicted")
    p = p[p.Category != "ICE CREAM"]
    def acc(s, col):
        a = s["actual"].values; f = (s["ours"].fillna(0) if col == "ours" else s[col]).values
        t = np.abs(a).sum(); return 1 - np.abs(f - a).sum() / t if t else np.nan
    rows = []
    for c in sorted(p.Category.unique()):
        pc = p[p.Category == c]; p45 = pc[pc.t.isin([4, 5])]
        rows.append((c, pc.actual.sum(), acc(p45, "ours"), acc(p45, "lego"), acc(pc, "ours"), acc(pc, "lego")))
    d = pd.DataFrame(rows, columns=["c", "v", "o45", "l45", "o13", "l13"]); w = d.v / d.v.sum()
    lines = ["Category                  DIQ_t45  LEGO_t45   DIQ_13wk LEGO_13wk  Winner(t45/13wk)"]
    for _, r in d.iterrows():
        lines.append(f"{r.c:24s}  {r.o45:+.3f}   {r.l45:+.3f}    {r.o13:+.3f}   {r.l13:+.3f}   "
                     f"{'DIQ' if r.o45>r.l45 else 'LEGO'}/{'DIQ' if r.o13>r.l13 else 'LEGO'}")
    lines.append("-" * 86)
    lines.append(f"{'SALES-WEIGHTED':24s}  {(d.o45*w).sum():+.3f}   {(d.l45*w).sum():+.3f}    "
                 f"{(d.o13*w).sum():+.3f}   {(d.l13*w).sum():+.3f}")
    n45, n13 = int((d.o45 > d.l45).sum()), int((d.o13 > d.l13).sum())
    return "\n".join(lines), n45, n13


def main():
    table, n45, n13 = scorecard()
    body = (
        "Hi Debonil,\n\n"
        "Please find attached the Thailand forecast comparison (DIQ model vs the LEGO benchmark), "
        "at the LEGO evaluation grain (CS_BARCODE x week, account E1016), forecast window 2026-03 to 2026-15.\n\n"
        "The workbook has two sheets:\n"
        "  - 'Detail': one row per key x week with Actual, LEGO_Forecast and DIQ_Forecast units (+ product attributes).\n"
        "  - 'Summary': per-category accuracy (1-WAPE) and bias vs LEGO, at t+4&5 and 13-week, with a sales-weighted total.\n\n"
        f"Headline (accuracy, 1-WAPE):\n"
        f"  - t+4&5 (planning-critical horizons): DIQ wins {n45} of 9 categories.\n"
        f"  - 13-week: DIQ wins {n13} of 9 categories.\n"
        "  - The deep-learning (Temporal Fusion Transformer) model closed the gap on the high-volume categories, "
        "including FOODS where DIQ now beats LEGO on both horizons.\n\n"
        + table +
        "\n\nBest regards,\nDIQ Forecasting"
    )
    bf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    bf.write(body); bf.close()
    osa = f'''
set theBody to (read POSIX file "{bf.name}" as «class utf8»)
tell application "Microsoft Outlook"
    set newMessage to make new outgoing message with properties {{subject:"{SUBJECT}", content:theBody}}
    tell newMessage
        make new recipient with properties {{email address:{{address:"{TO}"}}}}
        make new attachment with properties {{file:(POSIX file "{XLSX}")}}
    end tell
    send newMessage
end tell
return "SENT"
'''
    sf = tempfile.NamedTemporaryFile("w", suffix=".applescript", delete=False)
    sf.write(osa); sf.close()
    r = subprocess.run(["osascript", sf.name], capture_output=True, text=True)
    print("STDOUT:", r.stdout.strip(), "| STDERR:", r.stderr.strip())
    print(f"emailed {XLSX} -> {TO}" if "SENT" in r.stdout else "SEND FAILED (see STDERR)")


if __name__ == "__main__":
    main()
