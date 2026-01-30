import csv
from pathlib import Path
import pandas as pd

paths = [
    Path("352944080639365.csv"),
    Path("358057080902248.csv"),
    Path("866558034754119.csv"),
    Path("866817034014258.csv"),
    Path("867194030192184.csv"),
]


def peek_rows(p: Path, n=5):
    rows = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            rows.append(row)
            if i + 1 >= n:
                break
    return rows


def summarize_file(p: Path, scan_rows=200000):
    first_rows = peek_rows(p, n=5)
    max_cols = max(len(r) for r in first_rows)

    # 鍙鍙栧墠 scan_rows 琛岋紝閬垮厤鍐呭瓨鐖嗙偢
    cols_idx = list(range(max_cols))
    df = pd.read_csv(p, header=None, usecols=cols_idx, nrows=scan_rows, engine="c", dtype=str)

    # 缁忛獙锛氱 6 鍒楋紙绱㈠紩 5锛夋槸 epoch ms
    ts_col = 5 if max_cols > 5 else None
    dt_min = dt_max = None
    step_stats = None

    if ts_col is not None:
        ts = pd.to_numeric(df[ts_col], errors="coerce")
        dt = pd.to_datetime(ts, unit="ms", errors="coerce")
        dt_min = dt.min()
        dt_max = dt.max()

        dts = dt.sort_values().dropna().diff().dropna()
        if len(dts):
            step_stats = {
                "median_s": float(dts.median().total_seconds()),
                "p05_s": float(dts.quantile(0.05).total_seconds()),
                "p95_s": float(dts.quantile(0.95).total_seconds()),
            }

    return {
        "file": p.name,
        "approx_cols": max_cols,
        "nrows_scanned": len(df),
        "dt_min": str(dt_min) if dt_min is not None else None,
        "dt_max": str(dt_max) if dt_max is not None else None,
        "step": step_stats,
        "sample_row_10": " | ".join(first_rows[0][:min(10, len(first_rows[0]))]),
    }


for p in paths:
    s = summarize_file(p)
    print("\n==", s["file"], "==")
    print("approx_cols:", s["approx_cols"], "scanned:", s["nrows_scanned"])
    print("dt_min:", s["dt_min"], "dt_max:", s["dt_max"])
    print("step_stats:", s["step"])
    print("sample_row(first 10 fields):", s["sample_row_10"])
