import pandas as pd
from pathlib import Path

files = [
    "COMED_hourly.csv","DEOK_hourly.csv","EKPC_hourly.csv","FE_hourly.csv","NI_hourly.csv","PJM_Load_hourly.csv",
]

rows = []
for f in files:
    df = pd.read_csv(f)
    dt_col = df.columns[0]
    val_col = df.columns[1]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    n = len(df)
    dt_null = int(df[dt_col].isna().sum())
    val_null = int(df[val_col].isna().sum())
    dt_min = df[dt_col].min()
    dt_max = df[dt_col].max()

    s = df.sort_values(dt_col)[dt_col].dropna()
    deltas = s.diff().dropna()
    one_hour_ratio = float((deltas == pd.Timedelta(hours=1)).mean()) if len(deltas) else float("nan")
    dup_rate = float(s.duplicated().mean()) if len(s) else float("nan")

    rows.append({
        "file": f,
        "rows": n,
        "start": str(dt_min),
        "end": str(dt_max),
        "dt_null": dt_null,
        "val_null": val_null,
        "one_hour_ratio": round(one_hour_ratio, 6) if one_hour_ratio == one_hour_ratio else None,
        "dup_rate": round(dup_rate, 6) if dup_rate == dup_rate else None,
        "min_MW": float(df[val_col].min()),
        "max_MW": float(df[val_col].max()),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
