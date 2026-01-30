import pandas as pd
from pathlib import Path

files = [
    "352944080639365.csv",
    "358057080902248.csv",
    "866558034754119.csv",
    "866817034014258.csv",
    "867194030192184.csv",
]

# Files appear to have no header and ~18 columns; we assign temporary column names.
COLS = [
    "device_id",
    "device_model",
    "android_version",
    "battery_type",
    "battery_capacity_mAh",
    "timestamp_ms",
    "is_charging",
    "battery_level_pct",
    "foreground_app",
    "event_code",
    "battery_temp_C",
    "battery_voltage_mV",
    "battery_current_mA",
    "network_type",
    "screen_on",
    "col15",
    "col16",
    "col17",
]

rows = []

for f in files:
    p = Path(f)
    df = pd.read_csv(p, header=None, names=COLS, engine="c")

    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    dt = pd.to_datetime(df["timestamp_ms"], unit="ms", errors="coerce")

    dts = dt.sort_values().dropna().diff().dropna()
    step_median = float(dts.median().total_seconds()) if len(dts) else None
    step_p05 = float(dts.quantile(0.05).total_seconds()) if len(dts) else None
    step_p95 = float(dts.quantile(0.95).total_seconds()) if len(dts) else None

    key_cols = [
        "timestamp_ms",
        "battery_level_pct",
        "battery_temp_C",
        "battery_voltage_mV",
        "battery_current_mA",
        "network_type",
        "screen_on",
    ]
    nulls = df[key_cols].isna().sum()

    def rng(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return float(s.min()), float(s.max())

    lvl_min, lvl_max = rng("battery_level_pct")
    tmp_min, tmp_max = rng("battery_temp_C")
    v_min, v_max = rng("battery_voltage_mV")
    i_min, i_max = rng("battery_current_mA")

    rows.append({
        "file": f,
        "rows": int(len(df)),
        "start": str(dt.min()),
        "end": str(dt.max()),
        "step_median_s": step_median,
        "step_p05_s": step_p05,
        "step_p95_s": step_p95,
        "null_timestamp": int(nulls["timestamp_ms"]),
        "null_level": int(nulls["battery_level_pct"]),
        "null_temp": int(nulls["battery_temp_C"]),
        "null_voltage": int(nulls["battery_voltage_mV"]),
        "null_current": int(nulls["battery_current_mA"]),
        "null_network": int(nulls["network_type"]),
        "null_screen": int(nulls["screen_on"]),
        "level_min": lvl_min,
        "level_max": lvl_max,
        "temp_min": tmp_min,
        "temp_max": tmp_max,
        "voltage_min": v_min,
        "voltage_max": v_max,
        "current_min": i_min,
        "current_max": i_max,
        "network_counts": df["network_type"].value_counts(dropna=False).head(5).to_dict(),
        "screen_counts": df["screen_on"].value_counts(dropna=False).to_dict(),
        "charging_counts": df["is_charging"].value_counts(dropna=False).to_dict(),
    })

out = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 140)
print(
    out[[
        "file",
        "rows",
        "start",
        "end",
        "step_median_s",
        "step_p05_s",
        "step_p95_s",
        "null_timestamp",
        "null_level",
        "null_temp",
        "null_voltage",
        "null_current",
        "null_network",
        "null_screen",
        "level_min",
        "level_max",
        "temp_min",
        "temp_max",
        "voltage_min",
        "voltage_max",
        "current_min",
        "current_max",
    ]].to_string(index=False)
)

print("\n--- network top5 ---")
for r in rows:
    print(r["file"], r["network_counts"])

print("\n--- screen_on counts ---")
for r in rows:
    print(r["file"], r["screen_counts"])

print("\n--- is_charging counts ---")
for r in rows:
    print(r["file"], r["charging_counts"])
