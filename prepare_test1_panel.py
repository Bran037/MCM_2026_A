"""
Prepare a unified time-indexed panel from MCM_2026_A/Database/test_1.

Goal: create a continuous 1-minute grid with the four modeling drivers:
  1) CPU load + CPU frequency
  2) Screen on/off + brightness
  3) Network usage (type + bytes per minute)
  4) Battery temperature

Outputs:
  processed/test1/test1_panel_1min.csv
  processed/test1/test1_panel_summary.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TEST1_DIR = BASE_DIR / "Database" / "test_1"
OUT_DIR = BASE_DIR / "processed" / "test1"


def _ensure_out_dirs() -> None:
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def _parse_cpu_hz(x: object) -> float:
    """Parse CpuHertz strings like '1.7 GHz' into Hz."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    # Examples observed: "1.7 GHz"
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([GMK]?Hz)\s*$", s, flags=re.IGNORECASE)
    if not m:
        # try plain number
        try:
            return float(s)
        except Exception:
            return np.nan
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "ghz":
        return val * 1e9
    if unit == "mhz":
        return val * 1e6
    if unit == "khz":
        return val * 1e3
    return val


def _to_dt_from_uuid_ms(uuid_ms: pd.Series) -> pd.Series:
    """
    Convert UUID epoch milliseconds to naive datetime (local-naive for portability).

    Note: pd.to_datetime(series, utc=True) returns a Series[datetime64[ns, UTC]],
    which must be converted via `.dt.tz_convert(None)` rather than `.tz_convert`.
    """
    s = pd.to_numeric(uuid_ms, errors="coerce")
    dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.tz_convert(None)
    # Fallback (DatetimeIndex)
    return pd.Series(dt.tz_convert(None))


def _infer_temp_c(series: pd.Series) -> pd.Series:
    """Battery_temperature in T4 appears to be tenths of °C (e.g., 392 => 39.2°C)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    # Heuristic: if typical values > 100, it's very likely 0.1°C units.
    q95 = float(s.dropna().quantile(0.95))
    if q95 > 100:
        return s / 10.0
    return s


def _normalize_brightness(br: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    s = pd.to_numeric(br, errors="coerce")
    meta = {"brightness_divisor": np.nan, "brightness_max": float(np.nan)}
    if s.dropna().empty:
        return s, meta
    mx = float(s.dropna().max())
    meta["brightness_max"] = mx
    # Common Android scales: 0..255 or 0..100
    if mx > 100:
        div = 255.0
    else:
        div = 100.0
    meta["brightness_divisor"] = div
    return (s / div).clip(0, 1), meta


@dataclass
class PanelMeta:
    user_id: str
    start: str
    end: str
    n_minutes: int
    missing_rate: Dict[str, float]
    brightness_meta: Dict[str, float]


def build_panel(user_id: Optional[str] = None, freq: str = "1min") -> Tuple[pd.DataFrame, PanelMeta]:
    _ensure_out_dirs()

    # --- T4: core battery + CPU + traffic ---
    t4_path = TEST1_DIR / "T4.csv"
    usecols_t4 = [
        "Userid",
        "UUID",
        "CpuHertz",
        "Total_CPU",
        "connectedWifi_SSID",
        "connectedWifi_Level",
        "Traffic_TotalWifiRxBytes",
        "Traffic_TotalWifiTxBytes",
        "Traffic_MobileRxBytes",
        "Traffic_MobileTxBytes",
        "Battery_level",
        "Battery_temperature",
        "Battery_voltage",
        "Battery_current_avg",
        "Battery_plugged",
    ]
    df4 = pd.read_csv(t4_path, usecols=usecols_t4, low_memory=False)
    df4 = df4.rename(columns={"Userid": "UserID"})

    if user_id is None:
        user_id = str(df4["UserID"].dropna().astype(str).mode().iloc[0])
    df4 = df4[df4["UserID"].astype(str) == str(user_id)].copy()

    df4["UUID"] = pd.to_numeric(df4["UUID"], errors="coerce")
    df4 = df4.dropna(subset=["UUID"]).sort_values("UUID")
    df4["time"] = _to_dt_from_uuid_ms(df4["UUID"])
    df4 = df4.set_index("time")

    df4["cpu_hz"] = df4["CpuHertz"].map(_parse_cpu_hz)
    df4["cpu_load"] = pd.to_numeric(df4["Total_CPU"], errors="coerce") / 100.0
    df4["battery_temp_C"] = _infer_temp_c(df4["Battery_temperature"])

    # Traffic columns exist, but for modeling we only keep 3-class net type (none / wi-fi / mobile).
    # We still parse these columns for potential debugging, but we do NOT export traffic rate features.
    for c in ["Traffic_TotalWifiRxBytes", "Traffic_TotalWifiTxBytes", "Traffic_MobileRxBytes", "Traffic_MobileTxBytes"]:
        df4[c] = pd.to_numeric(df4[c], errors="coerce")

    # wifi connection hint (RSSI-like)
    df4["wifi_level"] = pd.to_numeric(df4["connectedWifi_Level"], errors="coerce")
    df4["wifi_connected"] = df4["wifi_level"].notna() & (df4["wifi_level"] > -120)

    # Resample core signals
    df4_1m = pd.DataFrame(index=pd.date_range(df4.index.min().floor(freq), df4.index.max().ceil(freq), freq=freq))
    df4_1m.index.name = "time"
    df4_1m["UserID"] = str(user_id)

    df4_1m["battery_level_pct"] = df4["Battery_level"].astype("float").resample(freq).mean()
    df4_1m["battery_temp_C"] = df4["battery_temp_C"].resample(freq).mean()
    df4_1m["battery_voltage_mV"] = pd.to_numeric(df4["Battery_voltage"], errors="coerce").resample(freq).mean()
    df4_1m["battery_current_mA"] = pd.to_numeric(df4["Battery_current_avg"], errors="coerce").resample(freq).median()
    df4_1m["battery_plugged"] = pd.to_numeric(df4["Battery_plugged"], errors="coerce").resample(freq).max()

    df4_1m["cpu_load"] = df4["cpu_load"].resample(freq).mean()
    df4_1m["cpu_hz"] = df4["cpu_hz"].resample(freq).median()
    max_hz = float(np.nanmax(df4_1m["cpu_hz"].values)) if np.isfinite(np.nanmax(df4_1m["cpu_hz"].values)) else np.nan
    df4_1m["cpu_freq_norm"] = (df4_1m["cpu_hz"] / max_hz) if max_hz and np.isfinite(max_hz) else np.nan

    df4_1m["wifi_level"] = df4["wifi_level"].resample(freq).mean()
    df4_1m["wifi_connected"] = df4["wifi_connected"].resample(freq).max()

    # NOTE: intentionally no traffic-per-minute features in output (user request).

    # --- ScreenOn: event log → state on the 1-min grid ---
    scr_path = TEST1_DIR / "ScreenOn.csv"
    dfs = pd.read_csv(scr_path, usecols=["UserID", "UUID", "ScreenOn"], low_memory=False)
    dfs = dfs[dfs["UserID"].astype(str) == str(user_id)].copy()
    dfs["UUID"] = pd.to_numeric(dfs["UUID"], errors="coerce")
    dfs = dfs.dropna(subset=["UUID"]).sort_values("UUID")
    dfs["time"] = _to_dt_from_uuid_ms(dfs["UUID"])
    dfs = dfs.set_index("time")
    # normalize boolean strings
    dfs["ScreenOn"] = dfs["ScreenOn"].astype(str).str.lower().map({"true": 1, "false": 0})
    dfs = dfs.dropna(subset=["ScreenOn"])
    # asof join: last event before each minute
    scr_state = pd.merge_asof(
        df4_1m.reset_index().sort_values("time"),
        dfs[["ScreenOn"]].reset_index().sort_values("time"),
        on="time",
        direction="backward",
    )["ScreenOn"].fillna(0).astype(int).values
    df4_1m["screen_on"] = scr_state

    # --- Brightness: T1 Status_Brightness_settings on the same 1-min grid ---
    t1_path = TEST1_DIR / "T1.csv"
    df1 = pd.read_csv(
        t1_path,
        usecols=["UserID", "UUID", "Status_BrightnessMode", "Status_Brightness_settings"],
        low_memory=False,
    )
    df1 = df1[df1["UserID"].astype(str) == str(user_id)].copy()
    df1["UUID"] = pd.to_numeric(df1["UUID"], errors="coerce")
    df1 = df1.dropna(subset=["UUID"]).sort_values("UUID")
    df1["time"] = _to_dt_from_uuid_ms(df1["UUID"])
    df1 = df1.set_index("time")

    df1["brightness_raw"] = pd.to_numeric(df1["Status_Brightness_settings"], errors="coerce")
    brightness_norm, brightness_meta = _normalize_brightness(df1["brightness_raw"])
    df1["brightness_norm"] = brightness_norm
    df1["brightness_mode"] = df1["Status_BrightnessMode"].astype(str)

    br_merge = pd.merge_asof(
        df4_1m.reset_index().sort_values("time"),
        df1[["brightness_raw", "brightness_norm", "brightness_mode"]].reset_index().sort_values("time"),
        on="time",
        direction="backward",
    )
    df4_1m["brightness_raw"] = br_merge["brightness_raw"].values
    df4_1m["brightness_norm"] = br_merge["brightness_norm"].values
    df4_1m["brightness_mode"] = br_merge["brightness_mode"].values

    # --- Network type: prefer wifi_connected; fallback to T0 telephony networkType ---
    t0_path = TEST1_DIR / "T0.csv"
    df0 = pd.read_csv(t0_path, usecols=["UserId", "UUID", "Telephoney_networkType"], low_memory=False)
    df0 = df0.rename(columns={"UserId": "UserID"})
    df0 = df0[df0["UserID"].astype(str) == str(user_id)].copy()
    df0["UUID"] = pd.to_numeric(df0["UUID"], errors="coerce")
    df0 = df0.dropna(subset=["UUID"]).sort_values("UUID")
    df0["time"] = _to_dt_from_uuid_ms(df0["UUID"])
    df0 = df0.set_index("time")
    df0["telephony_net_type_code"] = pd.to_numeric(df0["Telephoney_networkType"], errors="coerce")

    net_merge = pd.merge_asof(
        df4_1m.reset_index().sort_values("time"),
        df0[["telephony_net_type_code"]].reset_index().sort_values("time"),
        on="time",
        direction="backward",
    )
    df4_1m["telephony_net_type_code"] = net_merge["telephony_net_type_code"].values

    def _net_type_row(wifi_connected: object, code: object) -> str:
        if pd.notna(wifi_connected) and int(bool(wifi_connected)) == 1:
            return "wi-fi"
        if pd.notna(code):
            # In this dataset codes like 10/13/15 are mobile radio types.
            if float(code) > 0:
                return "mobile"
        return "none"

    df4_1m["net_type"] = [
        _net_type_row(w, c) for w, c in zip(df4_1m["wifi_connected"].values, df4_1m["telephony_net_type_code"].values)
    ]

    # --- Light missing-value filling (dataset is very complete; fill should be conservative) ---
    # Continuous signals: short-gap time interpolation + short forward fill
    for c in ["battery_level_pct", "battery_temp_C", "cpu_load", "cpu_freq_norm", "brightness_norm"]:
        if c in df4_1m.columns:
            df4_1m[c] = pd.to_numeric(df4_1m[c], errors="coerce")
            df4_1m[c] = df4_1m[c].interpolate(method="time", limit=5, limit_direction="both")
            df4_1m[c] = df4_1m[c].ffill(limit=5).bfill(limit=5)

    # Discrete states: forward fill (very short) then default
    df4_1m["screen_on"] = pd.to_numeric(df4_1m["screen_on"], errors="coerce").ffill().fillna(0).astype(int)
    df4_1m["net_type"] = df4_1m["net_type"].astype(str).replace({"nan": "none", "None": "none"}).fillna("none")

    # Brightness encoding requested:
    # - screen off: -1
    # - screen on: brightness in [0,1]
    # This gives an explicit discontinuity at the on/off boundary.
    df4_1m["brightness_state"] = np.where(df4_1m["screen_on"].values == 1, df4_1m["brightness_norm"].values, -1.0)

    # Net type code (requested for plotting): none=0, mobile=1, wi-fi=2
    net_map = {"none": 0, "mobile": 1, "wi-fi": 2}
    df4_1m["net_type_code"] = df4_1m["net_type"].map(net_map).astype("float")

    # --- Final cleanup / ordering ---
    df_out = df4_1m.copy()
    cols = [
        "UserID",
        "battery_level_pct",
        "battery_temp_C",
        "cpu_load",
        "cpu_freq_norm",
        "screen_on",
        "brightness_state",
        "net_type",
        "net_type_code",
        # keep a few diagnostics columns (low cost, helpful if something looks off)
        "brightness_mode",
        "wifi_connected",
        "telephony_net_type_code",
    ]
    df_out = df_out[cols]

    miss = {c: float(df_out[c].isna().mean()) for c in df_out.columns}
    meta = PanelMeta(
        user_id=str(user_id),
        start=str(df_out.index.min()),
        end=str(df_out.index.max()),
        n_minutes=int(len(df_out)),
        missing_rate=miss,
        brightness_meta=brightness_meta,
    )
    return df_out, meta


def main() -> None:
    df, meta = build_panel()
    out_csv = OUT_DIR / "test1_panel_1min.csv"
    df.to_csv(out_csv, index=True)

    out_meta = OUT_DIR / "test1_panel_summary.json"
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_meta}")


if __name__ == "__main__":
    main()

