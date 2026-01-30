from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_DIR = Path("processed")
OUT_DIR = Path("processed") / "discharge"
FIG_DIR = Path("figures") / "discharge_fits"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FILES = sorted(IN_DIR.glob("*_clean.csv"))

# Resample + smoothing
RESAMPLE_FREQ = "1min"
SMOOTH_WINDOW_MIN = 11  # odd recommended

# Segment rules (discharge only)
GAP_BREAK_MIN = 15          # if missing gap larger than this (minutes), start new segment
SOC_INCREASE_TOL = 0.004    # allow tiny upward noise (0.4% SOC)
MIN_SEG_LEN_MIN = 30        # minimum duration for a segment
MIN_SEG_DROP_PCT = 5.0      # minimum SOC drop in percentage points over a segment
INTERP_LIMIT_MIN = 10       # fill short gaps within segment (minutes)


@dataclass(frozen=True)
class SegmentPick:
    device_id: str
    segment_id: int
    start_utc: str
    end_utc: str
    minutes: int
    soc_drop_pct: float


def _to_bool(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    return x.map({"true": True, "false": False}).astype("boolean")


def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # numeric
    for c in ["battery_level_pct", "battery_temp_C", "battery_voltage_mV", "battery_current_mA", "col09_unknown"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # bool-like
    if d["is_charging"].dtype.name != "boolean":
        d["is_charging"] = _to_bool(d["is_charging"])
    if d["screen_on"].dtype.name != "boolean":
        d["screen_on"] = _to_bool(d["screen_on"])

    agg = {
        "battery_level_pct": "median",
        "battery_temp_C": "median",
        "battery_voltage_mV": "median",
        "battery_current_mA": "median",
        "col09_unknown": "median",
        "is_charging": "max",
        "screen_on": "max",
        "network_type": lambda x: x.dropna().iloc[-1] if len(x.dropna()) else pd.NA,
    }
    out = d.resample(RESAMPLE_FREQ).agg(agg)

    # gentle smoothing to reduce stair-steps
    w = int(SMOOTH_WINDOW_MIN)
    if w >= 3:
        for c in ["battery_level_pct", "battery_temp_C", "battery_voltage_mV", "col09_unknown"]:
            if c in out.columns:
                out[c] = out[c].rolling(window=w, center=True, min_periods=max(1, w // 3)).mean()
        if "battery_current_mA" in out.columns:
            out["battery_current_mA"] = out["battery_current_mA"].rolling(
                window=w, center=True, min_periods=max(1, w // 3)
            ).median()

    return out


def build_discharge_segments(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    Return discharge-only minute data with a segment_id, where each segment is:
    - is_charging == False
    - time-contiguous (short gaps allowed but large gaps break)
    - SOC is non-increasing up to a small tolerance (up-jumps break)
    Within each segment, short missing runs can be interpolated for plotting/fitting.
    """
    d = df1m.copy()
    d = d[d["is_charging"] == False].copy()  # noqa: E712 (boolean is pandas Boolean)

    # Keep only minutes with any SOC observation (we will reindex per segment later)
    d = d.dropna(subset=["battery_level_pct"])
    d["soc"] = d["battery_level_pct"] / 100.0

    # Break flags
    dt_min = d.index.to_series().diff().dt.total_seconds().div(60.0)
    soc_inc = d["soc"].diff()

    gap_break = (dt_min > GAP_BREAK_MIN).fillna(True)
    jump_break = (soc_inc > SOC_INCREASE_TOL).fillna(True)

    # Segment id by cumulative breaks
    seg_id = (gap_break | jump_break).cumsum().astype(int) - 1
    d["segment_id"] = seg_id

    # Interpolate within each segment (limited) to "fill jumps" caused by short missing spans
    out_parts = []
    for sid, g in d.groupby("segment_id", sort=True):
        g = g.sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=RESAMPLE_FREQ, tz="UTC")
        g2 = g.reindex(full_idx)
        g2["segment_id"] = sid
        # propagate discrete states conservatively
        for c in ["screen_on", "network_type", "is_charging"]:
            if c in g2.columns:
                g2[c] = g2[c].ffill(limit=INTERP_LIMIT_MIN)
        # interpolate numeric within short gaps
        for c in ["battery_level_pct", "soc", "battery_temp_C", "battery_voltage_mV", "battery_current_mA", "col09_unknown"]:
            if c in g2.columns:
                g2[c] = g2[c].interpolate(method="time", limit=INTERP_LIMIT_MIN)
        out_parts.append(g2)

    out = pd.concat(out_parts).sort_index()
    return out


def summarize_segments(dseg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in dseg.groupby("segment_id", sort=True):
        g = g.dropna(subset=["soc"]).copy()
        if len(g) < 2:
            continue
        start = g.index.min()
        end = g.index.max()
        minutes = int((end - start).total_seconds() // 60) + 1
        soc_drop_pct = float((g["soc"].iloc[0] - g["soc"].iloc[-1]) * 100.0)

        # Segment-level stats
        rows.append(
            {
                "segment_id": int(sid),
                "start_utc": str(start),
                "end_utc": str(end),
                "minutes": minutes,
                "soc_start_pct": float(g["soc"].iloc[0] * 100.0),
                "soc_end_pct": float(g["soc"].iloc[-1] * 100.0),
                "soc_drop_pct": soc_drop_pct,
                "temp_mean_C": float(np.nanmean(g["battery_temp_C"])) if "battery_temp_C" in g.columns else np.nan,
                "screen_on_ratio": float(np.nanmean(g["screen_on"].astype(float))) if "screen_on" in g.columns else np.nan,
                "net_mode": (
                    g["network_type"].astype("string").str.lower().mode().iloc[0]
                    if "network_type" in g.columns and g["network_type"].notna().any()
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["soc_drop_pct", "minutes"], ascending=False)


def pick_two_segments(seg_summary: pd.DataFrame) -> list[int]:
    if seg_summary.empty:
        return []
    filt = seg_summary[
        (seg_summary["minutes"] >= MIN_SEG_LEN_MIN) & (seg_summary["soc_drop_pct"] >= MIN_SEG_DROP_PCT)
    ].copy()
    if filt.empty:
        # fall back: take top-2 by drop
        return seg_summary.head(2)["segment_id"].astype(int).tolist()
    return filt.head(2)["segment_id"].astype(int).tolist()


def fit_segment_line(g: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Fit SOC% ~ a + b*t_minutes by least squares. Return (a,b) and R^2."""
    g = g.dropna(subset=["battery_level_pct"]).copy()
    if len(g) < 3:
        return np.array([np.nan, np.nan]), np.nan
    t0 = g.index.min()
    x = (g.index - t0).total_seconds().astype(float) / 60.0  # minutes
    y = g["battery_level_pct"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(g)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def plot_segment_fit(device_id: str, sid: int, g: pd.DataFrame) -> Path:
    beta, r2 = fit_segment_line(g)
    t0 = g.index.min()
    x = (g.index - t0).total_seconds().astype(float) / 60.0  # minutes
    yhat = beta[0] + beta[1] * x

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(g.index, g["battery_level_pct"], lw=1.2, label="SOC(%) (filled/smoothed)")
    ax.plot(g.index, yhat, lw=2.0, label=f"linear fit, slope={beta[1]:.3f}%/min, R2={r2:.3f}")
    ax.set_title(f"Device {device_id} segment {sid}: discharge-only SOC fit")
    ax.set_ylabel("SOC(%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = FIG_DIR / f"{device_id}_seg{sid}_soc_fit.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    all_picks: list[SegmentPick] = []
    device_meta_rows = []

    for f in FILES:
        df = pd.read_csv(f)
        device_id = str(df["device_id"].iloc[0]) if "device_id" in df.columns else f.stem.replace("_clean", "")

        # device meta (for "can we fit together?")
        device_meta_rows.append(
            {
                "device_id": device_id,
                "device_model": str(df["device_model"].iloc[0]) if "device_model" in df.columns else pd.NA,
                "battery_chemistry": str(df["battery_chemistry"].iloc[0]) if "battery_chemistry" in df.columns else pd.NA,
                "battery_capacity_mAh": float(pd.to_numeric(df["battery_capacity_mAh"].iloc[0], errors="coerce"))
                if "battery_capacity_mAh" in df.columns
                else np.nan,
                "android_version": str(df["android_version"].iloc[0]) if "android_version" in df.columns else pd.NA,
            }
        )

        df1m = resample_1min(df)
        dseg = build_discharge_segments(df1m)

        # Save discharge minute data
        out_csv = OUT_DIR / f"{device_id}_discharge_1min.csv"
        keep_cols = [
            "segment_id",
            "battery_level_pct",
            "soc",
            "battery_temp_C",
            "battery_voltage_mV",
            "battery_current_mA",
            "screen_on",
            "network_type",
            "col09_unknown",
        ]
        cols = [c for c in keep_cols if c in dseg.columns]
        out_df = dseg[cols].copy()
        out_df.index.name = "timestamp"
        out_df.to_csv(out_csv, encoding="utf-8")

        seg_sum = summarize_segments(dseg)
        seg_sum_path = OUT_DIR / f"{device_id}_segments_summary.csv"
        seg_sum.to_csv(seg_sum_path, index=False, encoding="utf-8")

        picks = pick_two_segments(seg_sum)
        for sid in picks:
            g = dseg[dseg["segment_id"] == sid].copy()
            g = g.dropna(subset=["battery_level_pct"])
            if g.empty:
                continue
            start = g.index.min()
            end = g.index.max()
            minutes = int((end - start).total_seconds() // 60) + 1
            soc_drop_pct = float(g["battery_level_pct"].iloc[0] - g["battery_level_pct"].iloc[-1])
            all_picks.append(
                SegmentPick(
                    device_id=device_id,
                    segment_id=int(sid),
                    start_utc=str(start),
                    end_utc=str(end),
                    minutes=minutes,
                    soc_drop_pct=soc_drop_pct,
                )
            )
            out_plot = plot_segment_fit(device_id, int(sid), g)
            print("wrote", out_plot)

        print("wrote", out_csv)
        print("wrote", seg_sum_path)

    # overall pick summary
    if all_picks:
        pd.DataFrame([p.__dict__ for p in all_picks]).to_csv(
            OUT_DIR / "chosen_two_segments_per_device.csv", index=False, encoding="utf-8"
        )
        print("wrote", OUT_DIR / "chosen_two_segments_per_device.csv")

    # device meta summary
    pd.DataFrame(device_meta_rows).to_csv(OUT_DIR / "device_meta_summary.csv", index=False, encoding="utf-8")
    print("wrote", OUT_DIR / "device_meta_summary.csv")


if __name__ == "__main__":
    main()

