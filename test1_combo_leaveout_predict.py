"""
Combo-based fuzzy matching prediction on test_1 (as requested).

Discrete "fuzzy" scenario labels (temperature treated as unknown/unavailable):
  - CPU load: 5 bins by percent: [0-20],[21-40],[41-60],[61-80],[81-100]
  - Network: 3 bins: none / mobile / wifi
  - Brightness: 4 bins: screen-off / low(0-33) / mid(34-66) / high(67-100)
    where brightness percent is derived from brightness_state in [0,1] -> [0,100],
    and screen_off if screen_on==0 or brightness_state<0.

Total combos: 5*4*3=60.

Protocol (leave-one-combo-out):
  For each target combo c:
    - Training data: all points NOT in combo c (i.e., other 59 combos), from discharge-only points.
    - Fit a model that generalizes across combos using only the 3 categorical factors:
        log(rate) = const + a_cpu[cpu_bin] + a_bri[bri_bin] + a_net[net_bin] + eps
      (main-effects log-linear; can predict a held-out combo by summing effects)
    - Test data: slices where combo == c for long enough duration.
    - Evaluate only final time-length (not curve):
        true_rate = drop_pct / dur_h
        pred_rate = exp(pred_log_rate)
        pred_dur_h = drop_pct / pred_rate
        acc = 1 - |pred_dur_h-true_dur_h|/true_dur_h

Cancel a combo if:
  - total test slice minutes < MIN_TEST_TOTAL_MIN, or
  - training points < MIN_TRAIN_POINTS after excluding the combo.

Outputs:
  processed/test1/combo_predict/
    combo_coverage.csv
    slices.csv
    metrics_by_slice.csv
    metrics_by_combo.csv
    summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
PANEL_PATH = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "combo_predict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# slice rules (for testing): rolling windows with dominance (fuzzy slices)
WIN_MIN = 120
STEP_MIN = 30
DOMINANCE = 0.80
MIN_TEST_TOTAL_MIN = 240

# training sufficiency
MIN_TRAIN_POINTS = 2000

# Use the full 1-min panel instead of the selected episodes subset.
USE_FULL_PANEL = True

# Discharge label parameters (full panel)
RATE_DIFF_MIN = 30   # for local rate label
MIN_RATE_PCT_PER_H = 0.5  # ignore near-zero due to SOC quantization noise


def _acc(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def cpu_bin_5(cpu_load: pd.Series) -> pd.Series:
    x = pd.to_numeric(cpu_load, errors="coerce") * 100.0
    # bins: 0..4
    b = pd.cut(
        x,
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=True,
        include_lowest=True,
    )
    return b.astype("Int64")


def net_bin_3(net_type_code: pd.Series) -> pd.Series:
    n = pd.to_numeric(net_type_code, errors="coerce")
    # map: 0/NaN -> none(0), 1 -> mobile(1), 2 -> wifi(2), else -> none
    out = pd.Series(np.zeros(len(n), dtype="int64"))
    out[(n == 1).to_numpy()] = 1
    out[(n == 2).to_numpy()] = 2
    return out.astype("Int64")


def bri_bin_4(screen_on: pd.Series, brightness_state: pd.Series) -> pd.Series:
    scr = pd.to_numeric(screen_on, errors="coerce").fillna(0).astype(int)
    b = pd.to_numeric(brightness_state, errors="coerce")
    # screen off if scr==0 or brightness_state<0 or NaN
    off = (scr == 0) | (b < 0) | (~np.isfinite(b))
    pct = np.clip(b.to_numpy(float) * 100.0, 0.0, 100.0)
    # low/mid/high based on percent; bin codes: off=0, low=1, mid=2, high=3
    bins = np.full(len(pct), 0, dtype="int64")
    bins[(~off).to_numpy() & (pct <= 33)] = 1
    bins[(~off).to_numpy() & (pct >= 34) & (pct <= 66)] = 2
    bins[(~off).to_numpy() & (pct >= 67)] = 3
    return pd.Series(bins).astype("Int64")


def make_combo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cpu5"] = cpu_bin_5(d["cpu_load"])
    d["net3"] = net_bin_3(d["net_type_code"])
    d["bri4"] = bri_bin_4(d["screen_on"], d["brightness_state"])
    d = d.dropna(subset=["cpu5", "net3", "bri4"]).copy()
    d["cpu5"] = d["cpu5"].astype(int)
    d["net3"] = d["net3"].astype(int)
    d["bri4"] = d["bri4"].astype(int)
    d["combo_id"] = d["cpu5"] * 12 + d["bri4"] * 3 + d["net3"]  # 0..59
    d["combo"] = d["cpu5"].astype(str) + "-" + d["bri4"].astype(str) + "-" + d["net3"].astype(str)
    return d


def build_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-window slices inside each discharge interval.
    A window is assigned to the dominant combo_id if its fraction >= DOMINANCE.
    """
    d = df.sort_values(["episode_id", "interval_id", "time"]).reset_index(drop=True)
    rows = []
    for (epi, itv), g in d.groupby(["episode_id", "interval_id"], sort=False):
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < WIN_MIN:
            continue
        for start in range(0, len(g) - WIN_MIN + 1, STEP_MIN):
            gg = g.iloc[start : start + WIN_MIN]
            vals, counts = np.unique(gg["combo_id"].to_numpy(int), return_counts=True)
            j = int(np.argmax(counts))
            dom = float(counts[j]) / float(len(gg))
            if dom < DOMINANCE:
                continue
            combo_id = int(vals[j])
            # map combo string/levels from first row that matches dominant combo
            gdom = gg[gg["combo_id"] == combo_id].iloc[0]
            soc0 = float(pd.to_numeric(gg["battery_level_pct"].iloc[0], errors="coerce"))
            soc1 = float(pd.to_numeric(gg["battery_level_pct"].iloc[-1], errors="coerce"))
            drop = float(soc0 - soc1)
            if drop <= 0:
                continue
            rows.append(
                {
                    "combo_id": combo_id,
                    "combo": str(gdom["combo"]),
                    "cpu5": int(gdom["cpu5"]),
                    "bri4": int(gdom["bri4"]),
                    "net3": int(gdom["net3"]),
                    "episode_id": int(epi),
                    "interval_id": int(itv),
                    "start_time": gg["time"].iloc[0],
                    "end_time": gg["time"].iloc[-1],
                    "minutes": int(len(gg)),
                    "dominance": dom,
                    "drop_pct": drop,
                }
            )
    return pd.DataFrame(rows)


def fit_main_effects_log_rate(train: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """
    Fit log(rate) = const + cpu5 dummies + bri4 dummies + net3 dummies.
    Baselines: cpu5=0, bri4=0(off), net3=0(none).
    Returns beta and names.
    """
    y = pd.to_numeric(train["rate_pct_per_h"], errors="coerce").to_numpy(float)
    m = np.isfinite(y) & (y > 1e-9)
    train = train.loc[m].copy()
    y = np.log(pd.to_numeric(train["rate_pct_per_h"], errors="coerce").to_numpy(float))

    n = len(train)
    const = np.ones(n)

    # dummies excluding baseline
    cpu = train["cpu5"].to_numpy(int)
    bri = train["bri4"].to_numpy(int)
    net = train["net3"].to_numpy(int)

    def dummies(vals: np.ndarray, levels: int) -> np.ndarray:
        out = np.zeros((n, levels - 1), dtype=float)
        for i in range(1, levels):
            out[:, i - 1] = (vals == i).astype(float)
        return out

    X = np.column_stack([const, dummies(cpu, 5), dummies(bri, 4), dummies(net, 3)])
    names = ["const"] + [f"cpu{i}" for i in range(1, 5)] + [f"bri{i}" for i in range(1, 4)] + [f"net{i}" for i in range(1, 3)]

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, names


def predict_log_rate(beta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    const = np.ones(n)
    cpu = df["cpu5"].to_numpy(int)
    bri = df["bri4"].to_numpy(int)
    net = df["net3"].to_numpy(int)

    def dummies(vals: np.ndarray, levels: int) -> np.ndarray:
        out = np.zeros((n, levels - 1), dtype=float)
        for i in range(1, levels):
            out[:, i - 1] = (vals == i).astype(float)
        return out

    X = np.column_stack([const, dummies(cpu, 5), dummies(bri, 4), dummies(net, 3)])
    return X @ beta


def main() -> None:
    if USE_FULL_PANEL:
        pts = pd.read_csv(PANEL_PATH, parse_dates=["time"]).sort_values(["time"]).copy()
        # mimic required columns
        pts["episode_id"] = 0
        pts["interval_id"] = 0
        # discharge mask via local rate label
        pts["battery_level_pct"] = pd.to_numeric(pts["battery_level_pct"], errors="coerce")
        pts["soc"] = pts["battery_level_pct"] / 100.0
        dt_h = float(RATE_DIFF_MIN) / 60.0
        pts["soc_lag"] = pts["soc"].shift(RATE_DIFF_MIN)
        pts["dsoc_dt"] = (pts["soc"] - pts["soc_lag"]) / dt_h
        pts["rate_pct_per_h"] = np.where(pts["dsoc_dt"] < 0, (-pts["dsoc_dt"]) * 100.0, np.nan)
        pts = pts.dropna(subset=["rate_pct_per_h"]).copy()
        pts = pts[pts["rate_pct_per_h"] >= MIN_RATE_PCT_PER_H].copy()
        pts = make_combo(pts)
    else:
        pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
        pts = pts[pts["is_discharge"] == 1].copy()
        pts = make_combo(pts)

    # coverage by combo (minutes)
    cov = pts.groupby(["combo_id", "combo", "cpu5", "bri4", "net3"]).size().reset_index(name="minutes")
    cov.to_csv(OUT_DIR / "combo_coverage.csv", index=False, encoding="utf-8")

    # build test slices (combo-dominant rolling windows)
    slices = build_slices(pts)
    slices.to_csv(OUT_DIR / "slices.csv", index=False, encoding="utf-8")

    # compute true rate for each slice (pct/h)
    if slices.empty:
        raise RuntimeError("No combo-dominant slices found. Consider lowering DOMINANCE or WIN_MIN.")
    slices["true_dur_h"] = slices["minutes"] / 60.0
    slices["true_rate_pct_per_h"] = slices["drop_pct"] / slices["true_dur_h"]

    # Evaluate each combo
    metrics_rows = []
    combo_rows = []
    for combo_id, g_slice in slices.groupby("combo_id"):
        combo_id = int(combo_id)
        test_total_min = int(g_slice["minutes"].sum())
        if test_total_min < MIN_TEST_TOTAL_MIN:
            continue

        train_pts = pts[pts["combo_id"] != combo_id].copy()
        if len(train_pts) < MIN_TRAIN_POINTS:
            continue

        # Training label: use 30-min differencing on points within each interval
        # Here we approximate rate at minute t by (SOC(t)-SOC(t-30))/0.5h inside each interval.
        DIFF_MIN = 30
        dt_h = float(DIFF_MIN) / 60.0
        train_pts["soc"] = pd.to_numeric(train_pts["battery_level_pct"], errors="coerce") / 100.0
        train_pts["soc_lag"] = train_pts.groupby(["episode_id", "interval_id"])["soc"].shift(DIFF_MIN)
        train_pts["dsoc_dt"] = (train_pts["soc"] - train_pts["soc_lag"]) / dt_h
        train_pts["rate_pct_per_h"] = np.where(train_pts["dsoc_dt"] < 0, (-train_pts["dsoc_dt"]) * 100.0, np.nan)
        train_pts = train_pts.dropna(subset=["rate_pct_per_h"]).copy()
        train_pts = train_pts[train_pts["rate_pct_per_h"] > 0].copy()
        if len(train_pts) < MIN_TRAIN_POINTS:
            continue

        beta, names = fit_main_effects_log_rate(train_pts)

        # Predict target combo rate from main effects (log-linear); same for all slices in this combo
        # Use the combo's categorical labels (cpu5,bri4,net3)
        c0 = g_slice.iloc[0]
        df_one = pd.DataFrame(
            {
                "cpu5": [int(c0["cpu5"])],
                "bri4": [int(c0["bri4"])],
                "net3": [int(c0["net3"])],
            }
        )
        pred_log = float(predict_log_rate(beta, df_one)[0])
        pred_rate = float(np.exp(pred_log))

        # slice-level metrics
        accs = []
        for _, r in g_slice.iterrows():
            true_dur = float(r["true_dur_h"])
            drop = float(r["drop_pct"])
            pred_dur = drop / pred_rate if pred_rate > 1e-9 else np.nan
            acc = _acc(pred_dur, true_dur)
            accs.append(acc)
            metrics_rows.append(
                {
                    "combo_id": combo_id,
                    "combo": str(r["combo"]),
                    "cpu5": int(r["cpu5"]),
                    "bri4": int(r["bri4"]),
                    "net3": int(r["net3"]),
                    "episode_id": int(r["episode_id"]),
                    "interval_id": int(r["interval_id"]),
                    "start_time": str(r["start_time"]),
                    "end_time": str(r["end_time"]),
                    "minutes": int(r["minutes"]),
                    "drop_pct": float(r["drop_pct"]),
                    "true_rate_pct_per_h": float(r["true_rate_pct_per_h"]),
                    "pred_rate_pct_per_h": pred_rate,
                    "true_dur_h": true_dur,
                    "pred_dur_h": pred_dur,
                    "acc_time": acc,
                    "n_train_points": int(len(train_pts)),
                }
            )

        combo_rows.append(
            {
                "combo_id": combo_id,
                "combo": str(c0["combo"]),
                "cpu5": int(c0["cpu5"]),
                "bri4": int(c0["bri4"]),
                "net3": int(c0["net3"]),
                "n_slices": int(len(g_slice)),
                "test_total_min": test_total_min,
                "pred_rate_pct_per_h": pred_rate,
                "acc_time_mean": float(np.nanmean(accs)) if len(accs) else np.nan,
                "acc_time_min": float(np.nanmin(accs)) if len(accs) else np.nan,
            }
        )

    mdf = pd.DataFrame(metrics_rows)
    cdf = pd.DataFrame(combo_rows)
    mdf.to_csv(OUT_DIR / "metrics_by_slice.csv", index=False, encoding="utf-8")
    cdf.to_csv(OUT_DIR / "metrics_by_combo.csv", index=False, encoding="utf-8")

    summary = {
        "n_combos_total": 60,
        "n_combos_with_slices": int(slices["combo_id"].nunique()),
        "n_combos_evaluated": int(cdf["combo_id"].nunique()) if len(cdf) else 0,
        "acc_mean_over_combos": float(cdf["acc_time_mean"].mean()) if len(cdf) else None,
        "acc_min_over_combos": float(cdf["acc_time_min"].min()) if len(cdf) else None,
        "WIN_MIN": WIN_MIN,
        "STEP_MIN": STEP_MIN,
        "DOMINANCE": DOMINANCE,
        "MIN_TEST_TOTAL_MIN": MIN_TEST_TOTAL_MIN,
        "MIN_TRAIN_POINTS": MIN_TRAIN_POINTS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote outputs to", OUT_DIR)
    print("[OK] summary:", summary)


if __name__ == "__main__":
    main()

