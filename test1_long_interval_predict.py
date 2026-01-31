"""
Long-segment (interval-level) fuzzy prediction on test_1.

This addresses the situation where perfectly "constant combo" slices are rare.
We instead evaluate on full discharge intervals (hundreds to 1000+ minutes),
which naturally contain multiple CPU/bin/screen/network combinations.

Protocol:
  - Build per-minute local discharge rate labels via 30-min differencing (pct/h).
  - Train fuzzy KNN on all intervals except the target interval.
  - Predict per-minute rate on target interval, take mean rate, and convert to a final time error:
      pred_duration = observed_drop / pred_rate
      accuracy = 1 - |pred_duration-true_duration|/true_duration

Outputs:
  processed/test1/fuzzy_predict_long/
    metrics_by_interval.csv
    best_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "fuzzy_predict_long"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_MIN = 30

K_NEIGHBORS_LIST = [120, 250, 400]
SIGMA_LIST = [0.8, 1.2, 1.6]
LAMBDA_SCR = 0.7
LAMBDA_NET = 0.7


def _accuracy(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def _add_local_rate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["battery_level_pct"] = pd.to_numeric(d["battery_level_pct"], errors="coerce")
    d["soc"] = d["battery_level_pct"] / 100.0
    dt_h = float(DIFF_MIN) / 60.0
    d["soc_lag"] = d.groupby(["episode_id", "interval_id"])["soc"].shift(DIFF_MIN)
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h
    d["rate_pct_per_h"] = np.where(d["dsoc_dt"] < 0, (-d["dsoc_dt"]) * 100.0, np.nan)
    return d


def _build_X(d: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    cpu = pd.to_numeric(d["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(d["cpu_freq_norm"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(d["battery_temp_C"], errors="coerce").to_numpy(float)
    br = pd.to_numeric(d["brightness_state"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(d["screen_on"], errors="coerce").fillna(0).to_numpy(int)
    net = pd.to_numeric(d["net_type_code"], errors="coerce").fillna(-1).to_numpy(int)
    y = pd.to_numeric(d["rate_pct_per_h"], errors="coerce").to_numpy(float)

    br2 = np.clip(br, 0.0, 1.0)
    br2 = np.where(scr == 0, 0.0, br2)

    X = np.column_stack([cpu, f, T, br2])
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    disc = {"scr": scr[m], "net": net[m]}

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-12
    Xz = (X - mu) / sd
    return Xz, disc, y


def _knn_predict(train_Xz: np.ndarray, train_disc: Dict[str, np.ndarray], train_y: np.ndarray, test_Xz: np.ndarray, test_disc: Dict[str, np.ndarray], *, k_neighbors: int, sigma: float) -> np.ndarray:
    yhat = np.full(len(test_Xz), np.nan, dtype=float)
    for i in range(len(test_Xz)):
        dx = train_Xz - test_Xz[i]
        d2 = np.sum(dx * dx, axis=1)
        d2 = d2 + LAMBDA_SCR * (train_disc["scr"] != test_disc["scr"][i]).astype(float)
        d2 = d2 + LAMBDA_NET * (train_disc["net"] != test_disc["net"][i]).astype(float)
        k = min(k_neighbors, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        dd = d2[idx]
        w = np.exp(-0.5 * dd / (sigma**2))
        s = float(np.sum(w))
        if s <= 1e-12:
            continue
        yhat[i] = float(np.sum(w * train_y[idx]) / s)
    return yhat


def main() -> None:
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()
    pts = _add_local_rate(pts)

    # unique intervals (episode_id, interval_id)
    intervals = sorted(set(zip(pts["episode_id"].astype(int), pts["interval_id"].astype(int))))
    best = None
    best_key = (-np.inf, -np.inf)
    best_rows = None

    for kn in K_NEIGHBORS_LIST:
        for sg in SIGMA_LIST:
            rows = []
            for epi, itv in intervals:
                # train on all other intervals, require rate label
                train = pts[(pts["episode_id"].astype(int) != epi) | (pts["interval_id"].astype(int) != itv)].dropna(subset=["rate_pct_per_h"]).copy()
                test = pts[(pts["episode_id"].astype(int) == epi) & (pts["interval_id"].astype(int) == itv)].dropna(subset=["rate_pct_per_h"]).copy()
                if len(train) < 1500 or len(test) < 200:
                    continue

                Xtr, dtr, ytr = _build_X(train)
                Xte, dte, yte = _build_X(test)
                if len(Xtr) < 1000 or len(Xte) < 100:
                    continue

                # true duration/drop from full interval (not rate-filtered)
                full = pts[(pts["episode_id"].astype(int) == epi) & (pts["interval_id"].astype(int) == itv)].sort_values("time")
                soc0 = float(pd.to_numeric(full["battery_level_pct"].iloc[0], errors="coerce"))
                soc1 = float(pd.to_numeric(full["battery_level_pct"].iloc[-1], errors="coerce"))
                drop = float(soc0 - soc1)
                dur_h = float(len(full)) / 60.0
                if drop <= 0 or dur_h <= 0:
                    continue

                yhat = _knn_predict(Xtr, dtr, ytr, Xte, dte, k_neighbors=int(kn), sigma=float(sg))
                pred_rate = float(np.nanmean(yhat))
                pred_dur_h = drop / pred_rate if pred_rate > 1e-9 else np.nan
                acc = _accuracy(pred_dur_h, dur_h)

                rows.append(
                    {
                        "k_neighbors": int(kn),
                        "sigma": float(sg),
                        "episode_id": int(epi),
                        "interval_id": int(itv),
                        "minutes": int(len(full)),
                        "drop_pct": drop,
                        "true_dur_h": dur_h,
                        "pred_rate_pct_per_h": pred_rate,
                        "pred_dur_h": pred_dur_h,
                        "acc_time": acc,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                    }
                )

            mdf = pd.DataFrame(rows)
            if mdf.empty:
                continue
            key = (float((mdf["acc_time"] >= 0.90).mean()), float(mdf["acc_time"].median()))
            if key > best_key:
                best_key = key
                best = {"k_neighbors": int(kn), "sigma": float(sg), "n_intervals": int(len(mdf)), "acc_ge_0.90_frac": float(key[0]), "acc_median": float(key[1])}
                best_rows = mdf

    if best is None or best_rows is None:
        raise RuntimeError("No valid interval-level results.")

    best_rows.to_csv(OUT_DIR / "metrics_by_interval.csv", index=False, encoding="utf-8")
    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] best:", best)
    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

