"""
CPU-bin leave-one-out fuzzy prediction on test_1.

Interpretation (closer to user's intent):
  - Bin CPU load into K=3..5 levels.
  - Pick multiple test slices where CPU bin is dominant (>=DOMINANCE_CPU) over a window.
  - For a target CPU bin b:
      train on ALL OTHER bins (cpu_bin != b)
      predict discharge speed for b (final time difference only)

We still use fuzzy matching (kernel KNN) on continuous features:
  cpu_load, cpu_freq_norm, temp, brightness, plus penalties for screen_on/net mismatches.

Outputs:
  processed/test1/fuzzy_predict_cpu_leaveout/
    summary_by_k.csv
    metrics_by_slice.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "fuzzy_predict_cpu_leaveout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN_MIN = 120
STEP_MIN = 30
DOMINANCE_CPU = 0.80

DIFF_MIN = 30  # for local rate labels (pct/h)

K_LIST = [3, 4, 5]
K_NEIGHBORS_LIST = [120, 250, 400]
SIGMA_LIST = [0.8, 1.2, 1.6]
LAMBDA_SCR = 0.7
LAMBDA_NET = 0.7

MIN_TRAIN = 600
MIN_TEST = 60


def _accuracy(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def _cpu_bins(series: pd.Series, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(x, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    bins = np.digitize(pd.to_numeric(series, errors="coerce").to_numpy(float), edges[1:-1], right=False)
    return bins.astype(int), edges.astype(float)


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


def _cpu_dominant_slices(df: pd.DataFrame, cpu_bins: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["cpu_bin"] = cpu_bins
    d = d.sort_values(["episode_id", "interval_id", "time"]).reset_index(drop=True)
    out = []
    for (epi, itv), g in d.groupby(["episode_id", "interval_id"], sort=False):
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < WIN_MIN:
            continue
        for start in range(0, len(g) - WIN_MIN + 1, STEP_MIN):
            gg = g.iloc[start : start + WIN_MIN]
            vals, counts = np.unique(gg["cpu_bin"].to_numpy(int), return_counts=True)
            j = int(np.argmax(counts))
            dom = float(counts[j]) / float(len(gg))
            if dom < DOMINANCE_CPU:
                continue
            out.append(
                {
                    "episode_id": int(epi),
                    "interval_id": int(itv),
                    "start_time": gg["time"].iloc[0],
                    "end_time": gg["time"].iloc[-1],
                    "minutes": int(len(gg)),
                    "cpu_bin": int(vals[j]),
                    "dominance": dom,
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()
    pts = _add_local_rate(pts)

    all_metrics = []
    all_summary = []

    for k in K_LIST:
        cpu_bins, edges = _cpu_bins(pts["cpu_load"], k)
        slices = _cpu_dominant_slices(pts, cpu_bins)
        if slices.empty:
            continue

        for kn, sg in itertools.product(K_NEIGHBORS_LIST, SIGMA_LIST):
            # Build training dataset rows where local rate exists
            d = pts.copy()
            d["cpu_bin"] = cpu_bins
            d = d.dropna(subset=["rate_pct_per_h"]).copy()
            Xz, disc, y = _build_X(d)
            d = d.reset_index(drop=True)
            assert len(d) == len(y)

            rows = []
            for _, s in slices.iterrows():
                b = int(s["cpu_bin"])
                st = pd.to_datetime(s["start_time"])
                et = pd.to_datetime(s["end_time"])

                test_mask_full = (pd.to_datetime(d["time"]) >= st) & (pd.to_datetime(d["time"]) <= et)
                test_mask_full = test_mask_full.to_numpy()
                if int(test_mask_full.sum()) < MIN_TEST:
                    continue

                train_mask = (d["cpu_bin"] != b).to_numpy()
                if int(train_mask.sum()) < MIN_TRAIN:
                    continue

                # true rate from original pts slice
                gtrue = pts[(pd.to_datetime(pts["time"]) >= st) & (pd.to_datetime(pts["time"]) <= et)].sort_values("time")
                soc0 = float(pd.to_numeric(gtrue["battery_level_pct"].iloc[0], errors="coerce"))
                soc1 = float(pd.to_numeric(gtrue["battery_level_pct"].iloc[-1], errors="coerce"))
                drop = float(soc0 - soc1)
                dur_h = float(len(gtrue)) / 60.0
                if drop <= 0 or dur_h <= 0:
                    continue
                true_rate = drop / dur_h

                yhat = _knn_predict(
                    train_Xz=Xz[train_mask],
                    train_disc={"scr": disc["scr"][train_mask], "net": disc["net"][train_mask]},
                    train_y=y[train_mask],
                    test_Xz=Xz[test_mask_full],
                    test_disc={"scr": disc["scr"][test_mask_full], "net": disc["net"][test_mask_full]},
                    k_neighbors=int(kn),
                    sigma=float(sg),
                )
                pred_rate = float(np.nanmean(yhat))
                pred_dur_h = drop / pred_rate if pred_rate > 1e-9 else np.nan
                acc_time = _accuracy(pred_dur_h, dur_h)

                rows.append(
                    {
                        "cpu_bins": k,
                        "k_neighbors": int(kn),
                        "sigma": float(sg),
                        "target_cpu_bin": b,
                        "episode_id": int(s["episode_id"]),
                        "interval_id": int(s["interval_id"]),
                        "start_time": str(st),
                        "end_time": str(et),
                        "minutes": int(len(gtrue)),
                        "dominance": float(s["dominance"]),
                        "drop_pct": drop,
                        "true_rate_pct_per_h": true_rate,
                        "pred_rate_pct_per_h": pred_rate,
                        "true_dur_h": dur_h,
                        "pred_dur_h": pred_dur_h,
                        "acc_time": acc_time,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask_full.sum()),
                    }
                )

            mdf = pd.DataFrame(rows)
            if len(mdf):
                all_metrics.append(mdf)
                all_summary.append(
                    {
                        "cpu_bins": k,
                        "k_neighbors": int(kn),
                        "sigma": float(sg),
                        "n_slices": int(len(mdf)),
                        "acc_time_mean": float(mdf["acc_time"].mean()),
                        "acc_time_median": float(mdf["acc_time"].median()),
                        "acc_time_ge_0.90_frac": float((mdf["acc_time"] >= 0.90).mean()),
                    }
                )

    if not all_summary:
        raise RuntimeError("No results. Try lowering DOMINANCE_CPU or MIN_TRAIN.")

    pd.concat(all_metrics, ignore_index=True).to_csv(OUT_DIR / "metrics_by_slice.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_summary).to_csv(OUT_DIR / "summary_by_k.csv", index=False, encoding="utf-8")
    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

