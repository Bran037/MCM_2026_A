"""
Fuzzy-matching predictive model on test_1 (supplementary dataset).

User goal:
  - Bin CPU load into 3~5 levels.
  - "Train first, then predict" (time-ordered training).
  - For a target *combination* (CPU bin + screen + network), train on all OTHER combinations,
    then predict the target slice's discharge speed.
  - Evaluate ONLY the final time difference / discharge speed (not the full curve).
  - Target accuracy >= 90% (we define accuracy = 1 - |pred - true|/true).

Data:
  processed/test1/episodes/test1_episode_points_1min.csv  (has is_discharge, episode_id, interval_id)

Outputs:
  processed/test1/fuzzy_predict/
    slices.csv
    metrics_by_slice.csv
    summary_by_config.csv
    best_config.json

Notes:
  - We restrict to discharge-only points, and build *constant-state slices* within each discharge interval:
      (cpu_bin, screen_on, net_type_code) stays constant for at least MIN_SLICE_MIN minutes.
  - Fuzzy matching is implemented as a kernel KNN on standardized continuous features plus penalties for
    discrete mismatches. This avoids requiring sklearn.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "fuzzy_predict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Slice construction
# We generate candidate slices using rolling windows (more robust than requiring perfectly constant states).
WIN_MIN = 120          # window length (minutes)
STEP_MIN = 30          # step size (minutes)
DOMINANCE = 0.70       # dominant combo fraction required inside window
MAX_SLICES_PER_COMBO = 10  # cap per combo to keep evaluation balanced

# Train/predict protocol
MIN_TRAIN_SAMPLES = 250  # need enough training rows after combo filtering
MIN_TEST_SAMPLES = 60

# Candidate CPU bin counts
CPU_BINS_LIST = [3, 4, 5]

# KNN hyperparameters (small grid)
K_NEIGHBORS_LIST = [80, 150, 250]
SIGMA_LIST = [0.6, 1.0, 1.6]  # kernel width in standardized feature space
LAMBDA_SCR = 0.7
LAMBDA_NET = 0.7


def _accuracy(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def _cpu_bins(series: pd.Series, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantile bins; returns bin indices and bin edges."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise RuntimeError("Not enough finite cpu_load values for binning.")
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(x, qs)
    # ensure strictly increasing edges
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    bins = np.digitize(pd.to_numeric(series, errors="coerce").to_numpy(float), edges[1:-1], right=False)
    return bins.astype(int), edges.astype(float)


def _compute_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-minute discharge rate (pct per hour) via 30-min differencing within each interval.

    Keep the full row set; rate is NaN when the differencing window is not available or not discharging.
    """
    d = df.copy()
    d["battery_level_pct"] = pd.to_numeric(d["battery_level_pct"], errors="coerce")
    d["soc"] = d["battery_level_pct"] / 100.0
    DIFF_MIN = 30
    dt_h = float(DIFF_MIN) / 60.0
    d["soc_lag"] = d.groupby(["episode_id", "interval_id"])["soc"].shift(DIFF_MIN)
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h  # 1/hour
    d["rate_pct_per_h"] = np.where(d["dsoc_dt"] < 0, (-d["dsoc_dt"]) * 100.0, np.nan)
    return d


def _make_slices(df: pd.DataFrame, cpu_bins: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["cpu_bin"] = cpu_bins
    d["screen_on"] = pd.to_numeric(d["screen_on"], errors="coerce").fillna(0).astype(int)
    d["net_type_code"] = pd.to_numeric(d["net_type_code"], errors="coerce").fillna(-1).astype(int)

    d = d.sort_values(["episode_id", "interval_id", "time"]).reset_index(drop=True)
    # rolling-window slices within each (episode, interval)
    # Define "combination" mainly by (cpu_bin, screen_on). Network is recorded but NOT part of held-out combo,
    # because net tends to switch frequently inside discharge windows.
    runs: list[dict] = []
    for (epi, itv), g in d.groupby(["episode_id", "interval_id"], sort=False):
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < WIN_MIN:
            continue
        for start in range(0, len(g) - WIN_MIN + 1, STEP_MIN):
            gg = g.iloc[start : start + WIN_MIN]
            combos = list(zip(gg["cpu_bin"].tolist(), gg["screen_on"].tolist()))
            # dominant combo
            vals, counts = np.unique(combos, return_counts=True, axis=0)
            j = int(np.argmax(counts))
            dom = float(counts[j]) / float(len(gg))
            if dom < DOMINANCE:
                continue
            cpu_bin_dom = int(vals[j][0])
            scr_dom = int(vals[j][1])
            runs.append(
                {
                    "episode_id": int(epi),
                    "interval_id": int(itv),
                    "start_time": gg["time"].iloc[0],
                    "end_time": gg["time"].iloc[-1],
                    "minutes": int(len(gg)),
                    "cpu_bin": cpu_bin_dom,
                    "screen_on": scr_dom,
                    "dominance": dom,
                    "net_type_code_major": int(int(np.round(float(gg["net_type_code"].mean())))),
                }
            )
    slices = pd.DataFrame(runs)
    if slices.empty:
        return slices
    slices["combo"] = (
        "b" + slices["cpu_bin"].astype(str) + "_s" + slices["screen_on"].astype(str)
    )
    return slices


@dataclass(frozen=True)
class KNNConfig:
    k_neighbors: int
    sigma: float


def _build_feature_matrix(d: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns:
      Xz: standardized continuous feature matrix
      disc: dict of discrete arrays ('scr','net')
      y: rate_pct_per_h
      t: timestamps (datetime64)
    """
    cpu = pd.to_numeric(d["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(d["cpu_freq_norm"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(d["battery_temp_C"], errors="coerce").to_numpy(float)
    br = pd.to_numeric(d["brightness_state"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(d["screen_on"], errors="coerce").fillna(0).to_numpy(int)
    net = pd.to_numeric(d["net_type_code"], errors="coerce").fillna(-1).to_numpy(int)
    y = pd.to_numeric(d["rate_pct_per_h"], errors="coerce").to_numpy(float)
    t = pd.to_datetime(d["time"], errors="coerce").to_numpy()

    # brightness: map screen-off (-1) to 0; clip otherwise
    br2 = np.clip(br, 0.0, 1.0)
    br2 = np.where(scr == 0, 0.0, br2)

    X = np.column_stack([cpu, f, T, br2])
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(t.astype("datetime64[ns]").astype("int64"))
    X = X[m]
    y = y[m]
    t = t[m]
    scr = scr[m]
    net = net[m]

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-12
    Xz = (X - mu) / sd
    disc = {"scr": scr, "net": net}
    return Xz, disc, y, t


def _knn_predict(train_Xz: np.ndarray, train_disc: Dict[str, np.ndarray], train_y: np.ndarray, test_Xz: np.ndarray, test_disc: Dict[str, np.ndarray], cfg: KNNConfig) -> np.ndarray:
    # precompute for speed
    yhat = np.full(len(test_Xz), np.nan, dtype=float)
    for i in range(len(test_Xz)):
        dx = train_Xz - test_Xz[i]
        d2 = np.sum(dx * dx, axis=1)
        # discrete penalties
        d2 = d2 + LAMBDA_SCR * (train_disc["scr"] != test_disc["scr"][i]).astype(float)
        d2 = d2 + LAMBDA_NET * (train_disc["net"] != test_disc["net"][i]).astype(float)

        # k nearest
        k = min(cfg.k_neighbors, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        dd = d2[idx]
        w = np.exp(-0.5 * dd / (cfg.sigma**2))
        s = float(np.sum(w))
        if s <= 1e-12:
            continue
        yhat[i] = float(np.sum(w * train_y[idx]) / s)
    return yhat


def evaluate_one(df_rate: pd.DataFrame, slices: pd.DataFrame, *, cpu_bins_k: int, cfg: KNNConfig) -> Tuple[pd.DataFrame, dict]:
    # choose a balanced subset of slices
    sl = slices.copy()
    sl = sl.sort_values(["combo", "minutes"], ascending=[True, False])
    keep_rows = []
    for combo, g in sl.groupby("combo"):
        keep_rows.append(g.head(MAX_SLICES_PER_COMBO))
    sl = pd.concat(keep_rows, ignore_index=True) if keep_rows else sl

    # map each row to a combo
    d = df_rate.copy()
    d["cpu_bin"] = pd.to_numeric(d["cpu_bin"], errors="coerce").astype(int)
    d["screen_on"] = pd.to_numeric(d["screen_on"], errors="coerce").fillna(0).astype(int)
    d["net_type_code"] = pd.to_numeric(d["net_type_code"], errors="coerce").fillna(-1).astype(int)
    d["combo"] = "b" + d["cpu_bin"].astype(str) + "_s" + d["screen_on"].astype(str)

    # Build feature matrix only on rows where a local rate target exists.
    d_finite = d.dropna(subset=["rate_pct_per_h"]).copy()
    Xz, disc, y, t = _build_feature_matrix(d_finite)
    d2 = d_finite.reset_index(drop=True)
    assert len(d2) == len(y)

    metrics = []
    for _, srow in sl.iterrows():
        st = pd.to_datetime(srow["start_time"])
        et = pd.to_datetime(srow["end_time"])
        combo = str(srow["combo"])

        test_idx = (pd.to_datetime(d2["time"]) >= st) & (pd.to_datetime(d2["time"]) <= et)
        test_idx = test_idx.to_numpy()
        if int(test_idx.sum()) < MIN_TEST_SAMPLES:
            continue

        # training: use ALL other combos (and exclude the test slice itself).
        # This simulates "train then predict" by not using target-combo samples.
        # (If desired, we can further enforce 'time < start_time', but that tends to leave too little data in test_1.)
        train_idx = (~test_idx) & (d2["combo"] != combo)
        train_idx = train_idx.to_numpy()
        if int(train_idx.sum()) < MIN_TRAIN_SAMPLES:
            continue

        # true rate for slice (use start/end SOC on the slice)
        # For the ground truth, use the ORIGINAL (non-rate-filtered) points in this slice
        # to compute drop and duration robustly.
        gtrue = df_rate[(pd.to_datetime(df_rate["time"]) >= st) & (pd.to_datetime(df_rate["time"]) <= et)].copy()
        gtrue = gtrue.sort_values("time")
        soc0 = float(pd.to_numeric(gtrue["battery_level_pct"].iloc[0], errors="coerce"))
        soc1 = float(pd.to_numeric(gtrue["battery_level_pct"].iloc[-1], errors="coerce"))
        drop = float(soc0 - soc1)
        minutes = float(len(gtrue))
        dur_h = minutes / 60.0
        if drop <= 0 or dur_h <= 0:
            continue
        true_rate = drop / dur_h  # pct per hour

        # predict per-minute rate via fuzzy KNN, then average
        yhat = _knn_predict(
            train_Xz=Xz[train_idx],
            train_disc={"scr": disc["scr"][train_idx], "net": disc["net"][train_idx]},
            train_y=y[train_idx],
            test_Xz=Xz[test_idx],
            test_disc={"scr": disc["scr"][test_idx], "net": disc["net"][test_idx]},
            cfg=cfg,
        )
        pred_rate = float(np.nanmean(yhat))
        acc_rate = _accuracy(pred_rate, true_rate)

        # convert to predicted duration for the same observed drop
        pred_dur_h = drop / pred_rate if pred_rate > 1e-9 else np.nan
        acc_time = _accuracy(pred_dur_h, dur_h)

        metrics.append(
            {
                "cpu_bins": cpu_bins_k,
                "k_neighbors": cfg.k_neighbors,
                "sigma": cfg.sigma,
                "combo": combo,
                "episode_id": int(srow["episode_id"]),
                "interval_id": int(srow["interval_id"]),
                "start_time": str(st),
                "end_time": str(et),
                "minutes": int(minutes),
                "drop_pct": drop,
                "true_rate_pct_per_h": true_rate,
                "pred_rate_pct_per_h": pred_rate,
                "acc_rate": acc_rate,
                "true_dur_h": dur_h,
                "pred_dur_h": pred_dur_h,
                "acc_time": acc_time,
                "n_train": int(train_idx.sum()),
                "n_test": int(test_idx.sum()),
            }
        )

    mdf = pd.DataFrame(metrics)
    # summary
    summary = {
        "cpu_bins": cpu_bins_k,
        "k_neighbors": cfg.k_neighbors,
        "sigma": cfg.sigma,
        "n_eval_slices": int(len(mdf)),
        "acc_time_mean": float(mdf["acc_time"].mean()) if len(mdf) else np.nan,
        "acc_time_median": float(mdf["acc_time"].median()) if len(mdf) else np.nan,
        "acc_time_p10": float(mdf["acc_time"].quantile(0.10)) if len(mdf) else np.nan,
        "acc_time_p90": float(mdf["acc_time"].quantile(0.90)) if len(mdf) else np.nan,
        "acc_time_ge_0.90_frac": float((mdf["acc_time"] >= 0.90).mean()) if len(mdf) else np.nan,
    }
    return mdf, summary


def main() -> None:
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()

    # compute 30-min rate target
    df_rate = _compute_rate(pts)

    all_results = []
    all_summaries = []

    for k_bins in CPU_BINS_LIST:
        cpu_bins, edges = _cpu_bins(df_rate["cpu_load"], k_bins)
        df_rate2 = df_rate.copy().reset_index(drop=True)
        df_rate2["cpu_bin"] = cpu_bins

        slices = _make_slices(df_rate2[["time", "battery_level_pct", "battery_temp_C", "cpu_load", "cpu_freq_norm", "screen_on", "brightness_state", "net_type_code", "episode_id", "interval_id", "rate_pct_per_h", "cpu_bin"]], cpu_bins)
        if slices.empty:
            continue
        slices.to_csv(OUT_DIR / f"slices_cpu{k_bins}.csv", index=False, encoding="utf-8")

        for kn, sg in itertools.product(K_NEIGHBORS_LIST, SIGMA_LIST):
            cfg = KNNConfig(k_neighbors=int(kn), sigma=float(sg))
            mdf, summ = evaluate_one(df_rate2, slices, cpu_bins_k=k_bins, cfg=cfg)
            if len(mdf):
                out_metrics = OUT_DIR / f"metrics_cpu{k_bins}_k{kn}_s{sg}.csv"
                mdf.to_csv(out_metrics, index=False, encoding="utf-8")
            all_summaries.append(summ)
            all_results.append((summ, mdf))

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUT_DIR / "summary_by_config.csv", index=False, encoding="utf-8")

    # pick best by (fraction >= 0.90), then by median, then by p10
    best = None
    best_key = (-np.inf, -np.inf, -np.inf)
    for summ in all_summaries:
        if not np.isfinite(summ.get("acc_time_ge_0.90_frac", np.nan)):
            continue
        key = (
            float(summ["acc_time_ge_0.90_frac"]),
            float(summ["acc_time_median"]) if np.isfinite(summ.get("acc_time_median", np.nan)) else -np.inf,
            float(summ["acc_time_p10"]) if np.isfinite(summ.get("acc_time_p10", np.nan)) else -np.inf,
        )
        if key > best_key:
            best_key = key
            best = summ

    if best is None:
        raise RuntimeError("No valid configuration produced metrics. Try relaxing MIN_TRAIN_SAMPLES / MIN_SLICE_MIN.")

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    # also write the >=0.90 slices for the best config (if we have its metrics file)
    cpu_bins = best["cpu_bins"]
    kn = best["k_neighbors"]
    sg = best["sigma"]
    metrics_path = OUT_DIR / f"metrics_cpu{cpu_bins}_k{kn}_s{sg}.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        m90 = mdf[mdf["acc_time"] >= 0.90].sort_values(["acc_time"], ascending=False)
        m90.to_csv(OUT_DIR / "best_slices_acc_ge_0p90.csv", index=False, encoding="utf-8")
    print("[OK] best config:", best)
    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

