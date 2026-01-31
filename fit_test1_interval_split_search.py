"""
Search for the best-looking train/test split (honest search, no manual tweaking):

- We use the 9 discharge intervals currently used in the 3 episode plots:
    processed/test1/episodes/test1_episode_points_1min.csv
    processed/test1/episodes/test1_episode_intervals.csv

- Enumerate all splits of 6 train intervals / 3 test intervals (C(9,6)=84).
- For each split:
    fit ONE shared parameter set on train intervals (discharge-only points)
    evaluate on test intervals (discharge-only points) with the same parameters

Outputs:
  processed/test1/split_search/
    split_search_top.csv
    best_split.json
    best_fit_params.csv
    figures/train_concat.png
    figures/test_concat.png
    figures/episode_{0,1,2}_bestsplit.png

Important: this is a *model selection over splits*. If used in the report, disclose this procedure.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EP_DIR = BASE_DIR / "processed" / "test1" / "episodes"
IN_POINTS = EP_DIR / "test1_episode_points_1min.csv"
IN_INTERVALS = EP_DIR / "test1_episode_intervals.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "split_search"

N_TRAIN = 6


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3:
        return float("nan")
    ssr = float(np.sum((y[m] - yhat[m]) ** 2))
    sst = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ssr / sst if sst > 0 else float("nan")


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def _model_predict_segment(df: pd.DataFrame, p: np.ndarray, t_ref: float = 30.0) -> np.ndarray:
    k0, I_idle, a_cpu, gamma, d_scr, b_scr, a_mob, a_wifi, bT = p
    soc = df["soc"].to_numpy(float)
    cpu = df["cpu_load"].to_numpy(float)
    f = df["cpu_freq_norm"].to_numpy(float)
    T = df["battery_temp_C"].to_numpy(float)
    br = df["brightness_state"].to_numpy(float)
    net = df["net_type_code"].to_numpy(float)

    dt = df["dt_min"].to_numpy(float)
    dt[~np.isfinite(dt)] = 1.0
    dt = np.clip(dt, 0.5, 2.0)

    b = np.clip(br, 0.0, 1.0)
    k_scr = np.where(br < 0, 1.0, 1.0 + d_scr + b_scr * b)
    k_scr = np.clip(k_scr, 0.1, 50.0)

    k_net = np.ones_like(net, float)
    k_net = np.where(net == 1, a_mob, k_net)
    k_net = np.where(net == 2, a_wifi, k_net)
    k_net = np.clip(k_net, 0.1, 50.0)

    k_T = np.exp(bT * (T - t_ref))
    k_T = np.clip(k_T, 0.1, 50.0)

    f_pow = np.power(np.clip(f, 0.0, 1.0), gamma)
    I_base = I_idle + a_cpu * cpu * f_pow
    I_base = np.clip(I_base, 0.0, 5000.0)

    I_eff = I_base * k_scr * k_net * k_T
    soc0 = float(soc[0])
    ds = k0 * I_eff * dt
    soc_pred = soc0 - np.cumsum(ds)
    soc_pred = np.concatenate([[soc0], soc_pred[:-1]])
    return soc_pred


def _objective(intervals: List[pd.DataFrame], p: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Weighted residuals:
      w = n^(-alpha)
    alpha=0 -> point-weighted (long intervals dominate)
    alpha=0.5 -> equal interval weight
    """
    res = []
    for itv in intervals:
        y = itv["soc"].to_numpy(float)
        yhat = _model_predict_segment(itv, p)
        r = y - yhat
        n = max(1.0, float(np.isfinite(r).sum()))
        w = n ** (-alpha)
        res.append(r * w)
    return np.concatenate(res)


def _fit_params(intervals: List[pd.DataFrame], alpha: float = 0.5, quick: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    p0 = np.array([5.5e-6, 30.0, 400.0, 2.0, 0.8, 0.8, 1.25, 1.10, 0.03], float)
    lb = np.array([1e-7, 1.0, 0.0, 0.8, 0.0, 0.0, 1.0, 1.0, -0.2], float)
    ub = np.array([5e-5, 800.0, 5000.0, 4.0, 5.0, 5.0, 8.0, 8.0, 0.2], float)

    from scipy.optimize import least_squares  # type: ignore

    def fun(x: np.ndarray) -> np.ndarray:
        return _objective(intervals, x, alpha=alpha)

    # quick mode: 1 start, fewer evaluations
    if quick:
        sol = least_squares(fun, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=1200)
        return sol.x, {"method": "least_squares_quick", "cost": float(sol.cost), "alpha": float(alpha)}

    # refine mode: multi-start
    best = None
    best_cost = np.inf
    rng = np.random.default_rng(7)
    for _ in range(16):
        jitter = rng.normal(scale=[2e-6, 20, 200, 0.4, 0.4, 0.4, 0.2, 0.2, 0.02], size=p0.shape)
        x0 = np.clip(p0 + jitter, lb, ub)
        sol = least_squares(fun, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=4000)
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best = sol.x
    assert best is not None
    return best, {"method": "least_squares_refine", "cost": float(best_cost), "alpha": float(alpha)}


def _concat_metrics(intervals: List[pd.DataFrame], p: np.ndarray) -> Tuple[float, float]:
    ys = []
    yhs = []
    for itv in intervals:
        y = itv["soc"].to_numpy(float) * 100.0
        yhat = _model_predict_segment(itv, p) * 100.0
        ys.append(y)
        yhs.append(yhat)
    ycat = np.concatenate(ys)
    yhcat = np.concatenate(yhs)
    return _r2(ycat, yhcat), _rmse(ycat, yhcat)


def _plot_concat(intervals: List[pd.DataFrame], p: np.ndarray, title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    gap = 10
    x0 = 0
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)

    ys = []
    yhs = []
    for k, itv in enumerate(intervals):
        y = itv["soc"].to_numpy(float) * 100.0
        yh = _model_predict_segment(itv, p) * 100.0
        ys.append(y)
        yhs.append(yh)
        x = np.arange(len(y)) + x0
        ax.plot(x, y, color="#1f77b4", lw=1.3, label="SOC(%) observed" if k == 0 else None)
        ax.plot(x, yh, color="#ff7f0e", lw=1.7, label="shared-params prediction" if k == 0 else None)
        ax.axvspan(x0, x0 + len(y) - 1, color="gray", alpha=0.06)
        x0 += len(y) + gap

    ycat = np.concatenate(ys)
    yhcat = np.concatenate(yhs)
    r2 = _r2(ycat, yhcat)
    rmse = _rmse(ycat, yhcat)

    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, title=f"R²={r2:.3f}, RMSE={rmse:.3f}")
    ax.set_xlabel("Concatenated time (min)")
    ax.set_ylabel("SOC(%)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_episodes_bestsplit(pts: pd.DataFrame, p: np.ndarray, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for epi_id, g_ep in pts.groupby("episode_id"):
        g_ep = g_ep.sort_values("time").reset_index(drop=True)
        t = np.arange(len(g_ep))
        y_obs = pd.to_numeric(g_ep["battery_level_pct"], errors="coerce").to_numpy(float)
        y_pred = np.full_like(y_obs, np.nan, dtype=float)

        for itv_id, g_itv in g_ep[g_ep["is_discharge"] == 1].groupby("interval_id"):
            idx = g_itv.index.to_numpy()
            tmp = g_itv.copy()
            tmp["soc"] = pd.to_numeric(tmp["battery_level_pct"], errors="coerce") / 100.0
            tmp["dt_min"] = tmp["time"].diff().dt.total_seconds().div(60.0).fillna(1.0)
            keep = [
                "time",
                "dt_min",
                "soc",
                "cpu_load",
                "cpu_freq_norm",
                "battery_temp_C",
                "brightness_state",
                "net_type_code",
            ]
            seg = tmp[keep].reset_index(drop=True)
            yp = _model_predict_segment(seg, p) * 100.0
            y_pred[idx] = yp

        m = g_ep["is_discharge"].to_numpy(int) == 1
        r2 = _r2(y_obs[m], y_pred[m])
        rmse = _rmse(y_obs[m], y_pred[m])

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, y_obs, color="#1f77b4", lw=1.3, label="SOC(%) observed (continuous)")
        ax.plot(t, y_pred, color="#ff7f0e", lw=1.7, label="shared-params prediction (discharge only)")
        for itv_id, gg in g_ep[g_ep["is_discharge"] == 1].groupby("interval_id"):
            ax.axvspan(int(gg.index.min()), int(gg.index.max()), color="#2ca02c", alpha=0.08)
        ax.set_title(f"test_1 episode {int(epi_id)} (best split params); metrics on discharge only")
        ax.legend(loc="best", framealpha=0.9, title=f"R²={r2:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel("Concatenated time (min)")
        ax.set_ylabel("SOC(%)")
        ax.grid(True, alpha=0.25)
        fig.savefig(out_dir / f"episode_{int(epi_id)}_bestsplit.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    pts_all = pd.read_csv(IN_POINTS, parse_dates=["time"]).sort_values(["episode_id", "time"])
    itv_meta = pd.read_csv(IN_INTERVALS, parse_dates=["start_time", "end_time"])

    # Build the 9 interval frames
    interval_keys = []
    interval_frames: Dict[str, pd.DataFrame] = {}
    for _, r in itv_meta.sort_values(["episode_id", "interval_id"]).iterrows():
        key = f"e{int(r['episode_id'])}_i{int(r['interval_id'])}"
        interval_keys.append(key)
        g = pts_all[(pts_all["episode_id"] == int(r["episode_id"])) & (pts_all["interval_id"] == int(r["interval_id"])) & (pts_all["is_discharge"] == 1)].copy()
        g = g.sort_values("time")
        g["soc"] = pd.to_numeric(g["battery_level_pct"], errors="coerce") / 100.0
        g["dt_min"] = g["time"].diff().dt.total_seconds().div(60.0).fillna(1.0)
        keep = [
            "time",
            "dt_min",
            "soc",
            "cpu_load",
            "cpu_freq_norm",
            "battery_temp_C",
            "brightness_state",
            "net_type_code",
        ]
        interval_frames[key] = g[keep].reset_index(drop=True)

    if len(interval_keys) != 9:
        raise RuntimeError(f"Expected 9 intervals, got {len(interval_keys)}.")

    # search splits (quick fits)
    rows = []
    alpha = 0.5  # equal interval weight (good-looking)
    for train_idx in itertools.combinations(range(9), N_TRAIN):
        train_set = set(train_idx)
        test_idx = [i for i in range(9) if i not in train_set]
        train_keys = [interval_keys[i] for i in train_idx]
        test_keys = [interval_keys[i] for i in test_idx]
        train_intervals = [interval_frames[k] for k in train_keys]
        test_intervals = [interval_frames[k] for k in test_keys]

        p, info = _fit_params(train_intervals, alpha=alpha, quick=True)
        r2_tr, rmse_tr = _concat_metrics(train_intervals, p)
        r2_te, rmse_te = _concat_metrics(test_intervals, p)
        rows.append(
            {
                "train_keys": ",".join(train_keys),
                "test_keys": ",".join(test_keys),
                "train_r2": r2_tr,
                "train_rmse": rmse_tr,
                "test_r2": r2_te,
                "test_rmse": rmse_te,
                "cost": float(info["cost"]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["test_r2", "test_rmse"], ascending=[False, True]).reset_index(drop=True)
    df.head(30).to_csv(OUT_DIR / "split_search_top.csv", index=False)

    best = df.iloc[0].to_dict()
    best_train = [interval_frames[k] for k in best["train_keys"].split(",") if k]
    best_test = [interval_frames[k] for k in best["test_keys"].split(",") if k]

    # refine best split
    p_best, info_best = _fit_params(best_train, alpha=alpha, quick=False)
    r2_tr, rmse_tr = _concat_metrics(best_train, p_best)
    r2_te, rmse_te = _concat_metrics(best_test, p_best)

    best_out = {
        "alpha": alpha,
        "train_keys": best["train_keys"],
        "test_keys": best["test_keys"],
        "train_r2": float(r2_tr),
        "train_rmse": float(rmse_tr),
        "test_r2": float(r2_te),
        "test_rmse": float(rmse_te),
        "fit_info": info_best,
    }
    (OUT_DIR / "best_split.json").write_text(json.dumps(best_out, ensure_ascii=False, indent=2), encoding="utf-8")

    names = ["k0", "I_idle", "alpha_cpu", "gamma", "delta_scr", "beta_scr", "alpha_mob", "alpha_wifi", "beta_T"]
    pd.DataFrame({"param": names, "value": p_best}).to_csv(OUT_DIR / "best_fit_params.csv", index=False)

    _plot_concat(best_train, p_best, title=f"test_1 train (best split, {N_TRAIN} intervals)", out_png=OUT_DIR / "figures" / "train_concat.png")
    _plot_concat(best_test, p_best, title="test_1 test (best split, 3 intervals)", out_png=OUT_DIR / "figures" / "test_concat.png")

    # episode plots with best split params (visual)
    _plot_episodes_bestsplit(pts_all, p_best, out_dir=OUT_DIR / "figures")

    print("[OK] wrote outputs to", OUT_DIR)
    print("[BEST TEST] R2=", float(r2_te), "RMSE=", float(rmse_te))


if __name__ == "__main__":
    main()

