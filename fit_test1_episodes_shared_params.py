"""
Fit shared parameters on test_1 episodes, but ONLY on discharge intervals.

We plot the FULL continuous episode time axis (including charging gaps),
while the orange prediction line is only drawn on discharge points.
R^2 / RMSE are computed ONLY on discharge points.

Inputs:
  processed/test1/episodes/test1_episode_points_1min.csv
  processed/test1/episodes/test1_episode_intervals.csv

Outputs:
  processed/test1/episodes_fit/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EP_DIR = BASE_DIR / "processed" / "test1" / "episodes"
IN_POINTS = EP_DIR / "test1_episode_points_1min.csv"
IN_INTERVALS = EP_DIR / "test1_episode_intervals.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "episodes_fit"


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
    """
    Predict SOC trajectory for one discharge interval with fixed initial SOC = observed first point.
    p = [k0, I_idle, alpha_cpu, gamma, delta_scr, beta_scr, alpha_mob, alpha_wifi, beta_T]
    """
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


def _objective(intervals: List[pd.DataFrame], p: np.ndarray) -> np.ndarray:
    # Equal weight per discharge interval (helps short / high-variance intervals not be ignored)
    res = []
    for itv in intervals:
        y = itv["soc"].to_numpy(float)
        yhat = _model_predict_segment(itv, p)
        r = y - yhat
        w = 1.0 / np.sqrt(max(1.0, float(np.isfinite(r).sum())))
        res.append(r * w)
    return np.concatenate(res)


def _fit_params(intervals: List[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    p0 = np.array([5.5e-6, 30.0, 400.0, 2.0, 0.8, 0.8, 1.25, 1.10, 0.03], float)
    lb = np.array([1e-7, 1.0, 0.0, 0.8, 0.0, 0.0, 1.0, 1.0, -0.2], float)
    ub = np.array([5e-5, 800.0, 5000.0, 4.0, 5.0, 5.0, 8.0, 8.0, 0.2], float)

    from scipy.optimize import least_squares  # type: ignore

    def fun(x: np.ndarray) -> np.ndarray:
        return _objective(intervals, x)

    best = None
    best_cost = np.inf
    rng = np.random.default_rng(7)
    for _ in range(24):
        jitter = rng.normal(scale=[2e-6, 20, 200, 0.4, 0.4, 0.4, 0.2, 0.2, 0.02], size=p0.shape)
        x0 = np.clip(p0 + jitter, lb, ub)
        sol = least_squares(fun, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=5000)
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best = sol.x
    assert best is not None
    return best, {"method": "scipy.optimize.least_squares", "cost": float(best_cost)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(IN_POINTS, parse_dates=["time"]).sort_values(["episode_id", "time"])
    itv_meta = pd.read_csv(IN_INTERVALS, parse_dates=["start_time", "end_time"])

    # Build discharge-interval frames (for fitting)
    intervals: List[pd.DataFrame] = []
    for (epi_id, itv_id), g in pts[pts["is_discharge"] == 1].groupby(["episode_id", "interval_id"]):
        g = g.sort_values("time").copy()
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
        intervals.append(g[keep].reset_index(drop=True))

    if not intervals:
        raise RuntimeError("No discharge intervals in episode points.")

    p, info = _fit_params(intervals)
    names = ["k0", "I_idle", "alpha_cpu", "gamma", "delta_scr", "beta_scr", "alpha_mob", "alpha_wifi", "beta_T"]
    pd.DataFrame({"param": names, "value": p}).to_csv(OUT_DIR / "fit_params.csv", index=False)
    (OUT_DIR / "fit_info.json").write_text(pd.Series(info).to_json(), encoding="utf-8")

    # Plot per episode on continuous axis; prediction only on discharge intervals
    import matplotlib.pyplot as plt

    ep_rows = []
    for epi_id, g_ep in pts.groupby("episode_id"):
        g_ep = g_ep.sort_values("time").reset_index(drop=True)
        t = np.arange(len(g_ep))
        y_obs = pd.to_numeric(g_ep["battery_level_pct"], errors="coerce").to_numpy(float)

        y_pred = np.full_like(y_obs, np.nan, dtype=float)

        # fill predictions interval-by-interval (piecewise initial condition)
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

        # Metrics only on discharge points
        m = g_ep["is_discharge"].to_numpy(int) == 1
        r2 = _r2(y_obs[m], y_pred[m])
        rmse = _rmse(y_obs[m], y_pred[m])
        ep_rows.append({"episode_id": int(epi_id), "r2_discharge": float(r2), "rmse_discharge": float(rmse)})

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, y_obs, color="#1f77b4", lw=1.4, label="SOC(%) observed (continuous)")
        ax.plot(t, y_pred, color="#ff7f0e", lw=1.8, label="shared-params prediction (discharge only)")

        # shade discharge intervals
        for itv_id, gg in g_ep[g_ep["is_discharge"] == 1].groupby("interval_id"):
            ax.axvspan(int(gg.index.min()), int(gg.index.max()), color="#2ca02c", alpha=0.08)

        ax.set_title(f"test_1 episode {int(epi_id)} (continuous incl. charging); metrics on discharge only")
        ax.legend(loc="best", framealpha=0.9, title=f"R²={r2:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel("Concatenated time (min)")
        ax.set_ylabel("SOC(%)")
        ax.grid(True, alpha=0.25)
        out = OUT_DIR / "figures" / f"episode_{int(epi_id)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(ep_rows).to_csv(OUT_DIR / "metrics_by_episode.csv", index=False)
    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

