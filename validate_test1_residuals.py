"""
Residual diagnostics for test_1 SOC fit (episode-based shared parameters).

We mirror the residual tests used on the main dataset (validate_discharge_model.py):
  - Durbin–Watson
  - Ljung–Box p-value (lag 20)
  - Jarque–Bera p-value
  - residual vs fitted, histogram, QQ, ACF plots

Inputs:
  processed/test1/episodes/test1_episode_points_1min.csv
  processed/test1/episodes_fit/fit_params.csv

Outputs (fixed paths):
  processed/test1/episodes_fit/
    residuals_discharge.csv
    residual_diagnostics.csv
    residual_diagnostics_by_episode.csv
    figures/diagnostics/
      residual_vs_fitted.png
      residual_hist.png
      residual_qq.png (if scipy available)
      residual_acf.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
PARAMS_PATH = BASE_DIR / "processed" / "test1" / "episodes_fit" / "fit_params.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "episodes_fit"
FIG_DIR = OUT_DIR / "figures" / "diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TREF_C = 30.0


def _try_import_scipy():
    try:
        import scipy.stats as st  # type: ignore

        return st
    except Exception:
        return None


STATS = _try_import_scipy()


def durbin_watson(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 3:
        return float("nan")
    num = float(np.sum(np.diff(resid) ** 2))
    den = float(np.sum(resid**2))
    return num / den if den > 0 else float("nan")


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 2:
        return np.full(nlags + 1, np.nan)
    x = x - np.mean(x)
    denom = float(np.sum(x * x))
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float(np.sum(x[:-k] * x[k:]) / denom) if denom > 0 else float("nan"))
    return np.array(out)


def ljung_box_pvalue(resid: np.ndarray, lags: int = 20) -> float:
    if STATS is None:
        return float("nan")
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    r = acf(resid, nlags=lags)[1:]  # drop lag0
    n = int(len(resid))
    if n <= lags + 1 or not np.all(np.isfinite(r)):
        return float("nan")
    Q = n * (n + 2) * float(np.sum((r**2) / (n - np.arange(1, lags + 1))))
    return float(STATS.chi2.sf(Q, df=lags))


def jarque_bera_pvalue(resid: np.ndarray) -> float:
    if STATS is None:
        return float("nan")
    x = np.asarray(resid, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    return float(STATS.jarque_bera(x).pvalue)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if int(m.sum()) < 3:
        return float("nan")
    ssr = float(np.sum((y[m] - yhat[m]) ** 2))
    sst = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ssr / sst if sst > 0 else float("nan")


def rmse_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if int(m.sum()) < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def _load_params() -> np.ndarray:
    df = pd.read_csv(PARAMS_PATH)
    df["param"] = df["param"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    m = df.dropna(subset=["param", "value"]).set_index("param")["value"]
    names = ["k0", "I_idle", "alpha_cpu", "gamma", "delta_scr", "beta_scr", "alpha_mob", "alpha_wifi", "beta_T"]
    p = np.array([float(m[n]) for n in names], dtype=float)
    return p


def _model_predict_segment(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """
    Predict SOC trajectory for one discharge interval with fixed initial SOC = observed first point.
    SOC in [0,1].
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

    k_T = np.exp(bT * (T - TREF_C))
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


def residual_plots(y: np.ndarray, yhat: np.ndarray, resid: np.ndarray) -> None:
    # residual vs fitted
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(yhat, resid, s=6, alpha=0.25)
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_title("Residual vs fitted (SOC %, discharge-only)")
    ax.set_xlabel("fitted SOC(%)")
    ax.set_ylabel("residual (pct-pt)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_vs_fitted.png", dpi=160)
    plt.close(fig)

    # hist
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(resid[np.isfinite(resid)], bins=60, alpha=0.85)
    ax.set_title("Residual histogram (SOC %, discharge-only)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_hist.png", dpi=160)
    plt.close(fig)

    # QQ plot (if scipy available)
    if STATS is not None:
        x = resid[np.isfinite(resid)]
        x = (x - float(np.mean(x))) / (float(np.std(x)) + 1e-12)
        n = len(x)
        if n > 50:
            theo = STATS.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            samp = np.sort(x)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(theo, samp, s=6, alpha=0.4)
            lo = float(min(theo.min(), samp.min()))
            hi = float(max(theo.max(), samp.max()))
            ax.plot([lo, hi], [lo, hi], color="k", lw=1, alpha=0.6)
            ax.set_title("QQ plot (standardized residual)")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "residual_qq.png", dpi=160)
            plt.close(fig)

    # ACF plot
    r = resid[np.isfinite(resid)]
    a = acf(r, nlags=60)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(np.arange(len(a)), a, width=0.8)
    ax.set_title("Residual ACF (pooled, naive concat) up to 60 lags")
    ax.set_xlabel("lag (min)")
    ax.set_ylabel("acf")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_acf.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    p = _load_params()
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()
    if pts.empty:
        raise RuntimeError("No discharge points in test1_episode_points_1min.csv (is_discharge==1).")

    # compute dt within each interval
    pts["battery_level_pct"] = pd.to_numeric(pts["battery_level_pct"], errors="coerce")
    pts["soc"] = pts["battery_level_pct"] / 100.0
    pts["dt_min"] = pts.groupby(["episode_id", "interval_id"])["time"].diff().dt.total_seconds().div(60.0).fillna(1.0)

    # predict interval-by-interval (piecewise initial condition)
    # NOTE: pts keeps original row indices after filtering, so do NOT index a length-N numpy array with them.
    pts["soc_pred"] = np.nan
    for (epi_id, itv_id), g in pts.groupby(["episode_id", "interval_id"]):
        idx = g.index
        seg = g[
            [
                "time",
                "dt_min",
                "soc",
                "cpu_load",
                "cpu_freq_norm",
                "battery_temp_C",
                "brightness_state",
                "net_type_code",
            ]
        ].reset_index(drop=True)
        yp = _model_predict_segment(seg, p)
        pts.loc[idx, "soc_pred"] = yp
    pts["y_obs_pct"] = pts["soc"] * 100.0
    pts["y_pred_pct"] = pts["soc_pred"] * 100.0
    pts["resid_pct"] = pts["y_obs_pct"] - pts["y_pred_pct"]

    out_cols = [
        "episode_id",
        "interval_id",
        "time",
        "y_obs_pct",
        "y_pred_pct",
        "resid_pct",
        "dt_min",
    ]
    pts[out_cols].to_csv(OUT_DIR / "residuals_discharge.csv", index=False, encoding="utf-8")

    # overall diagnostics (order by time within episode for DW/LB)
    ord_pts = pts.sort_values(["episode_id", "time"]).reset_index(drop=True)
    y = ord_pts["y_obs_pct"].to_numpy(float)
    yhat = ord_pts["y_pred_pct"].to_numpy(float)
    resid = ord_pts["resid_pct"].to_numpy(float)

    diag = {
        "n": int(np.isfinite(resid).sum()),
        "R2_in_sample": float(r2_score(y, yhat)),
        "RMSE_in_sample": float(rmse_score(y, yhat)),
        "resid_mean": float(np.nanmean(resid)),
        "resid_std": float(np.nanstd(resid)),
        "durbin_watson": float(durbin_watson(resid)),
        "ljung_box_pvalue_lag20": float(ljung_box_pvalue(resid, lags=20)),
        "jarque_bera_pvalue": float(jarque_bera_pvalue(resid)),
    }
    pd.DataFrame([diag]).to_csv(OUT_DIR / "residual_diagnostics.csv", index=False, encoding="utf-8")

    # per episode
    epi_rows = []
    for epi_id, g in ord_pts.groupby("episode_id"):
        y_e = g["y_obs_pct"].to_numpy(float)
        yh_e = g["y_pred_pct"].to_numpy(float)
        r_e = g["resid_pct"].to_numpy(float)
        epi_rows.append(
            {
                "episode_id": int(epi_id),
                "n": int(np.isfinite(r_e).sum()),
                "R2_in_sample": float(r2_score(y_e, yh_e)),
                "RMSE_in_sample": float(rmse_score(y_e, yh_e)),
                "resid_mean": float(np.nanmean(r_e)),
                "resid_std": float(np.nanstd(r_e)),
                "durbin_watson": float(durbin_watson(r_e)),
                "ljung_box_pvalue_lag20": float(ljung_box_pvalue(r_e, lags=20)),
                "jarque_bera_pvalue": float(jarque_bera_pvalue(r_e)),
            }
        )
    pd.DataFrame(epi_rows).to_csv(OUT_DIR / "residual_diagnostics_by_episode.csv", index=False, encoding="utf-8")

    residual_plots(y, yhat, resid)

    print("[OK] wrote", OUT_DIR / "residuals_discharge.csv")
    print("[OK] wrote", OUT_DIR / "residual_diagnostics.csv")
    print("[OK] wrote", OUT_DIR / "residual_diagnostics_by_episode.csv")
    print("[OK] wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()

