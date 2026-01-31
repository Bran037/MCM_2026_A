"""
Significance tests & independence tests on test_1 (supplementary dataset).

We work on the discharge-only episode/interval points:
  processed/test1/episodes/test1_episode_points_1min.csv

Target variables (groups):
  - CPU group: cpu_load, cpu_freq_norm, cpu_load*cpu_freq_norm
  - Brightness group: scr (screen_on), bright (brightness_state clipped to [0,1], 0 when screen off)
  - Network group: net_wifi (baseline: mobile; NOTE: in our discharge-only windows, net_type_code is always {1,2})
  - Temperature group: T0 = battery_temp_C - 30

We build a rate panel:
  r(t) = -dSOC/dt  (1/hour), using DIFF_MIN differencing within each discharge interval.
  y = log r

We estimate OLS and compute HAC (Newey–West) covariance clustered by interval
to avoid mixing discontinuous segments.

Outputs:
  processed/test1/significance/
    rate_panel_logr.csv
    model_summary.json
    params_with_hac_main.csv
    wald_tests_main.csv
    params_with_hac_interactions.csv
    wald_tests_interactions.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PTS_PATH = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREF_C = 30.0
DIFF_MIN = 5
HAC_LAG = 20


def _try_import_scipy():
    try:
        import scipy.stats as st  # type: ignore

        return st
    except Exception:
        return None


STATS = _try_import_scipy()


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    yhat: np.ndarray
    resid: np.ndarray
    r2: float
    rmse: float
    names: list[str]


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if int(m.sum()) < 3:
        return float("nan")
    ss_res = float(np.sum((y[m] - yhat[m]) ** 2))
    ss_tot = float(np.sum((y[m] - float(np.mean(y[m]))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if int(m.sum()) < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def ols_fit(X: np.ndarray, y: np.ndarray, names: list[str]) -> Fit:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    return Fit(beta=beta, yhat=yhat, resid=resid, r2=r2_score(y, yhat), rmse=rmse_score(y, yhat), names=names)


def ols_cov(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    n, p = X.shape
    rss = float(np.sum(resid**2))
    df = max(1, n - p)
    sigma2 = rss / df
    xtx_inv = np.linalg.pinv(X.T @ X)
    return sigma2 * xtx_inv


def hac_cov_by_cluster(panel: pd.DataFrame, X: np.ndarray, resid: np.ndarray, *, lag: int, cluster_col: str) -> np.ndarray:
    """
    Newey–West HAC covariance with clustering by cluster_col (interval).
    Bartlett kernel, same lag for each cluster.
    """
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    S = np.zeros((p, p), dtype=float)

    clusters = panel[cluster_col].astype(str).to_numpy()
    for cl in np.unique(clusters):
        idx = np.where(clusters == cl)[0]
        if len(idx) < 3:
            continue
        Xc = X[idx, :]
        uc = resid[idx]
        Tc = len(idx)

        # lag 0
        Xu = Xc * uc[:, None]
        S += Xu.T @ Xu

        L = min(lag, Tc - 1)
        for l in range(1, L + 1):
            w = 1.0 - l / (L + 1.0)
            Xu0 = Xc[l:, :] * uc[l:, None]
            XuL = Xc[:-l, :] * uc[:-l, None]
            S += w * (Xu0.T @ XuL + XuL.T @ Xu0)

    return xtx_inv @ S @ xtx_inv


def z_pvalues(z: np.ndarray) -> Optional[np.ndarray]:
    if STATS is None:
        return None
    return 2.0 * STATS.norm.sf(np.abs(z))


def chi2_pvalue(stat: float, df: int) -> Optional[float]:
    if STATS is None:
        return None
    return float(STATS.chi2.sf(stat, df=df))


def wald_test(beta: np.ndarray, cov: np.ndarray, idx: list[int], label: str) -> dict:
    b = beta[idx]
    C = cov[np.ix_(idx, idx)]
    Cinv = np.linalg.pinv(C)
    stat = float(b.T @ Cinv @ b)
    p = chi2_pvalue(stat, df=len(idx))
    return {"test": label, "df": len(idx), "stat": stat, "pvalue": p if p is not None else float("nan")}


def build_rate_panel() -> pd.DataFrame:
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()

    # numeric
    pts["battery_level_pct"] = pd.to_numeric(pts["battery_level_pct"], errors="coerce")
    pts["battery_temp_C"] = pd.to_numeric(pts["battery_temp_C"], errors="coerce")
    pts["cpu_load"] = pd.to_numeric(pts["cpu_load"], errors="coerce")
    pts["cpu_freq_norm"] = pd.to_numeric(pts["cpu_freq_norm"], errors="coerce")
    pts["screen_on"] = pd.to_numeric(pts["screen_on"], errors="coerce").fillna(0).astype(int)
    pts["brightness_state"] = pd.to_numeric(pts["brightness_state"], errors="coerce")
    pts["net_type_code"] = pd.to_numeric(pts["net_type_code"], errors="coerce")

    pts["soc"] = pts["battery_level_pct"] / 100.0

    # cluster key per discharge interval
    pts["cluster"] = pts["episode_id"].astype(int).astype(str) + "_" + pts["interval_id"].astype(int).astype(str)

    # build r(t) within each interval
    pts["soc_lag"] = pts.groupby("cluster")["soc"].shift(DIFF_MIN)
    dt_h = float(DIFF_MIN) / 60.0
    pts["dsoc_dt"] = (pts["soc"] - pts["soc_lag"]) / dt_h  # 1/hour
    pts = pts.dropna(subset=["dsoc_dt"]).copy()
    pts = pts[pts["dsoc_dt"] < 0].copy()
    pts["r"] = -pts["dsoc_dt"]
    pts = pts[pts["r"] > 0].copy()
    pts["log_r"] = np.log(pts["r"])

    # covariates
    pts["scr"] = pts["screen_on"].astype(int)
    pts["bright"] = np.clip(pts["brightness_state"].to_numpy(float), 0.0, 1.0)
    pts.loc[pts["scr"] == 0, "bright"] = 0.0

    pts["net_wifi"] = (pts["net_type_code"] == 2).astype(int)
    # IMPORTANT: in the selected discharge-only points, net_type_code is always 1(mobile) or 2(wifi),
    # so net_mobile = 1 - net_wifi, and including both dummies with an intercept causes perfect collinearity.

    pts["T0"] = pts["battery_temp_C"] - TREF_C

    pts["cpu_x"] = pts["cpu_load"] * pts["cpu_freq_norm"]

    cols = [
        "time",
        "episode_id",
        "interval_id",
        "cluster",
        "log_r",
        "r",
        "cpu_load",
        "cpu_freq_norm",
        "cpu_x",
        "scr",
        "bright",
        "net_wifi",
        "T0",
    ]
    out = pts[cols].dropna().copy()
    return out


def design_matrix(panel: pd.DataFrame, *, include_interactions: bool) -> Tuple[np.ndarray, list[str]]:
    n = len(panel)
    const = np.ones(n)
    cpu_load = panel["cpu_load"].to_numpy(float)
    cpu_freq = panel["cpu_freq_norm"].to_numpy(float)
    cpu_x = panel["cpu_x"].to_numpy(float)
    scr = panel["scr"].to_numpy(float)
    bright = panel["bright"].to_numpy(float)
    net_wifi = panel["net_wifi"].to_numpy(float)
    T0 = panel["T0"].to_numpy(float)

    X = [const, cpu_load, cpu_freq, cpu_x, scr, bright, net_wifi, T0]
    names = ["const", "cpu_load", "cpu_freq", "cpu_x", "scr", "bright", "net_wifi", "T0"]

    if include_interactions:
        inter = [
            # CPU × Brightness
            ("cpu_x_scr", cpu_load * scr),
            ("cpu_x_bright", cpu_load * bright),
            # CPU × Network
            ("cpu_x_wifi", cpu_load * net_wifi),
            # CPU × Temp
            ("cpu_x_T0", cpu_load * T0),
            # Brightness × Network
            ("scr_x_wifi", scr * net_wifi),
            ("bright_x_wifi", bright * net_wifi),
            # Brightness × Temp
            ("scr_x_T0", scr * T0),
            ("bright_x_T0", bright * T0),
            # Network × Temp
            ("wifi_x_T0", net_wifi * T0),
        ]
        for nm, col in inter:
            X.append(col)
            names.append(nm)

    Xmat = np.column_stack(X)
    return Xmat, names


def write_params(names: list[str], beta: np.ndarray, cov_ols: np.ndarray, cov_hac: np.ndarray, out_csv: Path) -> None:
    se_ols = np.sqrt(np.clip(np.diag(cov_ols), 1e-18, np.inf))
    z_ols = beta / se_ols
    p_ols = z_pvalues(z_ols)

    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 1e-18, np.inf))
    z_hac = beta / se_hac
    p_hac = z_pvalues(z_hac)

    df = pd.DataFrame(
        {
            "name": names,
            "beta": beta,
            "se_ols": se_ols,
            "z_ols": z_ols,
            "p_ols": p_ols if p_ols is not None else np.nan,
            "se_hac": se_hac,
            "z_hac": z_hac,
            "p_hac": p_hac if p_hac is not None else np.nan,
        }
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")


def main() -> None:
    panel = build_rate_panel()
    panel.to_csv(OUT_DIR / "rate_panel_logr.csv", index=False, encoding="utf-8")

    # order for HAC within clusters
    panel = panel.sort_values(["cluster", "time"]).reset_index(drop=True)

    # main effects model
    y = panel["log_r"].to_numpy(float)
    X, names = design_matrix(panel, include_interactions=False)
    fit = ols_fit(X, y, names)
    cov_ols = ols_cov(X, fit.resid)
    cov_hac = hac_cov_by_cluster(panel, X, fit.resid, lag=HAC_LAG, cluster_col="cluster")
    write_params(names, fit.beta, cov_ols, cov_hac, OUT_DIR / "params_with_hac_main.csv")

    # group significance (Wald, HAC)
    name_to_idx = {n: i for i, n in enumerate(names)}
    tests = [
        wald_test(fit.beta, cov_hac, [name_to_idx["cpu_load"], name_to_idx["cpu_freq"], name_to_idx["cpu_x"]], "CPU_group"),
        wald_test(fit.beta, cov_hac, [name_to_idx["scr"], name_to_idx["bright"]], "Brightness_group"),
        wald_test(fit.beta, cov_hac, [name_to_idx["net_wifi"]], "Network_group(net_wifi vs mobile baseline)"),
        wald_test(fit.beta, cov_hac, [name_to_idx["T0"]], "Temperature_group"),
    ]
    pd.DataFrame(tests).to_csv(OUT_DIR / "wald_tests_main.csv", index=False, encoding="utf-8")

    # interactions model for independence tests
    X2, names2 = design_matrix(panel, include_interactions=True)
    fit2 = ols_fit(X2, y, names2)
    cov2_ols = ols_cov(X2, fit2.resid)
    cov2_hac = hac_cov_by_cluster(panel, X2, fit2.resid, lag=HAC_LAG, cluster_col="cluster")
    write_params(names2, fit2.beta, cov2_ols, cov2_hac, OUT_DIR / "params_with_hac_interactions.csv")

    nm2 = {n: i for i, n in enumerate(names2)}
    # interaction index sets by pair
    idx_cpu_bright = [nm2["cpu_x_scr"], nm2["cpu_x_bright"]]
    idx_cpu_net = [nm2["cpu_x_wifi"]]
    idx_cpu_temp = [nm2["cpu_x_T0"]]
    idx_bright_net = [nm2["scr_x_wifi"], nm2["bright_x_wifi"]]
    idx_bright_temp = [nm2["scr_x_T0"], nm2["bright_x_T0"]]
    idx_net_temp = [nm2["wifi_x_T0"]]
    idx_all = idx_cpu_bright + idx_cpu_net + idx_cpu_temp + idx_bright_net + idx_bright_temp + idx_net_temp

    itests = [
        wald_test(fit2.beta, cov2_hac, idx_all, "Independence_all_interactions=0"),
        wald_test(fit2.beta, cov2_hac, idx_cpu_bright, "Independence_CPUxBrightness"),
        wald_test(fit2.beta, cov2_hac, idx_cpu_net, "Independence_CPUxNetwork"),
        wald_test(fit2.beta, cov2_hac, idx_cpu_temp, "Independence_CPUxTemperature"),
        wald_test(fit2.beta, cov2_hac, idx_bright_net, "Independence_BrightnessxNetwork"),
        wald_test(fit2.beta, cov2_hac, idx_bright_temp, "Independence_BrightnessxTemperature"),
        wald_test(fit2.beta, cov2_hac, idx_net_temp, "Independence_NetworkxTemperature"),
    ]
    pd.DataFrame(itests).to_csv(OUT_DIR / "wald_tests_interactions.csv", index=False, encoding="utf-8")

    summary = {
        "n": int(len(panel)),
        "DIFF_MIN": DIFF_MIN,
        "HAC_LAG": HAC_LAG,
        "R2_in_sample_main": float(fit.r2),
        "RMSE_in_sample_main": float(fit.rmse),
        "R2_in_sample_interactions": float(fit2.r2),
        "RMSE_in_sample_interactions": float(fit2.rmse),
        "scipy_available": bool(STATS is not None),
    }
    (OUT_DIR / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

