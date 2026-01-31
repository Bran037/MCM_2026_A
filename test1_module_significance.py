"""
Module-wise significance tests on test_1 using mechanistic log-decomposition.

Goal: test whether the 4 modules (CPU / Brightness / Network / Temperature) are significant
in explaining log r, using a regression that mirrors the multiplicative structure:

Model (discharge-only, within-interval differencing):
  r = -dSOC/dt  (1/hour), y = log r

From the fitted SOC model (episodes_fit/fit_params.csv), construct module terms:
  g_cpu  = log(I_idle + alpha_cpu * cpu_load * cpu_freq_norm^gamma)
  g_scr  = log(k_scr)  where k_scr = 1 (screen off) else 1+delta_scr+beta_scr*bright
  g_net  = log(k_net)  where k_net = alpha_mob (mobile) or alpha_wifi (wifi)
  g_temp = beta_T * (T-30)   (since log k_T = beta_T*(T-30))

Then run:
  y = c + b_cpu*g_cpu + b_scr*g_scr + b_net*g_net + b_temp*g_temp + eps

And optionally add coupling:
  + b_netT * (g_net * g_temp)   (proxy for network×temperature coupling)

Inference: OLS with HAC(Newey–West) covariance clustered by discharge interval.

Outputs:
  processed/test1/significance_modules/
    panel_logr_modules.csv
    params_with_hac_modules.csv
    wald_tests_modules.csv
    params_with_hac_modules_netT.csv
    wald_tests_modules_netT.csv
    model_summary.json
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
PARAMS_PATH = BASE_DIR / "processed" / "test1" / "episodes_fit" / "fit_params.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "significance_modules"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREF_C = 30.0
HAC_LAG = 20

# For "relative-independent extraction": use longer differencing windows to reduce SOC quantization noise.
# We will write outputs for each DIFF_MIN to separate files.
DIFF_MINS = [5, 15, 30]


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


def _load_params() -> dict:
    df = pd.read_csv(PARAMS_PATH)
    df["param"] = df["param"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    m = df.dropna(subset=["param", "value"]).set_index("param")["value"].to_dict()
    return {k: float(v) for k, v in m.items()}


def build_panel_modules(params: dict, *, diff_min: int) -> pd.DataFrame:
    pts = pd.read_csv(PTS_PATH, parse_dates=["time"]).sort_values(["episode_id", "interval_id", "time"])
    pts = pts[pts["is_discharge"] == 1].copy()

    pts["battery_level_pct"] = pd.to_numeric(pts["battery_level_pct"], errors="coerce")
    pts["soc"] = pts["battery_level_pct"] / 100.0
    pts["soc_lag"] = pts.groupby(["episode_id", "interval_id"])["soc"].shift(diff_min)
    dt_h = float(diff_min) / 60.0
    pts["dsoc_dt"] = (pts["soc"] - pts["soc_lag"]) / dt_h
    pts = pts.dropna(subset=["dsoc_dt"]).copy()
    pts = pts[pts["dsoc_dt"] < 0].copy()
    pts["r"] = -pts["dsoc_dt"]
    pts = pts[pts["r"] > 0].copy()
    pts["log_r"] = np.log(pts["r"])

    pts["cpu_load"] = pd.to_numeric(pts["cpu_load"], errors="coerce")
    pts["cpu_freq_norm"] = pd.to_numeric(pts["cpu_freq_norm"], errors="coerce")
    pts["battery_temp_C"] = pd.to_numeric(pts["battery_temp_C"], errors="coerce")
    pts["screen_on"] = pd.to_numeric(pts["screen_on"], errors="coerce").fillna(0).astype(int)
    pts["brightness_state"] = pd.to_numeric(pts["brightness_state"], errors="coerce")
    pts["net_type_code"] = pd.to_numeric(pts["net_type_code"], errors="coerce")

    # Build module terms from fitted parameters
    I_idle = params["I_idle"]
    alpha_cpu = params["alpha_cpu"]
    gamma = params["gamma"]
    delta_scr = params["delta_scr"]
    beta_scr = params["beta_scr"]
    alpha_mob = params["alpha_mob"]
    alpha_wifi = params["alpha_wifi"]
    beta_T = params["beta_T"]

    cpu = pts["cpu_load"].to_numpy(float)
    f = np.clip(pts["cpu_freq_norm"].to_numpy(float), 0.0, 1.0)
    I_base = I_idle + alpha_cpu * cpu * np.power(f, gamma)
    I_base = np.clip(I_base, 1e-6, np.inf)
    g_cpu = np.log(I_base)

    scr = pts["screen_on"].to_numpy(int)
    b = np.clip(pts["brightness_state"].to_numpy(float), 0.0, 1.0)
    k_scr = np.where(scr == 1, 1.0 + delta_scr + beta_scr * b, 1.0)
    k_scr = np.clip(k_scr, 1e-6, np.inf)
    g_scr = np.log(k_scr)

    net = pts["net_type_code"].to_numpy(float)
    k_net = np.ones_like(net, dtype=float)
    k_net = np.where(net == 1, alpha_mob, k_net)
    k_net = np.where(net == 2, alpha_wifi, k_net)
    k_net = np.clip(k_net, 1e-6, np.inf)
    g_net = np.log(k_net)

    T0 = pts["battery_temp_C"].to_numpy(float) - TREF_C
    g_temp = beta_T * T0  # this is already log k_T

    out = pd.DataFrame(
        {
            "time": pts["time"].to_numpy(),
            "episode_id": pts["episode_id"].astype(int).to_numpy(),
            "interval_id": pts["interval_id"].astype(int).to_numpy(),
            "cluster": (pts["episode_id"].astype(int).astype(str) + "_" + pts["interval_id"].astype(int).astype(str)).to_numpy(),
            "log_r": pts["log_r"].to_numpy(float),
            "g_cpu": g_cpu,
            "g_scr": g_scr,
            "g_net": g_net,
            "g_temp": g_temp,
            "T0": T0,
        }
    ).dropna()
    return out


def design_matrix(panel: pd.DataFrame, *, include_netT: bool) -> Tuple[np.ndarray, list[str]]:
    n = len(panel)
    const = np.ones(n)
    g_cpu = panel["g_cpu"].to_numpy(float)
    g_scr = panel["g_scr"].to_numpy(float)
    g_net = panel["g_net"].to_numpy(float)
    g_temp = panel["g_temp"].to_numpy(float)

    X = [const, g_cpu, g_scr, g_net, g_temp]
    names = ["const", "CPU_module", "Brightness_module", "Network_module", "Temperature_module"]

    if include_netT:
        X.append(g_net * g_temp)
        names.append("Network_x_Temperature")

    return np.column_stack(X), names


def write_params(names: list[str], beta: np.ndarray, cov_ols: np.ndarray, cov_hac: np.ndarray, out_csv: Path) -> None:
    se_ols = np.sqrt(np.clip(np.diag(cov_ols), 1e-18, np.inf))
    z_ols = beta / se_ols
    p_ols = z_pvalues(z_ols)

    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 1e-18, np.inf))
    z_hac = beta / se_hac
    p_hac = z_pvalues(z_hac)

    pd.DataFrame(
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
    ).to_csv(out_csv, index=False, encoding="utf-8")


def _run_one(diff_min: int) -> dict:
    params = _load_params()
    panel = build_panel_modules(params, diff_min=diff_min)
    panel.to_csv(OUT_DIR / f"panel_logr_modules_diff{diff_min}.csv", index=False, encoding="utf-8")

    panel = panel.sort_values(["cluster", "time"]).reset_index(drop=True)
    y = panel["log_r"].to_numpy(float)

    # base model
    X, names = design_matrix(panel, include_netT=False)
    fit = ols_fit(X, y, names)
    cov_ols = ols_cov(X, fit.resid)
    cov_hac = hac_cov_by_cluster(panel, X, fit.resid, lag=HAC_LAG, cluster_col="cluster")
    write_params(names, fit.beta, cov_ols, cov_hac, OUT_DIR / f"params_with_hac_modules_diff{diff_min}.csv")

    nm = {n: i for i, n in enumerate(names)}
    walds = [
        wald_test(fit.beta, cov_hac, [nm["CPU_module"]], "CPU_module"),
        wald_test(fit.beta, cov_hac, [nm["Brightness_module"]], "Brightness_module"),
        wald_test(fit.beta, cov_hac, [nm["Network_module"]], "Network_module"),
        wald_test(fit.beta, cov_hac, [nm["Temperature_module"]], "Temperature_module"),
        wald_test(
            fit.beta,
            cov_hac,
            [nm["CPU_module"], nm["Brightness_module"], nm["Network_module"], nm["Temperature_module"]],
            "All_4_modules",
        ),
    ]
    pd.DataFrame(walds).to_csv(OUT_DIR / f"wald_tests_modules_diff{diff_min}.csv", index=False, encoding="utf-8")

    # net-temp coupling model
    X2, names2 = design_matrix(panel, include_netT=True)
    fit2 = ols_fit(X2, y, names2)
    cov2_ols = ols_cov(X2, fit2.resid)
    cov2_hac = hac_cov_by_cluster(panel, X2, fit2.resid, lag=HAC_LAG, cluster_col="cluster")
    write_params(names2, fit2.beta, cov2_ols, cov2_hac, OUT_DIR / f"params_with_hac_modules_netT_diff{diff_min}.csv")

    nm2 = {n: i for i, n in enumerate(names2)}
    walds2 = [
        wald_test(fit2.beta, cov2_hac, [nm2["CPU_module"]], "CPU_module"),
        wald_test(fit2.beta, cov2_hac, [nm2["Brightness_module"]], "Brightness_module"),
        wald_test(fit2.beta, cov2_hac, [nm2["Network_module"]], "Network_module"),
        wald_test(fit2.beta, cov2_hac, [nm2["Temperature_module"]], "Temperature_module"),
        wald_test(fit2.beta, cov2_hac, [nm2["Network_x_Temperature"]], "Network_x_Temperature"),
        wald_test(fit2.beta, cov2_hac, [nm2["Network_module"], nm2["Temperature_module"], nm2["Network_x_Temperature"]], "Net_Temp_block"),
    ]
    pd.DataFrame(walds2).to_csv(OUT_DIR / f"wald_tests_modules_netT_diff{diff_min}.csv", index=False, encoding="utf-8")

    summary = {
        "n": int(len(panel)),
        "DIFF_MIN": diff_min,
        "HAC_LAG": HAC_LAG,
        "R2_base": float(fit.r2),
        "RMSE_base": float(fit.rmse),
        "R2_netT": float(fit2.r2),
        "RMSE_netT": float(fit2.rmse),
        "scipy_available": bool(STATS is not None),
        "params_source": str(PARAMS_PATH.as_posix()),
    }
    (OUT_DIR / f"model_summary_diff{diff_min}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    summaries = []
    for d in DIFF_MINS:
        summaries.append(_run_one(d))
    (OUT_DIR / "model_summary_all.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

