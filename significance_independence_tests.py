from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


IN_DIR = Path("processed") / "discharge"
OUT_DIR = IN_DIR

# Use the time-stamped panel built in validate_discharge_model.py
PANEL_PATH = IN_DIR / "rate_panel_logr_with_time.csv"

# Newey–West / HAC settings
HAC_LAG = 20  # minutes; can tune (e.g., 20~60). We'll start conservative.


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
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse_score(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else np.nan


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


def hac_cov_by_cluster(panel: pd.DataFrame, X: np.ndarray, resid: np.ndarray, lag: int) -> np.ndarray:
    """
    Newey–West HAC covariance with clustering by device_id (time series per device),
    Bartlett kernel, same lag for each device.

    cov = (X'X)^{-1} S (X'X)^{-1}
    S = sum_c sum_{l=-L..L} w_l * sum_t u_t u_{t-l} x_t x_{t-l}'
    """
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)

    # Ensure we can index X/resid by row order
    S = np.zeros((p, p), dtype=float)
    devs = panel["device_id"].astype(str).to_numpy()

    # row indices per device
    for dev in np.unique(devs):
        idx = np.where(devs == dev)[0]
        if len(idx) < 3:
            continue
        Xd = X[idx, :]
        ud = resid[idx]
        Td = len(idx)

        # lag 0
        Xu = Xd * ud[:, None]
        S += Xu.T @ Xu

        # positive lags
        L = min(lag, Td - 1)
        for l in range(1, L + 1):
            w = 1.0 - l / (L + 1.0)  # Bartlett
            Xu0 = Xd[l:, :] * ud[l:, None]
            XuL = Xd[:-l, :] * ud[:-l, None]
            # add both +l and -l (symmetry)
            S += w * (Xu0.T @ XuL + XuL.T @ Xu0)

    cov = xtx_inv @ S @ xtx_inv
    return cov


def z_pvalues(z: np.ndarray) -> Optional[np.ndarray]:
    if STATS is None:
        return None
    return 2.0 * STATS.norm.sf(np.abs(z))


def chi2_pvalue(stat: float, df: int) -> Optional[float]:
    if STATS is None:
        return None
    return float(STATS.chi2.sf(stat, df=df))


def design_matrix(panel: pd.DataFrame, devs: list[str], include_interactions: bool) -> Tuple[np.ndarray, list[str]]:
    n = len(panel)
    const = np.ones(n)
    scr = panel["scr"].to_numpy()
    net_wifi = panel["net_wifi"].to_numpy()
    net_mobile = panel["net_mobile"].to_numpy()
    T0 = panel["T0"].to_numpy()

    dev_to_idx = {d: i for i, d in enumerate(devs)}
    dummies = np.zeros((n, max(0, len(devs) - 1)))
    if len(devs) > 1:
        for i, dev in enumerate(panel["device_id"].astype(str).tolist()):
            k = dev_to_idx[dev]
            if k >= 1:
                dummies[i, k - 1] = 1.0

    X = [const, scr, net_wifi, net_mobile, T0]
    names = ["const", "scr", "net_wifi", "net_mobile", "T0"] + [f"dev_{d}" for d in devs[1:]]
    if dummies.shape[1] > 0:
        X.append(dummies)

    if include_interactions:
        inter = [
            ("scr_x_wifi", scr * net_wifi),
            ("scr_x_mob", scr * net_mobile),
            ("scr_x_T0", scr * T0),
            ("wifi_x_T0", net_wifi * T0),
            ("mob_x_T0", net_mobile * T0),
        ]
        for nm, col in inter:
            X.append(col)
            names.append(nm)

    # Flatten list into matrix
    cols = []
    for item in X:
        if item.ndim == 1:
            cols.append(item.reshape(-1, 1))
        else:
            cols.append(item)
    Xmat = np.hstack(cols)
    return Xmat, names


def wald_test(beta: np.ndarray, cov: np.ndarray, idx: list[int], label: str) -> dict:
    b = beta[idx]
    C = cov[np.ix_(idx, idx)]
    Cinv = np.linalg.pinv(C)
    stat = float(b.T @ Cinv @ b)
    p = chi2_pvalue(stat, df=len(idx))
    return {"test": label, "df": len(idx), "stat": stat, "pvalue": p if p is not None else np.nan}


def summarize_params(names: list[str], beta: np.ndarray, cov_ols: np.ndarray, cov_hac: np.ndarray) -> None:
    se_ols = np.sqrt(np.clip(np.diag(cov_ols), 1e-18, np.inf))
    t_ols = beta / se_ols
    # For large n, normal approx is ok; also works even if scipy absent (p -> NaN)
    p_ols = z_pvalues(t_ols)

    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 1e-18, np.inf))
    z_hac = beta / se_hac
    p_hac = z_pvalues(z_hac)

    df = pd.DataFrame(
        {
            "name": names,
            "beta": beta,
            "se_ols": se_ols,
            "z_ols": t_ols,
            "p_ols": p_ols if p_ols is not None else np.nan,
            "se_hac": se_hac,
            "z_hac": z_hac,
            "p_hac": p_hac if p_hac is not None else np.nan,
        }
    )
    df.to_csv(OUT_DIR / "params_with_hac.csv", index=False, encoding="utf-8")


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    # order for HAC by device then time
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["timestamp"]).sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    # dev list for dummies
    devs = sorted(panel["device_id"].astype(str).unique().tolist())

    y = panel["log_r"].to_numpy(dtype=float)

    # main model
    X_main, names_main = design_matrix(panel, devs=devs, include_interactions=False)
    fit_main = ols_fit(X_main, y, names_main)
    cov_main_ols = ols_cov(X_main, fit_main.resid)
    cov_main_hac = hac_cov_by_cluster(panel, X_main, fit_main.resid, lag=HAC_LAG)

    # interaction model
    X_int, names_int = design_matrix(panel, devs=devs, include_interactions=True)
    fit_int = ols_fit(X_int, y, names_int)
    cov_int_ols = ols_cov(X_int, fit_int.resid)
    cov_int_hac = hac_cov_by_cluster(panel, X_int, fit_int.resid, lag=HAC_LAG)

    # write model fit summary
    pd.DataFrame(
        [
            {"model": "log_main", "n": len(panel), "p": X_main.shape[1], "R2": fit_main.r2, "RMSE": fit_main.rmse, "hac_lag": HAC_LAG},
            {"model": "log_with_interactions", "n": len(panel), "p": X_int.shape[1], "R2": fit_int.r2, "RMSE": fit_int.rmse, "hac_lag": HAC_LAG},
        ]
    ).to_csv(OUT_DIR / "significance_model_fit_summary.csv", index=False, encoding="utf-8")

    # parameter table (interaction model, because it contains main+interaction)
    summarize_params(names_int, fit_int.beta, cov_int_ols, cov_int_hac)

    # independence tests: are interaction coefficients jointly zero?
    inter_names = ["scr_x_wifi", "scr_x_mob", "scr_x_T0", "wifi_x_T0", "mob_x_T0"]
    idx_inter = [names_int.index(nm) for nm in inter_names if nm in names_int]
    wald_ols = wald_test(fit_int.beta, cov_int_ols, idx_inter, label="Wald_interactions_OLS")
    wald_hac = wald_test(fit_int.beta, cov_int_hac, idx_inter, label="Wald_interactions_HAC")

    # also test screen/network/temp main effects jointly (excluding device dummies)
    main5 = ["scr", "net_wifi", "net_mobile", "T0"]
    idx_main = [names_int.index(nm) for nm in main5 if nm in names_int]
    wald_main_hac = wald_test(fit_int.beta, cov_int_hac, idx_main, label="Wald_main_effects_HAC")

    pd.DataFrame([wald_ols, wald_hac, wald_main_hac]).to_csv(OUT_DIR / "wald_tests.csv", index=False, encoding="utf-8")

    print("wrote", OUT_DIR / "params_with_hac.csv")
    print("wrote", OUT_DIR / "wald_tests.csv")
    print("wrote", OUT_DIR / "significance_model_fit_summary.csv")


if __name__ == "__main__":
    main()

