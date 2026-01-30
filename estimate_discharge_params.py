from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


IN_DIR = Path("processed") / "discharge"
OUT_DIR = IN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREF_C = 30.0
NETS = ["none", "wi-fi", "mobile"]

# rate construction
DIFF_MIN = 5  # 5-min differencing to reduce SOC quantization


def _try_pvalue_from_t(t: np.ndarray, df: int) -> Optional[np.ndarray]:
    """
    Two-sided p-values for t-stats. Uses scipy if available; otherwise returns None.
    """
    try:
        from scipy import stats  # type: ignore

        return 2.0 * stats.t.sf(np.abs(t), df=df)
    except Exception:
        return None


@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: Optional[np.ndarray]
    r2: float
    adj_r2: float
    rmse: float
    n: int
    p_dim: int


def ols(X: np.ndarray, y: np.ndarray) -> OLSResult:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat

    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    df = max(1, n - p)
    sigma2 = rss / df
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, np.inf))
    tstat = beta / se
    pvals = _try_pvalue_from_t(tstat, df=df)

    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df if n > 1 else np.nan
    rmse = float(np.sqrt(rss / n)) if n > 0 else np.nan
    return OLSResult(beta=beta, se=se, t=tstat, p=pvals, r2=r2, adj_r2=adj_r2, rmse=rmse, n=n, p_dim=p)


def load_device_discharge_1min(path: Path, device_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # written with timestamp as index, but in CSV it is a column named "timestamp"
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    else:
        # best effort: first column might be timestamp
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    df["device_id"] = device_id
    # normalize types
    if "screen_on" in df.columns:
        df["screen_on"] = df["screen_on"].astype("boolean")
    if "network_type" in df.columns:
        df["network_type"] = df["network_type"].astype("string").str.lower()
    for c in ["battery_level_pct", "soc", "battery_temp_C", "battery_voltage_mV", "battery_current_mA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_rate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rate panel from discharge 1-min data:
    r(t) = - dSOC/dt (1/hour) via DIFF_MIN differencing.
    """
    d = df.copy()
    d = d.dropna(subset=["soc", "battery_temp_C", "screen_on", "network_type"])
    d = d[d["network_type"].isin(NETS)].copy()

    d["soc_lag"] = d["soc"].shift(DIFF_MIN)
    dt_h = float(DIFF_MIN) / 60.0
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h  # 1/hour

    # keep only discharging steps (SOC decreasing)
    d = d.dropna(subset=["dsoc_dt"])
    d = d[d["dsoc_dt"] < 0].copy()
    d["r"] = -d["dsoc_dt"]

    # features
    d["scr"] = d["screen_on"].astype(int)
    d["net_wifi"] = (d["network_type"] == "wi-fi").astype(int)
    d["net_mobile"] = (d["network_type"] == "mobile").astype(int)
    d["T0"] = d["battery_temp_C"] - TREF_C

    # drop non-positive r for log model
    d = d[d["r"] > 0].copy()
    d["log_r"] = np.log(d["r"])
    return d[["device_id", "log_r", "r", "scr", "net_wifi", "net_mobile", "T0"]].copy()


def fit_models(panel: pd.DataFrame) -> None:
    # device fixed effects (intercept dummies)
    devs = sorted(panel["device_id"].astype(str).unique().tolist())
    dev_to_idx = {d: i for i, d in enumerate(devs)}
    n = len(panel)

    # base X: const + scr + net_wifi + net_mobile + T0 + device_dummies(excluding first)
    const = np.ones(n)
    scr = panel["scr"].to_numpy()
    net_wifi = panel["net_wifi"].to_numpy()
    net_mobile = panel["net_mobile"].to_numpy()
    T0 = panel["T0"].to_numpy()

    dummies = np.zeros((n, max(0, len(devs) - 1)))
    if len(devs) > 1:
        for i, dev in enumerate(panel["device_id"].astype(str).tolist()):
            k = dev_to_idx[dev]
            if k >= 1:
                dummies[i, k - 1] = 1.0

    X_main = np.column_stack([const, scr, net_wifi, net_mobile, T0, dummies])
    y = panel["log_r"].to_numpy()

    res_main = ols(X_main, y)

    # interactions for "independence test"
    inter_cols = [
        ("scr_x_wifi", scr * net_wifi),
        ("scr_x_mob", scr * net_mobile),
        ("scr_x_T0", scr * T0),
        ("wifi_x_T0", net_wifi * T0),
        ("mob_x_T0", net_mobile * T0),
    ]
    X_int = np.column_stack([X_main] + [c for _, c in inter_cols])
    res_int = ols(X_int, y)

    # write pooled results
    def _write_params(path: Path, names: list[str], res: OLSResult) -> None:
        df = pd.DataFrame(
            {
                "name": names,
                "beta": res.beta,
                "se": res.se,
                "t": res.t,
                "p": res.p if res.p is not None else np.nan,
            }
        )
        df.to_csv(path, index=False, encoding="utf-8")

    names_main = ["const", "scr", "net_wifi", "net_mobile", "T0"] + [f"dev_{d}" for d in devs[1:]]
    names_int = names_main + [n for n, _ in inter_cols]

    _write_params(OUT_DIR / "params_pooled_log_main.csv", names_main, res_main)
    _write_params(OUT_DIR / "params_pooled_log_with_interactions.csv", names_int, res_int)

    pd.DataFrame(
        [
            {
                "model": "pooled_log_main",
                "n": res_main.n,
                "p": res_main.p_dim,
                "R2": res_main.r2,
                "adj_R2": res_main.adj_r2,
                "RMSE": res_main.rmse,
            },
            {
                "model": "pooled_log_with_interactions",
                "n": res_int.n,
                "p": res_int.p_dim,
                "R2": res_int.r2,
                "adj_R2": res_int.adj_r2,
                "RMSE": res_int.rmse,
            },
        ]
    ).to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    # per-device fits (no dummies)
    per_rows = []
    for dev in devs:
        sub = panel[panel["device_id"].astype(str) == dev].copy()
        if len(sub) < 50:
            continue
        n2 = len(sub)
        X = np.column_stack(
            [
                np.ones(n2),
                sub["scr"].to_numpy(),
                sub["net_wifi"].to_numpy(),
                sub["net_mobile"].to_numpy(),
                sub["T0"].to_numpy(),
            ]
        )
        y2 = sub["log_r"].to_numpy()
        res = ols(X, y2)
        per_rows.append(
            {
                "device_id": dev,
                "n": res.n,
                "R2": res.r2,
                "adj_R2": res.adj_r2,
                "RMSE": res.rmse,
                "const": res.beta[0],
                "scr": res.beta[1],
                "net_wifi": res.beta[2],
                "net_mobile": res.beta[3],
                "T0": res.beta[4],
            }
        )
    pd.DataFrame(per_rows).to_csv(OUT_DIR / "params_per_device_log_main.csv", index=False, encoding="utf-8")

    print("wrote", OUT_DIR / "params_pooled_log_main.csv")
    print("wrote", OUT_DIR / "params_pooled_log_with_interactions.csv")
    print("wrote", OUT_DIR / "params_per_device_log_main.csv")
    print("wrote", OUT_DIR / "model_comparison.csv")


def main() -> None:
    # load meta to answer "same vendor/cell?"
    meta_path = IN_DIR / "device_meta_summary.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    # build panel
    parts = []
    for p in sorted(IN_DIR.glob("*_discharge_1min.csv")):
        device_id = p.name.split("_")[0]
        df = load_device_discharge_1min(p, device_id)
        parts.append(build_rate_panel(df))
    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(OUT_DIR / "rate_panel_logr.csv", index=False, encoding="utf-8")
    print("panel rows:", len(panel))
    print("wrote", OUT_DIR / "rate_panel_logr.csv")

    fit_models(panel)

    # quick meta view
    if not meta.empty:
        meta.to_csv(OUT_DIR / "device_meta_summary.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()

