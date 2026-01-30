from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_DIR = Path("processed") / "discharge"
OUT_DIR = IN_DIR
FIG_DIR = Path("figures") / "diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TREF_C = 30.0
NETS = ["none", "wi-fi", "mobile"]
DIFF_MIN = 5
SEED = 42


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


def ols_fit(X: np.ndarray, y: np.ndarray, names: list[str]) -> Fit:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else np.nan
    return Fit(beta=beta, yhat=yhat, resid=resid, r2=r2, rmse=rmse, names=names)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse_score(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else np.nan


def load_discharge_1min(device_id: str) -> pd.DataFrame:
    p = IN_DIR / f"{device_id}_discharge_1min.csv"
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    else:
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    df["device_id"] = device_id
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64")
    df["soc"] = pd.to_numeric(df.get("soc", df.get("battery_level_pct", np.nan) / 100.0), errors="coerce")
    df["battery_level_pct"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    if "battery_temp_C" in df.columns:
        df["battery_temp_C"] = pd.to_numeric(df["battery_temp_C"], errors="coerce")
    if "screen_on" in df.columns:
        df["screen_on"] = df["screen_on"].astype("boolean")
    if "network_type" in df.columns:
        df["network_type"] = df["network_type"].astype("string").str.lower()
    return df


def build_rate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct log r panel with segment_id preserved.
    r(t) = -dSOC/dt (1/hour), using DIFF_MIN differencing within each segment.
    """
    d = df.copy()
    d = d.dropna(subset=["soc", "battery_temp_C", "screen_on", "network_type", "segment_id"])
    d = d[d["network_type"].isin(NETS)].copy()

    # differencing within each segment to avoid cross-boundary artifacts
    d["soc_lag"] = d.groupby("segment_id")["soc"].shift(DIFF_MIN)
    dt_h = float(DIFF_MIN) / 60.0
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h  # 1/hour
    d = d.dropna(subset=["dsoc_dt"])
    d = d[d["dsoc_dt"] < 0].copy()
    d["r"] = -d["dsoc_dt"]
    d = d[d["r"] > 0].copy()
    d["log_r"] = np.log(d["r"])

    d["scr"] = d["screen_on"].astype(int)
    d["net_wifi"] = (d["network_type"] == "wi-fi").astype(int)
    d["net_mobile"] = (d["network_type"] == "mobile").astype(int)
    d["T0"] = d["battery_temp_C"] - TREF_C

    return d[["device_id", "segment_id", "log_r", "r", "scr", "net_wifi", "net_mobile", "T0"]].copy()


def design_matrix(panel: pd.DataFrame, *, devs: list[str]) -> tuple[np.ndarray, list[str]]:
    """const + scr + net_wifi + net_mobile + T0 + device dummies(excluding first)."""
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

    X = np.column_stack([const, scr, net_wifi, net_mobile, T0, dummies])
    names = ["const", "scr", "net_wifi", "net_mobile", "T0"] + [f"dev_{d}" for d in devs[1:]]
    return X, names


def durbin_watson(resid: np.ndarray) -> float:
    if len(resid) < 3:
        return np.nan
    num = float(np.sum(np.diff(resid) ** 2))
    den = float(np.sum(resid**2))
    return num / den if den > 0 else np.nan


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 2:
        return np.full(nlags + 1, np.nan)
    x = x - np.mean(x)
    denom = np.sum(x * x)
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float(np.sum(x[:-k] * x[k:]) / denom) if denom > 0 else np.nan)
    return np.array(out)


def ljung_box_pvalue(resid: np.ndarray, lags: int = 20) -> float:
    if STATS is None:
        return np.nan
    r = acf(resid, nlags=lags)[1:]  # drop lag0
    n = int(np.sum(np.isfinite(resid)))
    if n <= lags + 1 or not np.all(np.isfinite(r)):
        return np.nan
    Q = n * (n + 2) * float(np.sum((r**2) / (n - np.arange(1, lags + 1))))
    # chi-square with dof=lags
    return float(STATS.chi2.sf(Q, df=lags))


def jarque_bera_pvalue(resid: np.ndarray) -> float:
    if STATS is None:
        return np.nan
    x = resid[np.isfinite(resid)]
    if len(x) < 20:
        return np.nan
    return float(STATS.jarque_bera(x).pvalue)


def cv_time_split(panel: pd.DataFrame, devs: list[str]) -> pd.DataFrame:
    """
    Per device: first 70% time as train, last 30% as test.
    Fit pooled model on all train points (device dummies allowed), evaluate on held-out time points.
    """
    rows = []
    parts_train = []
    parts_test = []
    for dev in devs:
        d = panel[panel["device_id"] == dev].sort_values("timestamp") if "timestamp" in panel.columns else panel[panel["device_id"] == dev]
        if len(d) < 200:
            continue
        cut = int(len(d) * 0.7)
        parts_train.append(d.iloc[:cut])
        parts_test.append(d.iloc[cut:])

    train = pd.concat(parts_train, ignore_index=True)
    test = pd.concat(parts_test, ignore_index=True)

    Xtr, names = design_matrix(train, devs=devs)
    ytr = train["log_r"].to_numpy()
    fit = ols_fit(Xtr, ytr, names)

    Xte, _ = design_matrix(test, devs=devs)
    yte = test["log_r"].to_numpy()
    yhat = Xte @ fit.beta

    # overall
    rows.append(
        {
            "split": "time_70_30_overall",
            "device_id": "ALL",
            "n_train": len(train),
            "n_test": len(test),
            "R2_test": r2_score(yte, yhat),
            "RMSE_test": rmse_score(yte, yhat),
        }
    )
    # per device
    for dev in devs:
        te = test[test["device_id"] == dev]
        if len(te) < 50:
            continue
        Xd, _ = design_matrix(te, devs=devs)
        yd = te["log_r"].to_numpy()
        yhd = Xd @ fit.beta
        rows.append(
            {
                "split": "time_70_30",
                "device_id": dev,
                "n_train": int((train["device_id"] == dev).sum()),
                "n_test": int((test["device_id"] == dev).sum()),
                "R2_test": r2_score(yd, yhd),
                "RMSE_test": rmse_score(yd, yhd),
            }
        )
    return pd.DataFrame(rows)


def cv_four_segment_holdout(panel: pd.DataFrame, devs: list[str]) -> pd.DataFrame:
    """
    Proper segment-level validation on the *same four segments* we selected for the hard fit:
    for each device, take its 4 selected discharge segments, order them by segment start time,
    then evaluate:
      - train on first 3 segments, test on last 1
      - train on first 2 segments, test on last 2

    This matches: "前三段/两段学习，最后一段/两段测试".
    """
    four_path = IN_DIR / "four_segment_shared_fit_summary.csv"
    if not four_path.exists():
        return pd.DataFrame()

    four = pd.read_csv(four_path)
    # ensure timestamps are datetime for start-time ordering
    panel2 = panel.copy()
    panel2["timestamp"] = pd.to_datetime(panel2["timestamp"], utc=True, errors="coerce")
    panel2 = panel2.dropna(subset=["timestamp", "segment_id"]).copy()

    rows = []

    def _run_split(train_parts: list[pd.DataFrame], test_parts: list[pd.DataFrame], split_name: str) -> None:
        if not train_parts or not test_parts:
            return
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        Xtr, names = design_matrix(train, devs=devs)
        ytr = train["log_r"].to_numpy()
        fit = ols_fit(Xtr, ytr, names)

        Xte, _ = design_matrix(test, devs=devs)
        yte = test["log_r"].to_numpy()
        yhat = Xte @ fit.beta

        rows.append(
            {
                "split": f"{split_name}_overall",
                "device_id": "ALL",
                "n_train": len(train),
                "n_test": len(test),
                "R2_test": r2_score(yte, yhat),
                "RMSE_test": rmse_score(yte, yhat),
            }
        )

        for dev in devs:
            te = test[test["device_id"] == dev]
            if len(te) < 30:
                continue
            Xd, _ = design_matrix(te, devs=devs)
            yd = te["log_r"].to_numpy()
            yhd = Xd @ fit.beta
            rows.append(
                {
                    "split": split_name,
                    "device_id": dev,
                    "n_train": int((train["device_id"] == dev).sum()),
                    "n_test": int((test["device_id"] == dev).sum()),
                    "R2_test": r2_score(yd, yhd),
                    "RMSE_test": rmse_score(yd, yhd),
                }
            )

    # Build per-device train/test sets using the chosen 4 segments
    train_3, test_1 = [], []
    train_2, test_2 = [], []

    for dev in devs:
        row = four[four["device_id"].astype(str) == str(dev)]
        if row.empty:
            continue
        seg_ids = str(row["segment_ids"].iloc[0]).split(";")
        seg_ids = [int(s) for s in seg_ids if s.strip() != ""]
        if len(seg_ids) < 3:
            continue

        d = panel2[panel2["device_id"].astype(str) == str(dev)].copy()
        d = d[d["segment_id"].isin(seg_ids)].copy()
        if d.empty:
            continue

        # order segments by start time (first timestamp in that segment)
        seg_start = d.groupby("segment_id")["timestamp"].min().sort_values()
        ordered = seg_start.index.astype(int).tolist()
        if len(ordered) < 3:
            continue

        # split 3/1
        tr_ids_31 = set(ordered[:-1])
        te_ids_31 = {ordered[-1]}
        train_3.append(d[d["segment_id"].isin(tr_ids_31)])
        test_1.append(d[d["segment_id"].isin(te_ids_31)])

        # split 2/2 (if possible)
        if len(ordered) >= 4:
            tr_ids_22 = set(ordered[:2])
            te_ids_22 = set(ordered[2:4])
            train_2.append(d[d["segment_id"].isin(tr_ids_22)])
            test_2.append(d[d["segment_id"].isin(te_ids_22)])

    _run_split(train_3, test_1, "fourseg_train3_test1")
    _run_split(train_2, test_2, "fourseg_train2_test2")

    return pd.DataFrame(rows)

def cv_segment_split(panel: pd.DataFrame, devs: list[str]) -> pd.DataFrame:
    """
    Per device: random 80% segments train, 20% segments test.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    train_parts = []
    test_parts = []

    for dev in devs:
        d = panel[panel["device_id"] == dev].copy()
        segs = d["segment_id"].dropna().unique().tolist()
        if len(segs) < 6:
            continue
        rng.shuffle(segs)
        k = max(1, int(0.8 * len(segs)))
        seg_train = set(segs[:k])
        seg_test = set(segs[k:])
        train_parts.append(d[d["segment_id"].isin(seg_train)])
        test_parts.append(d[d["segment_id"].isin(seg_test)])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    Xtr, names = design_matrix(train, devs=devs)
    ytr = train["log_r"].to_numpy()
    fit = ols_fit(Xtr, ytr, names)

    Xte, _ = design_matrix(test, devs=devs)
    yte = test["log_r"].to_numpy()
    yhat = Xte @ fit.beta

    rows.append(
        {
            "split": "segment_80_20_overall",
            "device_id": "ALL",
            "n_train": len(train),
            "n_test": len(test),
            "R2_test": r2_score(yte, yhat),
            "RMSE_test": rmse_score(yte, yhat),
        }
    )
    for dev in devs:
        te = test[test["device_id"] == dev]
        if len(te) < 50:
            continue
        Xd, _ = design_matrix(te, devs=devs)
        yd = te["log_r"].to_numpy()
        yhd = Xd @ fit.beta
        rows.append(
            {
                "split": "segment_80_20",
                "device_id": dev,
                "n_train": int((train["device_id"] == dev).sum()),
                "n_test": int((test["device_id"] == dev).sum()),
                "R2_test": r2_score(yd, yhd),
                "RMSE_test": rmse_score(yd, yhd),
            }
        )
    return pd.DataFrame(rows)


def residual_plots(y: np.ndarray, yhat: np.ndarray, resid: np.ndarray) -> None:
    # residual vs fitted
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(yhat, resid, s=6, alpha=0.25)
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_title("Residual vs fitted (log r)")
    ax.set_xlabel("fitted log r")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_vs_fitted.png", dpi=160)
    plt.close(fig)

    # hist
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(resid[np.isfinite(resid)], bins=60, alpha=0.85)
    ax.set_title("Residual histogram (log r)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_hist.png", dpi=160)
    plt.close(fig)

    # QQ plot (if scipy available)
    if STATS is not None:
        x = resid[np.isfinite(resid)]
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
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
    ax.set_xlabel("lag")
    ax.set_ylabel("acf")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_acf.png", dpi=160)
    plt.close(fig)


def main() -> None:
    meta = pd.read_csv(IN_DIR / "device_meta_summary.csv")
    devs = meta["device_id"].astype(str).tolist()

    # build panel with timestamps (for time CV); include timestamp by carrying index to column
    parts = []
    for dev in devs:
        df = load_discharge_1min(dev)
        p = build_rate_panel(df)
        # attach timestamp to support time-split CV and ordering
        # align timestamps: p is built from df with same index; rebuild by recomputing and carrying index
        # easiest: recompute directly on a merged copy with timestamp column
        # here, we approximate by using df.index after dropna, matching by index
        # -> to be robust, rebuild with timestamp as column before selecting final fields
        # We'll just attach timestamp from df after filtering similarly:
        df2 = df.copy()
        df2 = df2.dropna(subset=["soc", "battery_temp_C", "screen_on", "network_type", "segment_id"])
        df2 = df2[df2["network_type"].isin(NETS)].copy()
        df2["soc_lag"] = df2.groupby("segment_id")["soc"].shift(DIFF_MIN)
        dt_h = float(DIFF_MIN) / 60.0
        df2["dsoc_dt"] = (df2["soc"] - df2["soc_lag"]) / dt_h
        df2 = df2.dropna(subset=["dsoc_dt"])
        df2 = df2[df2["dsoc_dt"] < 0].copy()
        df2["r"] = -df2["dsoc_dt"]
        df2 = df2[df2["r"] > 0].copy()
        df2["log_r"] = np.log(df2["r"])
        df2["scr"] = df2["screen_on"].astype(int)
        df2["net_wifi"] = (df2["network_type"] == "wi-fi").astype(int)
        df2["net_mobile"] = (df2["network_type"] == "mobile").astype(int)
        df2["T0"] = df2["battery_temp_C"] - TREF_C
        df2 = df2[["device_id", "segment_id", "log_r", "r", "scr", "net_wifi", "net_mobile", "T0"]].copy()
        df2["timestamp"] = df2.index
        parts.append(df2.reset_index(drop=True))

    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(OUT_DIR / "rate_panel_logr_with_time.csv", index=False, encoding="utf-8")

    # fit pooled model on full data
    X, names = design_matrix(panel, devs=devs)
    y = panel["log_r"].to_numpy()
    fit = ols_fit(X, y, names)

    # residual diagnostics
    # order by (device_id, timestamp) for DW and LB to be more meaningful
    panel_ord = panel.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    Xo, _ = design_matrix(panel_ord, devs=devs)
    yo = panel_ord["log_r"].to_numpy()
    yhat_o = Xo @ fit.beta
    resid_o = yo - yhat_o

    diag = {
        "n": int(len(panel)),
        "R2_in_sample": float(fit.r2),
        "RMSE_in_sample": float(fit.rmse),
        "durbin_watson": float(durbin_watson(resid_o)),
        "ljung_box_pvalue_lag20": float(ljung_box_pvalue(resid_o, lags=20)),
        "jarque_bera_pvalue": float(jarque_bera_pvalue(resid_o)),
    }
    pd.DataFrame([diag]).to_csv(OUT_DIR / "residual_diagnostics.csv", index=False, encoding="utf-8")

    residual_plots(yo, yhat_o, resid_o)

    # cross validation
    cv2 = cv_segment_split(panel, devs=devs)
    cv2.to_csv(OUT_DIR / "cv_segment_split.csv", index=False, encoding="utf-8")
    cv3 = cv_four_segment_holdout(panel_ord, devs=devs)
    cv3.to_csv(OUT_DIR / "cv_four_segment_holdout.csv", index=False, encoding="utf-8")

    print("wrote", OUT_DIR / "rate_panel_logr_with_time.csv")
    print("wrote", OUT_DIR / "residual_diagnostics.csv")
    print("wrote", OUT_DIR / "cv_segment_split.csv")
    print("wrote", OUT_DIR / "cv_four_segment_holdout.csv")
    print("wrote", FIG_DIR / "residual_vs_fitted.png")
    print("wrote", FIG_DIR / "residual_hist.png")
    print("wrote", FIG_DIR / "residual_acf.png")
    if (FIG_DIR / "residual_qq.png").exists():
        print("wrote", FIG_DIR / "residual_qq.png")


if __name__ == "__main__":
    main()

