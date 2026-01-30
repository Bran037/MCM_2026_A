from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_DIR = Path("processed") / "discharge"
OUT_DIR = IN_DIR
FIG_DIR = Path("figures") / "segment_soc_fits"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHOSEN = IN_DIR / "chosen_two_segments_per_device.csv"
RESAMPLE_FREQ = "1min"

# Target requirement (user)
TARGET_R2 = 0.95


@dataclass(frozen=True)
class FitOut:
    beta: np.ndarray
    r2: float
    rmse: float
    yhat: np.ndarray
    model_name: str
    n_params: int


def _ols(X: np.ndarray, y: np.ndarray) -> FitOut:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else np.nan
    return FitOut(beta=beta, r2=r2, rmse=rmse, yhat=yhat, model_name="ols", n_params=X.shape[1])


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> FitOut:
    # (X'X + lam I) beta = X'y
    p = X.shape[1]
    xtx = X.T @ X
    rhs = X.T @ y
    beta = np.linalg.solve(xtx + lam * np.eye(p), rhs)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else np.nan
    return FitOut(beta=beta, r2=r2, rmse=rmse, yhat=yhat, model_name=f"ridge_lam{lam:g}", n_params=p)


def load_device_file(device_id: str) -> pd.DataFrame:
    path = IN_DIR / f"{device_id}_discharge_1min.csv"
    df = pd.read_csv(path)
    # timestamp index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    else:
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    # normalize
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64")
    df["battery_level_pct"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    df["soc"] = pd.to_numeric(df.get("soc", df["battery_level_pct"] / 100.0), errors="coerce")
    if "battery_temp_C" in df.columns:
        df["battery_temp_C"] = pd.to_numeric(df["battery_temp_C"], errors="coerce")
    if "screen_on" in df.columns:
        df["screen_on"] = df["screen_on"].astype("boolean")
    if "network_type" in df.columns:
        df["network_type"] = df["network_type"].astype("string").str.lower()
    return df


def _segment_design(g: pd.DataFrame, level: int) -> tuple[np.ndarray, list[str]]:
    """
    Build design matrix to fit SOC% directly:
      y(t) = y0 - theta' * C(t)
    where C(t) are cumulative integrals of basis functions up to time t.

    level:
      0: y ~ const + t
      1: add cumulative screen/net/temp bases (additive discharge contributions)
      2: add interaction bases
      3: add coarse time-block baseline (per-hour blocks) to absorb unobserved load
    """
    # time axis
    t0 = g.index.min()
    dt_min = (g.index.to_series().diff().dt.total_seconds().fillna(60.0) / 60.0).to_numpy()
    # clamp dt to [1, 2] minutes (since we expect 1min grid but allow small anomalies)
    dt_min = np.clip(dt_min, 1.0, 2.0)
    t_min = ((g.index - t0).total_seconds().astype(float) / 60.0).to_numpy()

    y = g["battery_level_pct"].to_numpy(dtype=float)

    # base: const + t
    X = [np.ones(len(g)), t_min]
    names = ["const", "t_min"]

    if level >= 1:
        scr = g["screen_on"].fillna(False).astype(int).to_numpy()
        net = g["network_type"].fillna("none").astype(str).to_numpy()
        net_wifi = (net == "wi-fi").astype(int)
        net_mob = (net == "mobile").astype(int)
        T0 = (g["battery_temp_C"] - 30.0).fillna(0.0).to_numpy()

        # cumulative integrals (minutes) for additive discharge contributions
        C_scr = np.cumsum(scr * dt_min)
        C_wifi = np.cumsum(net_wifi * dt_min)
        C_mob = np.cumsum(net_mob * dt_min)
        C_T0 = np.cumsum(T0 * dt_min)

        X += [C_scr, C_wifi, C_mob, C_T0]
        names += ["C_scr", "C_wifi", "C_mob", "C_T0"]

    if level >= 2:
        # interactions (also cumulative)
        scr = g["screen_on"].fillna(False).astype(int).to_numpy()
        net = g["network_type"].fillna("none").astype(str).to_numpy()
        net_wifi = (net == "wi-fi").astype(int)
        net_mob = (net == "mobile").astype(int)
        T0 = (g["battery_temp_C"] - 30.0).fillna(0.0).to_numpy()

        C_scr_wifi = np.cumsum((scr * net_wifi) * dt_min)
        C_scr_mob = np.cumsum((scr * net_mob) * dt_min)
        C_scr_T0 = np.cumsum((scr * T0) * dt_min)
        C_mob_T0 = np.cumsum((net_mob * T0) * dt_min)
        C_wifi_T0 = np.cumsum((net_wifi * T0) * dt_min)

        X += [C_scr_wifi, C_scr_mob, C_scr_T0, C_wifi_T0, C_mob_T0]
        names += ["C_scr_wifi", "C_scr_mob", "C_scr_T0", "C_wifi_T0", "C_mob_T0"]

    if level >= 3:
        # coarse baseline blocks every 60 minutes (excluding the first block)
        block = (t_min // 60).astype(int)
        k = int(np.max(block))
        for b in range(1, k + 1):
            ind = (block == b).astype(int)
            C_blk = np.cumsum(ind * dt_min)
            X.append(C_blk)
            names.append(f"C_blk{b:02d}")

    Xmat = np.column_stack(X)
    return Xmat, names


def fit_one_segment(g: pd.DataFrame) -> FitOut:
    # drop remaining missing
    g = g.dropna(subset=["battery_level_pct"]).copy()
    if len(g) < 20:
        # too short, just do linear
        X, _ = _segment_design(g, level=0)
        return _ols(X, g["battery_level_pct"].to_numpy(dtype=float))

    y = g["battery_level_pct"].to_numpy(dtype=float)

    # escalate model complexity until R2 meets target
    best: FitOut | None = None
    best_names: str = ""

    for level in [0, 1, 2, 3]:
        X, names = _segment_design(g, level=level)
        res = _ols(X, y)
        res = FitOut(
            beta=res.beta,
            r2=res.r2,
            rmse=res.rmse,
            yhat=res.yhat,
            model_name=f"level{level}_ols",
            n_params=X.shape[1],
        )
        if best is None or (np.isfinite(res.r2) and res.r2 > best.r2):
            best = res
            best_names = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
        if np.isfinite(res.r2) and res.r2 >= TARGET_R2:
            return res

    assert best is not None
    # if still not met, try a mild ridge on the most complex level to stabilize
    X, _ = _segment_design(g, level=3)
    for lam in [1e-2, 1e-1, 1.0, 10.0]:
        resr = _ridge(X, y, lam=lam)
        resr = FitOut(beta=resr.beta, r2=resr.r2, rmse=resr.rmse, yhat=resr.yhat, model_name=resr.model_name, n_params=X.shape[1])
        if np.isfinite(resr.r2) and resr.r2 >= TARGET_R2:
            return resr
        if np.isfinite(resr.r2) and resr.r2 > best.r2:
            best = resr

    # return best available, even if < target (user will decide)
    return best


def plot_fit(device_id: str, segment_id: int, g: pd.DataFrame, fit: FitOut) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(g.index, g["battery_level_pct"], lw=1.2, label="SOC(%) (discharge segment, filled)")
    ax.plot(g.index, fit.yhat, lw=2.0, label=f"fit: {fit.model_name}, R2={fit.r2:.3f}, RMSE={fit.rmse:.3f}")
    ax.set_title(f"Device {device_id} seg {segment_id}: SOC fit (target R2≥{TARGET_R2:.2f})")
    ax.set_ylabel("SOC(%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = FIG_DIR / f"{device_id}_seg{segment_id}_soc_fit_r2_{fit.r2:.3f}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    chosen = pd.read_csv(CHOSEN)
    # group by device, keep 2 segments
    rows = []

    for device_id, sub in chosen.groupby("device_id", sort=True):
        df = load_device_file(str(device_id))
        for _, r in sub.iterrows():
            seg_id = int(r["segment_id"])
            g = df[df["segment_id"] == seg_id].copy()
            # keep within the recorded start/end window (already)
            g = g.dropna(subset=["battery_level_pct"])
            if g.empty:
                continue
            fit = fit_one_segment(g)
            out_fig = plot_fit(str(device_id), seg_id, g, fit)
            print("wrote", out_fig)
            rows.append(
                {
                    "device_id": str(device_id),
                    "segment_id": seg_id,
                    "n_points": int(len(g)),
                    "start_utc": str(g.index.min()),
                    "end_utc": str(g.index.max()),
                    "model": fit.model_name,
                    "n_params": int(fit.n_params),
                    "R2_soc": float(fit.r2),
                    "RMSE_soc_pct": float(fit.rmse),
                    "plot": str(out_fig.as_posix()),
                }
            )

    out_csv = OUT_DIR / "segment_soc_fit_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()

