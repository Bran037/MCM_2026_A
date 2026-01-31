"""
Train/Test split for test_1 segments to validate interpretability / generalization.

We have 9 discharge segments prepared by `prepare_test1_discharge_segments.py`:
  processed/test1/segments/test1_segments_1min.csv
  processed/test1/segments/test1_segments_chosen16.csv (summary; legacy filename)

This script:
  - sorts segments by start_time
  - fits ONE shared parameter set on the first 6 segments (train)
  - evaluates R^2 / RMSE on the last 3 segments (test) without refitting
  - writes plots and CSV summaries

Output:
  processed/test1/cv_train6_test3/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SEG_DIR = BASE_DIR / "processed" / "test1" / "segments"
IN_POINTS = SEG_DIR / "test1_segments_1min.csv"
IN_CHOSEN = SEG_DIR / "test1_segments_chosen16.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "cv_train6_test3"

N_TOTAL = 9
N_TRAIN = 6
N_TEST = N_TOTAL - N_TRAIN


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
    Predict SOC trajectory for one segment with fixed initial SOC = observed first point.
    p = [k0, I_idle, alpha_cpu, gamma, delta_scr, beta_scr, alpha_mob, alpha_wifi, beta_T]
    """
    k0, I_idle, a_cpu, gamma, d_scr, b_scr, a_mob, a_wifi, bT = p

    soc = df["soc"].to_numpy(float)
    cpu = df["cpu_load"].to_numpy(float)
    f = df["cpu_freq_norm"].to_numpy(float)
    T = df["battery_temp_C"].to_numpy(float)
    br = df["brightness_state"].to_numpy(float)
    net = df["net_type_code"].to_numpy(float)

    # 1-min grid (still compute dt for safety)
    dt = df["dt_min"].to_numpy(float)
    dt[~np.isfinite(dt)] = 1.0
    dt = np.clip(dt, 0.5, 2.0)

    # k_scr
    b = np.clip(br, 0.0, 1.0)
    k_scr = np.where(br < 0, 1.0, 1.0 + d_scr + b_scr * b)
    k_scr = np.clip(k_scr, 0.1, 50.0)

    # k_net
    k_net = np.ones_like(net, float)
    k_net = np.where(net == 1, a_mob, k_net)
    k_net = np.where(net == 2, a_wifi, k_net)
    k_net = np.clip(k_net, 0.1, 50.0)

    # k_T
    k_T = np.exp(bT * (T - t_ref))
    k_T = np.clip(k_T, 0.1, 50.0)

    # I_base
    f_pow = np.power(np.clip(f, 0.0, 1.0), gamma)
    I_base = I_idle + a_cpu * cpu * f_pow
    I_base = np.clip(I_base, 0.0, 5000.0)

    I_eff = I_base * k_scr * k_net * k_T

    # integrate
    soc0 = float(soc[0])
    ds = k0 * I_eff * dt
    soc_pred = soc0 - np.cumsum(ds)
    soc_pred = np.concatenate([[soc0], soc_pred[:-1]])
    return soc_pred


def _objective(all_segments: List[pd.DataFrame], p: np.ndarray) -> np.ndarray:
    # equal weight per segment
    res = []
    for seg in all_segments:
        y = seg["soc"].to_numpy(float)
        yhat = _model_predict_segment(seg, p)
        r = y - yhat
        w = 1.0 / np.sqrt(max(1.0, float(np.isfinite(r).sum())))
        res.append(r * w)
    return np.concatenate(res)


def _fit_shared_params(all_segments: List[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    p0 = np.array(
        [
            5.5e-6,  # k0
            30.0,  # I_idle (mA)
            400.0,  # alpha_cpu (mA)
            2.0,  # gamma
            0.8,  # delta_scr
            0.8,  # beta_scr
            1.5,  # alpha_mob
            1.1,  # alpha_wifi
            0.03,  # beta_T
        ],
        dtype=float,
    )

    # Same bounds as the 9-seg shared fit (avoid pathological inversions)
    lb = np.array([1e-7, 1.0, 0.0, 0.8, 0.0, 0.0, 1.0, 1.0, -0.2], dtype=float)
    ub = np.array([5e-5, 800.0, 5000.0, 4.0, 5.0, 5.0, 8.0, 8.0, 0.2], dtype=float)

    from scipy.optimize import least_squares  # type: ignore

    def fun(x: np.ndarray) -> np.ndarray:
        return _objective(all_segments, x)

    best = None
    best_cost = np.inf
    rng = np.random.default_rng(7)
    for _ in range(12):
        jitter = rng.normal(scale=[2e-6, 20, 250, 0.4, 0.4, 0.4, 0.3, 0.3, 0.02], size=p0.shape)
        x0 = np.clip(p0 + jitter, lb, ub)
        sol = least_squares(fun, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=2500)
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best = sol.x
    assert best is not None
    return best, {"method": "scipy.optimize.least_squares", "cost": float(best_cost)}


def _build_segment_frames(df: pd.DataFrame, seg_ids: List[int]) -> Dict[int, pd.DataFrame]:
    df = df[df["seg_id"].astype(int).isin(seg_ids)].copy()
    df = df.sort_values(["seg_id", "time"])
    segments: Dict[int, pd.DataFrame] = {}
    for seg_id, g in df.groupby("seg_id"):
        g = g.sort_values("time").copy()
        g["soc"] = pd.to_numeric(g["soc_pct"], errors="coerce") / 100.0
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
        segments[int(seg_id)] = g[keep].reset_index(drop=True)
    return segments


def _plot_concat(segments: Dict[int, pd.DataFrame], seg_ids: List[int], p: np.ndarray, title: str, out_png: Path) -> Dict[str, float]:
    import matplotlib.pyplot as plt

    gap = 10
    x0 = 0
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)

    y_all = []
    yhat_all = []
    for j, sid in enumerate(seg_ids):
        seg = segments[sid]
        y = seg["soc"].to_numpy(float) * 100.0
        yhat = _model_predict_segment(seg, p) * 100.0
        y_all.append(y)
        yhat_all.append(yhat)
        x = np.arange(len(y)) + x0
        ax.plot(x, y, color="#1f77b4", lw=1.4, label=f"SOC(%) observed ({len(seg_ids)} segments)" if j == 0 else None)
        ax.plot(x, yhat, color="#ff7f0e", lw=1.8, label="shared-params prediction" if j == 0 else None)
        ax.axvspan(x0, x0 + len(y) - 1, color="gray", alpha=0.06)
        x0 += len(y) + gap

    y_cat = np.concatenate(y_all)
    yhat_cat = np.concatenate(yhat_all)
    r2 = _r2(y_cat, yhat_cat)
    rmse = _rmse(y_cat, yhat_cat)

    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, title=f"R²={r2:.3f}, RMSE={rmse:.3f}")
    ax.set_xlabel("Concatenated time (min)")
    ax.set_ylabel("SOC(%)")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"r2": float(r2), "rmse": float(rmse)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_POINTS, parse_dates=["time"])
    chosen = pd.read_csv(IN_CHOSEN)
    # normalize columns (legacy merges may create *_x/*_y)
    if "start_time" not in chosen.columns:
        if "start_time_x" in chosen.columns:
            chosen = chosen.rename(columns={"start_time_x": "start_time"})
        elif "start_time_y" in chosen.columns:
            chosen = chosen.rename(columns={"start_time_y": "start_time"})
    if "end_time" not in chosen.columns:
        if "end_time_x" in chosen.columns:
            chosen = chosen.rename(columns={"end_time_x": "end_time"})
        elif "end_time_y" in chosen.columns:
            chosen = chosen.rename(columns={"end_time_y": "end_time"})
    chosen["start_time"] = pd.to_datetime(chosen["start_time"], errors="coerce")
    chosen["end_time"] = pd.to_datetime(chosen["end_time"], errors="coerce")

    meta = chosen[["seg_id", "start_time", "end_time"]].copy()
    meta["seg_id"] = meta["seg_id"].astype(int)
    meta = meta.sort_values("start_time").reset_index(drop=True)

    if len(meta) < N_TOTAL:
        raise RuntimeError(f"Need {N_TOTAL} segments in chosen file; got {len(meta)}.")
    meta = meta.iloc[:N_TOTAL].copy()

    train_ids = meta.iloc[:N_TRAIN]["seg_id"].astype(int).tolist()
    test_ids = meta.iloc[N_TRAIN:]["seg_id"].astype(int).tolist()

    segments = _build_segment_frames(df, seg_ids=train_ids + test_ids)
    train_segments = [segments[sid] for sid in train_ids]

    # fit only on train
    p, info = _fit_shared_params(train_segments)

    # save params
    names = ["k0", "I_idle", "alpha_cpu", "gamma", "delta_scr", "beta_scr", "alpha_mob", "alpha_wifi", "beta_T"]
    pd.DataFrame({"param": names, "value": p}).to_csv(OUT_DIR / "fit_params_train.csv", index=False)
    (OUT_DIR / "fit_info.json").write_text(pd.Series(info).to_json(), encoding="utf-8")

    # per-seg metrics (train/test)
    rows = []
    for sid in train_ids + test_ids:
        seg = segments[sid]
        y = seg["soc"].to_numpy(float) * 100.0
        yhat = _model_predict_segment(seg, p) * 100.0
        rows.append(
            {
                "split": "train" if sid in train_ids else "test",
                "seg_id": int(sid),
                "n": int(len(y)),
                "r2": _r2(y, yhat),
                "rmse": _rmse(y, yhat),
            }
        )
    df_seg = pd.DataFrame(rows).sort_values(["split", "r2"], ascending=[True, True])
    df_seg.to_csv(OUT_DIR / "metrics_by_segment.csv", index=False)

    # concat metrics
    def concat_metrics(seg_list: List[int]) -> Tuple[float, float]:
        y = []
        yh = []
        for sid in seg_list:
            seg = segments[sid]
            y.append(seg["soc"].to_numpy(float) * 100.0)
            yh.append(_model_predict_segment(seg, p) * 100.0)
        ycat = np.concatenate(y)
        yhcat = np.concatenate(yh)
        return _r2(ycat, yhcat), _rmse(ycat, yhcat)

    r2_tr, rmse_tr = concat_metrics(train_ids)
    r2_te, rmse_te = concat_metrics(test_ids)
    pd.DataFrame(
        [
            {"split": "train", "n_segments": len(train_ids), "r2_concat": r2_tr, "rmse_concat": rmse_tr},
            {"split": "test", "n_segments": len(test_ids), "r2_concat": r2_te, "rmse_concat": rmse_te},
        ]
    ).to_csv(OUT_DIR / "metrics_by_split.csv", index=False)

    # plots
    _plot_concat(
        segments,
        train_ids,
        p,
        title=f"test_1 train (first {N_TRAIN} segments), shared-params fit, segs={train_ids}",
        out_png=OUT_DIR / "figures" / "train_concat.png",
    )
    _plot_concat(
        segments,
        test_ids,
        p,
        title=f"test_1 test (last {N_TEST} segments), shared-params prediction, segs={test_ids}",
        out_png=OUT_DIR / "figures" / "test_concat.png",
    )

    print("[OK] wrote outputs to", OUT_DIR)
    print("[TRAIN] R2=", float(r2_tr), "RMSE=", float(rmse_tr))
    print("[TEST]  R2=", float(r2_te), "RMSE=", float(rmse_te))


if __name__ == "__main__":
    main()

