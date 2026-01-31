"""
Fit test_1 discharge segments with ONE shared parameter set.

We:
  1) Load processed/test1/segments/test1_segments_1min.csv and chosen list
  2) Build a concatenated time axis for groups (GROUP_SIZE segments each)
  3) Fit a continuous-time SOC model discretized on 1-min grid with shared params
  4) Output per-segment R^2 and plots per group

Model (1-min discretization):
  SOC_{k+1} = SOC_k - k0 * I_eff(t_k) * dt_min
where
  I_eff = (I_idle + alpha_cpu * cpu_load * cpu_freq_norm^gamma) * k_scr * k_net * k_T
  k_scr = 1 if brightness_state<0 else (1 + delta_scr + beta_scr * b)
  k_net = 1 (none), alpha_mob (mobile), alpha_wifi (wi-fi)
  k_T = exp(beta_T * (T - T_ref))

We do NOT include charging; segments were chosen to be pure discharge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SEG_DIR = BASE_DIR / "processed" / "test1" / "segments"
IN_POINTS = SEG_DIR / "test1_segments_1min.csv"
IN_CHOSEN = SEG_DIR / "test1_segments_chosen16.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "fit12"

# 3×3 = 9 segments, 3 groups
N_SEG = 9
GROUP_SIZE = 3


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3:
        return float("nan")
    ssr = float(np.sum((y[m] - yhat[m]) ** 2))
    sst = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ssr / sst if sst > 0 else float("nan")


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
    # align length (soc_pred is for steps; keep same length as observations)
    soc_pred = np.concatenate([[soc0], soc_pred[:-1]])
    return soc_pred


def _objective(all_segments: List[pd.DataFrame], p: np.ndarray) -> np.ndarray:
    # residuals stacked (weight by number of points; better for "big segment" fits)
    res = []
    for seg in all_segments:
        y = seg["soc"].to_numpy(float)
        yhat = _model_predict_segment(seg, p)
        res.append(y - yhat)
    return np.concatenate(res)


def _fit_shared_params(all_segments: List[pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    # Try scipy if available; otherwise fallback to simple random search + local coordinate refinement.
    p0 = np.array(
        [
            5.5e-6,  # k0 ~ 1/(60*Ceff)
            30.0,  # I_idle (mA)
            400.0,  # alpha_cpu (mA)
            2.0,  # gamma
            0.8,  # delta_scr
            0.8,  # beta_scr
            1.25,  # alpha_mob
            1.10,  # alpha_wifi
            0.03,  # beta_T
        ],
        dtype=float,
    )

    # Bounds tightened to avoid weird inversions (e.g. wifi < 1) unless data forces it.
    lb = np.array([1e-7, 1.0, 0.0, 0.8, 0.0, 0.0, 1.0, 1.0, -0.2], dtype=float)
    ub = np.array([5e-5, 800.0, 5000.0, 4.0, 5.0, 5.0, 8.0, 8.0, 0.2], dtype=float)

    try:
        from scipy.optimize import least_squares  # type: ignore

        def fun(x: np.ndarray) -> np.ndarray:
            return _objective(all_segments, x)

        best = None
        best_cost = np.inf
        # multi-start improves chances for high R^2
        rng = np.random.default_rng(7)
        for _ in range(16):
            jitter = rng.normal(scale=[2e-6, 20, 200, 0.4, 0.4, 0.4, 0.2, 0.2, 0.02], size=p0.shape)
            x0 = np.clip(p0 + jitter, lb, ub)
            sol = least_squares(fun, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=3500)
            if sol.cost < best_cost:
                best_cost = float(sol.cost)
                best = sol.x
        assert best is not None
        p = best
        info = {"method": "scipy.optimize.least_squares", "cost": float(best_cost)}
        return p, info
    except Exception as e:
        # fallback: random search (coarse)
        rng = np.random.default_rng(7)
        best_p = p0.copy()
        best_sse = float(np.sum(_objective(all_segments, best_p) ** 2))
        for _ in range(2000):
            cand = rng.uniform(lb, ub)
            sse = float(np.sum(_objective(all_segments, cand) ** 2))
            if sse < best_sse:
                best_sse, best_p = sse, cand
        return best_p, {"method": f"random_search_fallback: {type(e).__name__}", "sse": float(best_sse)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_POINTS, parse_dates=["time"])
    chosen = pd.read_csv(IN_CHOSEN)
    # tolerate merged-column names like start_time_x/start_time_y
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
    for c in ["start_time", "end_time"]:
        if c in chosen.columns:
            chosen[c] = pd.to_datetime(chosen[c], errors="coerce")
    chosen_ids = chosen["seg_id"].astype(int).tolist()
    if len(chosen_ids) < N_SEG:
        raise RuntimeError(f"Need {N_SEG} segments, got {len(chosen_ids)}.")
    chosen_ids = chosen_ids[:N_SEG]

    df = df[df["seg_id"].astype(int).isin(chosen_ids)].copy()
    df = df.sort_values(["seg_id", "time"])

    # Build per-segment frames with minimal columns
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

    # deterministic grouping: by start time (N_SEG / GROUP_SIZE groups)
    meta = chosen.set_index("seg_id").loc[chosen_ids].reset_index()
    meta = meta.sort_values("start_time").reset_index(drop=True)
    n_groups = int(np.ceil(len(meta) / GROUP_SIZE))
    groups: List[List[int]] = [
        meta.iloc[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]["seg_id"].astype(int).tolist() for i in range(n_groups)
    ]

    # Fit shared params across all segments
    all_segments = [segments[sid] for sid in meta["seg_id"].astype(int).tolist()]
    p, info = _fit_shared_params(all_segments)

    # Evaluate R^2 per segment and per group (group-level is the primary metric)
    rows = []
    group_rows = []
    for gi, group in enumerate(groups, start=1):
        y_all = []
        yhat_all = []
        for sid in group:
            seg = segments[sid]
            y = seg["soc"].to_numpy(float)
            yhat = _model_predict_segment(seg, p)
            rows.append({"group": gi, "seg_id": sid, "r2": _r2(y, yhat), "n": int(len(y))})
            y_all.append(y)
            yhat_all.append(yhat)
        group_rows.append({"group": gi, "r2_concat": _r2(np.concatenate(y_all), np.concatenate(yhat_all))})

    df_r2 = pd.DataFrame(rows).sort_values(["group", "r2"], ascending=[True, True])
    df_group = pd.DataFrame(group_rows)

    # Save params
    names = ["k0", "I_idle", "alpha_cpu", "gamma", "delta_scr", "beta_scr", "alpha_mob", "alpha_wifi", "beta_T"]
    df_params = pd.DataFrame({"param": names, "value": p})
    df_params.to_csv(OUT_DIR / "fit_params.csv", index=False)
    df_r2.to_csv(OUT_DIR / "r2_by_segment.csv", index=False)
    df_group.to_csv(OUT_DIR / "r2_by_group.csv", index=False)
    (OUT_DIR / "fit_info.json").write_text(pd.Series(info).to_json(), encoding="utf-8")

    # Plots per group (concatenated time axis): match the main-dataset figures style
    # (blue observed, orange fit; legend contains R²+RMSE; no per-segment text labels).
    import matplotlib.pyplot as plt

    for gi, group in enumerate(groups, start=1):
        gap = 10  # minutes between segments
        x0 = 0
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1)
        # group-level r2
        y_all = []
        yhat_all = []
        for j, sid in enumerate(group):
            seg = segments[sid]
            y = seg["soc"].to_numpy(float) * 100.0
            yhat = _model_predict_segment(seg, p) * 100.0
            y_all.append(y)
            yhat_all.append(yhat)
            x = np.arange(len(y)) + x0
            ax.plot(x, y, color="#1f77b4", lw=1.4, label=f"SOC(%) observed ({len(group)} segments)" if j == 0 else None)
            ax.plot(
                x,
                yhat,
                color="#ff7f0e",
                lw=1.8,
                label="shared-params fit (level1)" if j == 0 else None,
            )
            ax.axvspan(x0, x0 + len(y) - 1, color="gray", alpha=0.06)
            x0 += len(y) + gap
        y_cat = np.concatenate(y_all)
        yhat_cat = np.concatenate(yhat_all)
        r2g = _r2(y_cat, yhat_cat)
        rmse = float(np.sqrt(np.mean((y_cat - yhat_cat) ** 2)))
        segs_txt = ",".join(str(s) for s in group)
        ax.set_title(f"test_1: group {gi} ({len(group)} discharge segments), shared parameters, segs=[{segs_txt}]")
        ax.legend(loc="best", framealpha=0.9, title=f"R²={r2g:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel("Concatenated time (min)")
        ax.set_ylabel("SOC(%)")
        ax.grid(True, alpha=0.25)
        out = OUT_DIR / "figures" / f"group_{gi}_soc_fit.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("[OK] wrote outputs to", OUT_DIR)
    print("[INFO]", info)
    print("[R2] worst segment:", float(df_r2["r2"].min()))


if __name__ == "__main__":
    main()

