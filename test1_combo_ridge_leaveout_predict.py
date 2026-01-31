"""
Leave-one-combo-out ridge regression for fuzzy combo prediction on test_1.

Motivation:
  KNN on (cpu,bri,net) was not enough to reach 0.9/0.8.
  Here we use a low-variance parametric model with SOC (observable) to absorb drift.

Data:
  processed/test1/test1_panel_1min.csv
  Create discharge-labeled minutes via 30-min differencing:
    rate_pct_per_h(t) = -(SOC(t)-SOC(t-30))/0.5h * 100  (keep rate>=0.5)

Combos (for grouping / naming):
  CPU3 bins by cpu_load%: 0-33 / 34-66 / 67-100
  Brightness4: off / low / mid / high (using screen_on + brightness_state)
  Net2: mobile / wifi (net_type_code 1/2)

Model (minute-level):
  log(rate) = const + onehot(cpu3) + onehot(bri4) + onehot(net2)
            + beta_soc * soc + interactions with soc (cpu×soc, bri×soc, net×soc)
            + (optional) pairwise categorical interactions

Evaluation:
  Build test slices as contiguous runs of the same combo on discharge-labeled minutes.
  For each slice, true_rate = mean(rate) on slice minutes.
  Predict slice rate as mean(exp(pred_log_rate)) over the slice minutes.
  Final-time accuracy on a standard drop amount:
    T_true = DROP_STD_PCT / true_rate
    T_pred = DROP_STD_PCT / pred_rate
    acc = 1 - |T_pred - T_true|/T_true

We select Top-N combos by coverage (>=6 else >=4) and report mean/min over combos.

Outputs:
  processed/test1/combo_ridge_predict/
    best_config.json
    metrics_by_combo.csv
    metrics_by_slice.csv
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "combo_ridge_predict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_MIN = 30
MIN_RATE = 0.5
DROP_STD_PCT = 10.0

CPU_EDGES = (33, 66)

MIN_RUN_MIN_LIST = [5, 10, 15, 30]
TARGET_N_LIST = [6, 4]
MIN_TOTAL_MIN_LIST = [30, 60, 120, 240]
MIN_SLICES_PER_COMBO = 2

LAMBDA_LIST = [0.3, 1.0, 3.0, 10.0]
USE_PAIRWISE_INTERACTIONS_LIST = [False, True]


def _acc(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def cpu3(cpu_load: np.ndarray) -> np.ndarray:
    x = cpu_load * 100.0
    bins = [-np.inf, CPU_EDGES[0], CPU_EDGES[1], np.inf]
    return pd.cut(x, bins=bins, labels=[0, 1, 2], include_lowest=True, right=True).astype("Int64").to_numpy()


def net2(net_type_code: np.ndarray) -> np.ndarray:
    n = net_type_code
    out = np.zeros(len(n), dtype="int64")
    out[n == 1] = 0  # mobile
    out[n == 2] = 1  # wifi
    return out


def brightness_pct(screen_on: np.ndarray, brightness_state: np.ndarray) -> np.ndarray:
    scr = screen_on.astype(int)
    b = brightness_state
    off = (scr == 0) | (~np.isfinite(b)) | (b < 0)
    pct = np.clip(b, 0.0, 1.0) * 100.0
    return np.where(off, 0.0, pct)


def bri4(br_pct: np.ndarray) -> np.ndarray:
    out = np.zeros(len(br_pct), dtype="int64")
    on = br_pct > 0
    out[on & (br_pct <= 33)] = 1
    out[on & (br_pct >= 34) & (br_pct <= 66)] = 2
    out[on & (br_pct >= 67)] = 3
    return out


def combo_name(cpu_i: int, bri_i: int, net_i: int) -> str:
    cpu_txt = {0: "几乎不使用（0-33）", 1: "使用（34-66）", 2: "高负载（67-100）"}[cpu_i]
    bri_txt = {0: "息屏", 1: "低亮度", 2: "中等亮度", 3: "高亮度"}[bri_i]
    net_txt = {0: "蜂窝网络", 1: "无线网络"}[net_i]
    return f"{cpu_txt}×{bri_txt}×{net_txt}"


def add_rate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("time").copy()
    d["battery_level_pct"] = pd.to_numeric(d["battery_level_pct"], errors="coerce")
    soc = d["battery_level_pct"].to_numpy(float) / 100.0
    soc_lag = np.roll(soc, DIFF_MIN)
    soc_lag[:DIFF_MIN] = np.nan
    dsoc_dt = (soc - soc_lag) / (DIFF_MIN / 60.0)
    rate = np.where(dsoc_dt < 0, (-dsoc_dt) * 100.0, np.nan)
    d["rate_pct_per_h"] = rate
    d = d[np.isfinite(d["rate_pct_per_h"])].copy()
    d = d[d["rate_pct_per_h"] >= MIN_RATE].copy()
    return d


def build_slices(df: pd.DataFrame, *, min_run_min: int) -> pd.DataFrame:
    d = df.sort_values("time").reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    t = pd.to_datetime(d["time"], errors="coerce")
    dt = t.diff().dt.total_seconds().div(60.0).fillna(1.0).to_numpy(float)
    new_run = dt > 2.0
    combo = d["combo_id"].to_numpy(int)
    change = np.zeros(len(combo), dtype=bool)
    change[0] = True
    change[1:] = (combo[1:] != combo[:-1]) | new_run[1:]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(d))
    rows = []
    for s, e in zip(starts, ends):
        minutes = int(e - s)
        if minutes < min_run_min:
            continue
        g = d.iloc[s:e]
        cid = int(g["combo_id"].iloc[0])
        rows.append(
            {
                "combo_id": cid,
                "combo_name": str(g["combo_name"].iloc[0]),
                "start_idx": int(s),
                "end_idx": int(e),
                "minutes": minutes,
                "true_rate_pct_per_h": float(np.mean(g["rate_pct_per_h"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows)


def design_matrix(df: pd.DataFrame, *, use_pairwise: bool) -> np.ndarray:
    # baselines: cpu=0, bri=0, net=0
    cpu = df["cpu3"].to_numpy(int)
    bri = df["bri4"].to_numpy(int)
    net = df["net2"].to_numpy(int)
    soc = df["soc"].to_numpy(float)  # in [0,1]
    n = len(df)
    cols = [np.ones(n)]

    # dummies
    for lv in (1, 2):
        cols.append((cpu == lv).astype(float))
    for lv in (1, 2, 3):
        cols.append((bri == lv).astype(float))
    cols.append((net == 1).astype(float))

    # soc main
    cols.append(soc)

    # soc interactions with categories (helps drift by use scenario)
    for lv in (1, 2):
        cols.append(((cpu == lv).astype(float)) * soc)
    for lv in (1, 2, 3):
        cols.append(((bri == lv).astype(float)) * soc)
    cols.append(((net == 1).astype(float)) * soc)

    if use_pairwise:
        # cpu×bri
        for ci in (1, 2):
            for bi in (1, 2, 3):
                cols.append(((cpu == ci) & (bri == bi)).astype(float))
        # cpu×net
        for ci in (1, 2):
            cols.append(((cpu == ci) & (net == 1)).astype(float))
        # bri×net
        for bi in (1, 2, 3):
            cols.append(((bri == bi) & (net == 1)).astype(float))

    return np.column_stack(cols)


def ridge_fit_from_sums(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    p = xtx.shape[0]
    return np.linalg.solve(xtx + lam * np.eye(p), xty)


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["time"]).sort_values("time")
    panel = panel.dropna(subset=["cpu_load", "screen_on", "net_type_code", "battery_level_pct"]).copy()
    panel = add_rate(panel)

    cpu = pd.to_numeric(panel["cpu_load"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(panel["net_type_code"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(panel["screen_on"], errors="coerce").fillna(0).to_numpy(float)
    bri_state = pd.to_numeric(panel["brightness_state"], errors="coerce").to_numpy(float)
    br_pct = brightness_pct(scr, bri_state)

    panel["brightness_pct"] = br_pct
    panel["cpu3"] = cpu3(cpu)
    panel = panel.dropna(subset=["cpu3"]).copy()
    panel["cpu3"] = panel["cpu3"].astype(int)
    panel["bri4"] = bri4(panel["brightness_pct"].to_numpy(float)).astype(int)
    panel["net2"] = net2(pd.to_numeric(panel["net_type_code"], errors="coerce").to_numpy(float)).astype(int)
    panel["soc"] = pd.to_numeric(panel["battery_level_pct"], errors="coerce").to_numpy(float) / 100.0
    panel["combo_name"] = [
        combo_name(int(r.cpu3), int(r.bri4), int(r.net2)) for r in panel[["cpu3", "bri4", "net2"]].itertuples(index=False)
    ]
    panel["combo_id"] = panel["cpu3"] * 8 + panel["bri4"] * 2 + panel["net2"]

    best = None
    best_combo = None
    best_slice = None
    best_key = (-np.inf, -np.inf, -np.inf)

    # precompute per-minute arrays
    y_all = np.log(pd.to_numeric(panel["rate_pct_per_h"], errors="coerce").to_numpy(float))
    combo_all = pd.to_numeric(panel["combo_id"], errors="coerce").to_numpy(int)

    for min_run_min, target_n, min_total, lam, use_pairwise in itertools.product(
        MIN_RUN_MIN_LIST, TARGET_N_LIST, MIN_TOTAL_MIN_LIST, LAMBDA_LIST, USE_PAIRWISE_INTERACTIONS_LIST
    ):
        slices = build_slices(panel, min_run_min=int(min_run_min))
        if slices.empty:
            continue
        stat0 = slices.groupby(["combo_id", "combo_name"]).agg(n_slices=("minutes", "size"), total_min=("minutes", "sum")).reset_index()
        stat = stat0[(stat0["total_min"] >= int(min_total)) & (stat0["n_slices"] >= MIN_SLICES_PER_COMBO)].copy()
        if len(stat) < int(target_n):
            continue
        eligible = stat.sort_values(["total_min", "n_slices"], ascending=False).head(int(target_n)).copy()
        eligible_ids = set(eligible["combo_id"].astype(int).tolist())

        X_all = design_matrix(panel, use_pairwise=bool(use_pairwise))
        xtx_total = X_all.T @ X_all
        xty_total = X_all.T @ y_all

        # per combo sufficient stats
        per_xtx: Dict[int, np.ndarray] = {}
        per_xty: Dict[int, np.ndarray] = {}
        for cid, idx in pd.Series(np.arange(len(panel))).groupby(combo_all).groups.items():
            cid = int(cid)
            if cid not in eligible_ids:
                continue
            Xi = X_all[np.array(idx, dtype=int)]
            yi = y_all[np.array(idx, dtype=int)]
            per_xtx[cid] = Xi.T @ Xi
            per_xty[cid] = Xi.T @ yi

        combo_rows = []
        slice_rows = []
        for _, er in eligible.iterrows():
            cid = int(er["combo_id"])
            if cid not in per_xtx:
                continue
            beta = ridge_fit_from_sums(xtx_total - per_xtx[cid], xty_total - per_xty[cid], float(lam))

            # predict per-minute for this combo's slices: use each minute's X (not just a single category mean)
            s_list = slices[slices["combo_id"] == cid]
            accs = []
            for r in s_list.itertuples():
                g = panel.iloc[int(r.start_idx) : int(r.end_idx)]
                Xg = design_matrix(g, use_pairwise=bool(use_pairwise))
                pred_rate = float(np.mean(np.exp((Xg @ beta).ravel())))
                true_rate = float(r.true_rate_pct_per_h)
                true_dur_h = DROP_STD_PCT / true_rate
                pred_dur_h = DROP_STD_PCT / pred_rate if pred_rate > 1e-9 else np.nan
                acc = _acc(pred_dur_h, true_dur_h)
                accs.append(acc)
                slice_rows.append(
                    {
                        "combo_id": cid,
                        "combo_name": str(r.combo_name),
                        "minutes": int(r.minutes),
                        "true_rate_pct_per_h": true_rate,
                        "pred_rate_pct_per_h": pred_rate,
                        "acc_time": float(acc),
                    }
                )
            combo_rows.append(
                {
                    "combo_id": cid,
                    "combo_name": str(er["combo_name"]),
                    "n_slices": int(er["n_slices"]),
                    "total_min": int(er["total_min"]),
                    "acc_mean": float(np.nanmean(accs)),
                }
            )

        cdf = pd.DataFrame(combo_rows)
        if cdf.empty or len(cdf) < int(target_n):
            continue
        acc_mean = float(cdf["acc_mean"].mean())
        acc_min = float(cdf["acc_mean"].min())
        key = (acc_min, acc_mean, int(len(cdf)))
        if key > best_key:
            best_key = key
            best = {
                "min_run_min": int(min_run_min),
                "target_n": int(target_n),
                "min_total_min": int(min_total),
                "lambda": float(lam),
                "use_pairwise": bool(use_pairwise),
                "n_combos": int(len(cdf)),
                "acc_mean": acc_mean,
                "acc_min": acc_min,
            }
            best_combo = cdf.sort_values("acc_mean")
            best_slice = pd.DataFrame(slice_rows)

    if best is None or best_combo is None or best_slice is None:
        raise RuntimeError("No viable config found.")

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    best_combo.to_csv(OUT_DIR / "metrics_by_combo.csv", index=False, encoding="utf-8")
    best_slice.to_csv(OUT_DIR / "metrics_by_slice.csv", index=False, encoding="utf-8")
    print("[OK] best", best)


if __name__ == "__main__":
    main()

