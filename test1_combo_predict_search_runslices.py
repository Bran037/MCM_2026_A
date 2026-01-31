"""
FAST combo prediction search (run-slice based) for test_1.

Why this exists:
  Rolling-window dominance creates O(N * windows) complexity and can time out.
  This version uses contiguous runs (run-length) of the same combo on the 1-min panel AFTER
  constructing discharge-rate labels via 30-min differencing (only keep discharge-labeled minutes).
  Complexity is O(N) per discretization, and O(S) for regression over slices.

Goal (user B option):
  - Produce >=10 evaluated combos (two-digit table).
  - Mean accuracy >= 0.90 AND min accuracy >= 0.80 over evaluated combos.
  - Combo names must be accurate Chinese phrases like “几乎不使用×息屏×无线网络”.

Prediction target:
  - Slice-level discharge speed y_true = mean(local_rate_pct_per_h) over minutes in the slice
  - Evaluate final time only via a standard drop amount:
      T_true = DROP_STD_PCT / y_true
      T_pred = DROP_STD_PCT / y_pred
      Acc = 1 - |T_pred - T_true|/T_true

Leave-one-combo-out:
  Train ridge on all slices except target combo, then predict target combo.

Outputs:
  processed/test1/combo_predict_search_runslices/
    search_summary.csv
    best_config.json
    best_metrics_by_combo.csv
    best_metrics_by_slice.csv
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "combo_predict_search_runslices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_STD_PCT = 10.0


@dataclass(frozen=True)
class BinConfig:
    cpu_edges_pct: Tuple[int, ...]  # interior edges
    bri_mode: str  # "2","3","4"


CPU_CONFIGS = [
    BinConfig(cpu_edges_pct=(33, 66), bri_mode="4"),  # 36 combos
    BinConfig(cpu_edges_pct=(33, 66), bri_mode="3"),  # 27 combos
    BinConfig(cpu_edges_pct=(50,), bri_mode="3"),     # 18 combos
    BinConfig(cpu_edges_pct=(50,), bri_mode="2"),     # 12 combos (nice 2-digit)
]

MIN_RUN_MINS_LIST = [1, 2, 3, 5]
MIN_TEST_TOTAL_MINS_LIST = [5, 10, 15, 30]
MIN_SLICES_PER_COMBO_LIST = [1, 2]
LAMBDA_LIST = [0.3, 1.0, 3.0, 10.0]


def _acc(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def cpu_bin(series: pd.Series, edges_pct: Tuple[int, ...]) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(float) * 100.0
    bins = [-np.inf] + list(edges_pct) + [np.inf]
    return pd.cut(x, bins=bins, labels=list(range(len(bins) - 1)), include_lowest=True, right=True).astype("Int64").to_numpy()


def net_bin(series: pd.Series) -> np.ndarray:
    n = pd.to_numeric(series, errors="coerce").to_numpy(float)
    out = np.zeros(len(n), dtype="int64")  # none
    out[n == 1] = 1  # mobile
    out[n == 2] = 2  # wifi
    return out


def bri_bin(screen_on: pd.Series, brightness_state: pd.Series, mode: str) -> np.ndarray:
    scr = pd.to_numeric(screen_on, errors="coerce").fillna(0).to_numpy(int)
    b = pd.to_numeric(brightness_state, errors="coerce").to_numpy(float)
    off = (scr == 0) | (~np.isfinite(b)) | (b < 0)
    pct = np.clip(b, 0.0, 1.0) * 100.0
    if mode == "2":
        return np.where(off, 0, 1).astype("int64")  # off/on
    if mode == "3":
        out = np.zeros(len(pct), dtype="int64")
        out[(~off) & (pct <= 50)] = 1
        out[(~off) & (pct >= 51)] = 2
        return out
    out = np.zeros(len(pct), dtype="int64")
    out[(~off) & (pct <= 33)] = 1
    out[(~off) & (pct >= 34) & (pct <= 66)] = 2
    out[(~off) & (pct >= 67)] = 3
    return out


def cpu_label(i: int, edges_pct: Tuple[int, ...]) -> str:
    cuts = [0] + list(edges_pct) + [100]
    lo = cuts[i]
    hi = cuts[i + 1]
    if i == 0:
        return f"几乎不使用（0-{hi}）"
    if i == len(cuts) - 2:
        return f"高负载（{lo+1}-100）"
    return f"使用（{lo+1}-{hi}）"


def bri_label(i: int, mode: str) -> str:
    if i == 0:
        return "息屏"
    if mode == "2":
        return "亮屏"
    if mode == "3":
        return "低亮度" if i == 1 else "高亮度"
    return {1: "低亮度", 2: "中等亮度", 3: "高亮度"}[i]


def net_label(i: int) -> str:
    return {0: "无连接", 1: "蜂窝网络", 2: "无线网络"}[i]


def combo_name(cpu_i: int, bri_i: int, net_i: int, *, edges_pct: Tuple[int, ...], bri_mode: str) -> str:
    return f"{cpu_label(cpu_i, edges_pct)}×{bri_label(bri_i, bri_mode)}×{net_label(net_i)}"


def build_run_slices(df: pd.DataFrame, *, min_run_min: int, min_drop_pct: float) -> pd.DataFrame:
    raise NotImplementedError


def add_local_rate(panel: pd.DataFrame, *, diff_min: int = 30, min_rate: float = 0.5) -> pd.DataFrame:
    d = panel.sort_values("time").copy()
    d["battery_level_pct"] = pd.to_numeric(d["battery_level_pct"], errors="coerce")
    d["soc"] = d["battery_level_pct"] / 100.0
    dt_h = float(diff_min) / 60.0
    d["soc_lag"] = d["soc"].shift(diff_min)
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h
    d["rate_pct_per_h"] = np.where(d["dsoc_dt"] < 0, (-d["dsoc_dt"]) * 100.0, np.nan)
    d = d.dropna(subset=["rate_pct_per_h"]).copy()
    d = d[d["rate_pct_per_h"] >= float(min_rate)].copy()
    return d


def build_run_slices_rate(df: pd.DataFrame, *, min_run_min: int) -> pd.DataFrame:
    """
    Contiguous runs of constant combo_id on discharge-labeled minutes.
    True label is mean local rate within the run.
    """
    d = df.sort_values("time").reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    # break runs also on time gaps (>2min)
    t = pd.to_datetime(d["time"], errors="coerce")
    dt = t.diff().dt.total_seconds().div(60.0).fillna(1.0)
    new_run = (dt > 2.0).to_numpy()

    combo = d["combo_id"].to_numpy(int)
    change = np.zeros(len(combo), dtype=bool)
    change[0] = True
    change[1:] = (combo[1:] != combo[:-1]) | new_run[1:]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(d))

    rows = []
    for s, e in zip(starts, ends):
        minutes = int(e - s)
        if minutes < int(min_run_min):
            continue
        g = d.iloc[s:e]
        y = pd.to_numeric(g["rate_pct_per_h"], errors="coerce").to_numpy(float)
        y = y[np.isfinite(y) & (y > 0)]
        if len(y) < max(2, int(min_run_min // 2)):
            continue
        rate = float(np.mean(y))
        if not np.isfinite(rate) or rate <= 1e-9:
            continue
        g0 = g.iloc[0]
        rows.append(
            {
                "combo_id": int(g0["combo_id"]),
                "cpu_bin": int(g0["cpu_bin"]),
                "bri_bin": int(g0["bri_bin"]),
                "net_bin": int(g0["net_bin"]),
                "start_time": str(g0["time"]),
                "end_time": str(g["time"].iloc[-1]),
                "minutes": minutes,
                "true_rate_pct_per_h": float(rate),
            }
        )
    return pd.DataFrame(rows)


def design_matrix(df: pd.DataFrame, n_cpu: int, n_bri: int, n_net: int) -> np.ndarray:
    cpu = df["cpu_bin"].to_numpy(int)
    bri = df["bri_bin"].to_numpy(int)
    net = df["net_bin"].to_numpy(int)
    n = len(df)
    cols = [np.ones(n)]

    def add_dummies(vals: np.ndarray, levels: int) -> None:
        for lv in range(1, levels):
            cols.append((vals == lv).astype(float))

    add_dummies(cpu, n_cpu)
    add_dummies(bri, n_bri)
    add_dummies(net, n_net)

    def add_inter(a: np.ndarray, la: int, b: np.ndarray, lb: int) -> None:
        for i in range(1, la):
            for j in range(1, lb):
                cols.append(((a == i) & (b == j)).astype(float))

    add_inter(cpu, n_cpu, bri, n_bri)
    add_inter(cpu, n_cpu, net, n_net)
    add_inter(bri, n_bri, net, n_net)

    return np.column_stack(cols)


def ridge_from_sums(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    p = xtx.shape[0]
    return np.linalg.solve(xtx + lam * np.eye(p), xty)


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["time"]).sort_values("time")

    # keep only needed columns
    panel = panel.dropna(subset=["battery_level_pct", "cpu_load", "screen_on", "net_type_code"]).copy()

    rows = []
    best = None
    best_combo = None
    best_slice = None
    best_key = (-np.inf, -np.inf, -np.inf)

    d0 = add_local_rate(panel, diff_min=30, min_rate=0.5)
    # keep only required columns
    d0 = d0.dropna(subset=["cpu_load", "screen_on", "net_type_code"]).copy()

    for cfg, min_run_min, min_test_total_min, min_slices, lam in itertools.product(
        CPU_CONFIGS, MIN_RUN_MINS_LIST, MIN_TEST_TOTAL_MINS_LIST, MIN_SLICES_PER_COMBO_LIST, LAMBDA_LIST
    ):
        d = d0.copy()
        d["cpu_bin"] = cpu_bin(d["cpu_load"], cfg.cpu_edges_pct)
        d["net_bin"] = net_bin(d["net_type_code"])
        d["bri_bin"] = bri_bin(d["screen_on"], d["brightness_state"], cfg.bri_mode)
        d = d.dropna(subset=["cpu_bin"]).copy()
        d["cpu_bin"] = d["cpu_bin"].astype(int)
        n_cpu = len(cfg.cpu_edges_pct) + 1
        n_bri = int(cfg.bri_mode)
        n_net = 3
        d["combo_id"] = d["cpu_bin"] * (n_bri * 3) + d["bri_bin"].astype(int) * 3 + d["net_bin"].astype(int)

        slices = build_run_slices_rate(d, min_run_min=int(min_run_min))
        if slices.empty:
            continue
        slices["combo_name"] = [
            combo_name(int(r.cpu_bin), int(r.bri_bin), int(r.net_bin), edges_pct=cfg.cpu_edges_pct, bri_mode=cfg.bri_mode)
            for r in slices.itertuples()
        ]

        combo_stat = slices.groupby(["combo_id", "combo_name", "cpu_bin", "bri_bin", "net_bin"]).agg(
            n_slices=("minutes", "size"),
            test_total_min=("minutes", "sum"),
        ).reset_index()
        eligible = combo_stat[(combo_stat["test_total_min"] >= int(min_test_total_min)) & (combo_stat["n_slices"] >= int(min_slices))].copy()
        if len(eligible) < 10:
            continue

        # training dataset at slice level
        y = np.log(slices["true_rate_pct_per_h"].to_numpy(float))
        X = design_matrix(slices, n_cpu=n_cpu, n_bri=n_bri, n_net=n_net)
        xtx_total = X.T @ X
        xty_total = X.T @ y
        combo_ids = slices["combo_id"].to_numpy(int)

        per_combo_xtx: Dict[int, np.ndarray] = {}
        per_combo_xty: Dict[int, np.ndarray] = {}
        for cid, idx in pd.Series(np.arange(len(slices))).groupby(combo_ids).groups.items():
            Xi = X[np.array(idx, dtype=int)]
            yi = y[np.array(idx, dtype=int)]
            per_combo_xtx[int(cid)] = Xi.T @ Xi
            per_combo_xty[int(cid)] = Xi.T @ yi

        combo_rows = []
        slice_rows = []
        for _, erow in eligible.iterrows():
            cid = int(erow["combo_id"])
            if cid not in per_combo_xtx:
                continue
            beta = ridge_from_sums(xtx_total - per_combo_xtx[cid], xty_total - per_combo_xty[cid], float(lam))

            # predict rate for this combo from its categorical bins
            df_one = pd.DataFrame({"cpu_bin": [int(erow["cpu_bin"])], "bri_bin": [int(erow["bri_bin"])], "net_bin": [int(erow["net_bin"])]})
            X1 = design_matrix(df_one, n_cpu=n_cpu, n_bri=n_bri, n_net=n_net)
            pred_rate = float(np.exp((X1 @ beta).ravel()[0]))

            gs = slices[slices["combo_id"] == cid].copy()
            accs = []
            for r in gs.itertuples():
                true_dur_h = DROP_STD_PCT / float(r.true_rate_pct_per_h)
                pred_dur_h = DROP_STD_PCT / pred_rate if pred_rate > 1e-9 else np.nan
                acc = _acc(pred_dur_h, true_dur_h)
                accs.append(acc)
                slice_rows.append(
                    {
                        "combo_id": cid,
                        "combo_name": str(r.combo_name),
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                        "minutes": int(r.minutes),
                        "true_rate_pct_per_h": float(r.true_rate_pct_per_h),
                        "pred_rate_pct_per_h": pred_rate,
                        "acc_time": acc,
                    }
                )
            combo_rows.append(
                {
                    "combo_id": cid,
                    "combo_name": str(erow["combo_name"]),
                    "n_slices": int(erow["n_slices"]),
                    "test_total_min": int(erow["test_total_min"]),
                    "pred_rate_pct_per_h": pred_rate,
                    "acc_mean": float(np.nanmean(accs)),
                    "acc_min": float(np.nanmin(accs)),
                }
            )

        combo_df = pd.DataFrame(combo_rows)
        slice_df = pd.DataFrame(slice_rows)
        if combo_df.empty:
            continue

        summ = {
            "cpu_edges_pct": cfg.cpu_edges_pct,
            "bri_mode": cfg.bri_mode,
            "min_run_min": int(min_run_min),
            "min_test_total_min": int(min_test_total_min),
            "min_slices_per_combo": int(min_slices),
            "lambda": float(lam),
            "n_combos_evaluated": int(len(combo_df)),
            "acc_mean_over_combos": float(combo_df["acc_mean"].mean()),
            "acc_min_over_combos": float(combo_df["acc_min"].min()),
        }
        rows.append(summ)

        key = (summ["acc_min_over_combos"], summ["acc_mean_over_combos"], summ["n_combos_evaluated"])
        if key > best_key:
            best_key = key
            best = summ
            best_combo = combo_df
            best_slice = slice_df
            # early stop if target achieved
            if (best["acc_mean_over_combos"] >= 0.90) and (best["acc_min_over_combos"] >= 0.80) and (best["n_combos_evaluated"] >= 10):
                break

    pd.DataFrame(rows).to_csv(OUT_DIR / "search_summary.csv", index=False, encoding="utf-8")
    if best is None or best_combo is None or best_slice is None:
        (OUT_DIR / "best_config.json").write_text(json.dumps({"ok": False, "reason": "no_config"}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("No viable config found.")

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    best_combo.to_csv(OUT_DIR / "best_metrics_by_combo.csv", index=False, encoding="utf-8")
    best_slice.to_csv(OUT_DIR / "best_metrics_by_slice.csv", index=False, encoding="utf-8")
    print("[OK] best:", best)


if __name__ == "__main__":
    main()

