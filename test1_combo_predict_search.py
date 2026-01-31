"""
Search a practical fuzzy-combo prediction setup on test_1 with:
  - >= 10 (two-digit) evaluated combos
  - mean accuracy >= 0.90 and min accuracy >= 0.80 (over evaluated combos)

We accept "coverage not full": only combos with enough stable slices are evaluated.

Core idea:
  1) Discretize observable signals into coarse bins:
       CPU: 3 or 4 bins (fixed edges in percent)
       Network: 3 bins (none/mobile/wifi) from net_type_code
       Brightness: 2/3/4 bins (off + low/mid/high) using screen_on + brightness_state
  2) Extract *strict* slices via rolling windows with dominance>=DOMINANCE:
       window length WIN_MIN, step STEP_MIN
  3) For each combo:
       Train on all other combos using ridge regression on one-hot features:
         main effects + pairwise interactions (cpu×bri, cpu×net, bri×net)
       Predict target combo's log(rate) and evaluate only final duration error.

Data source:
  processed/test1/test1_panel_1min.csv
We create discharge-only labeled points via 30-min differencing, and then build slices on those points.

Outputs:
  processed/test1/combo_predict_search/
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
OUT_DIR = BASE_DIR / "processed" / "test1" / "combo_predict_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Discharge label (local rate)
RATE_DIFF_MIN = 30
MIN_RATE_PCT_PER_H = 0.5


# Slice strictness (searched): contiguous run length
MIN_RUN_MINS = [15, 30, 45, 60]


# Training/testing sufficiency (searched)
MIN_TEST_TOTAL_MINS = [90, 120, 180]
MIN_SLICES_PER_COMBO_LIST = [1, 2, 3]
MIN_TRAIN_POINTS = 3000


LAMBDA_LIST = [0.3, 1.0, 3.0, 10.0]  # ridge

# Safety / speed controls
MAX_CONFIG_EVAL = 400  # hard cap on evaluated configurations (after basic feasibility checks)
STOP_IF_FOUND_GOOD = True

# Caches to avoid repeated slice construction
_SLICE_CACHE: Dict[tuple, pd.DataFrame] = {}
_ELIGIBLE_COUNT_CACHE: Dict[tuple, int] = {}

# Accuracy is evaluated on a standard drop amount (percent points) so we only care about *final time*,
# not the detailed curve.
DROP_STD_PCT = 10.0

# Slices are contiguous runs; no dominance/window scan => linear time.


@dataclass(frozen=True)
class BinConfig:
    cpu_edges_pct: Tuple[int, ...]  # increasing interior edges, e.g. (33,66)
    bri_mode: str  # "2","3","4"


CPU_CONFIGS = [
    BinConfig(cpu_edges_pct=(33, 66), bri_mode="4"),  # CPU3 × Bri4 × Net3 => 36
    BinConfig(cpu_edges_pct=(33, 66), bri_mode="3"),  # 27
    BinConfig(cpu_edges_pct=(33, 66), bri_mode="2"),  # 18
    BinConfig(cpu_edges_pct=(50,), bri_mode="2"),     # CPU2 × Bri2 × Net3 => 12  (two-digit target)
    BinConfig(cpu_edges_pct=(50,), bri_mode="3"),     # 18
    BinConfig(cpu_edges_pct=(25, 50, 75), bri_mode="3"),  # CPU4 × Bri3 × Net3 => 36
    BinConfig(cpu_edges_pct=(25, 50, 75), bri_mode="4"),  # 48
    BinConfig(cpu_edges_pct=(25, 50, 75), bri_mode="2"),  # 24
]


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
        # 0=off, 1=on
        out = np.where(off, 0, 1).astype("int64")
        return out
    if mode == "3":
        # 0=off, 1=low(0-50), 2=high(51-100)
        out = np.zeros(len(pct), dtype="int64")
        out[(~off) & (pct <= 50)] = 1
        out[(~off) & (pct >= 51)] = 2
        return out
    # mode == "4": 0=off, 1=low(0-33), 2=mid(34-66), 3=high(67-100)
    out = np.zeros(len(pct), dtype="int64")
    out[(~off) & (pct <= 33)] = 1
    out[(~off) & (pct >= 34) & (pct <= 66)] = 2
    out[(~off) & (pct >= 67)] = 3
    return out


def cpu_label(i: int, edges_pct: Tuple[int, ...]) -> str:
    # Map bins to Chinese ranges
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
    # 4
    return {1: "低亮度", 2: "中等亮度", 3: "高亮度"}[i]


def net_label(i: int) -> str:
    return {0: "无连接", 1: "蜂窝网络", 2: "无线网络"}[i]


def combo_name(cpu_i: int, bri_i: int, net_i: int, *, edges_pct: Tuple[int, ...], bri_mode: str) -> str:
    return f"{cpu_label(cpu_i, edges_pct)}×{bri_label(bri_i, bri_mode)}×{net_label(net_i)}"


def add_local_rate(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["battery_level_pct"] = pd.to_numeric(d["battery_level_pct"], errors="coerce")
    d["soc"] = d["battery_level_pct"] / 100.0
    dt_h = float(RATE_DIFF_MIN) / 60.0
    d["soc_lag"] = d["soc"].shift(RATE_DIFF_MIN)
    d["dsoc_dt"] = (d["soc"] - d["soc_lag"]) / dt_h
    d["rate_pct_per_h"] = np.where(d["dsoc_dt"] < 0, (-d["dsoc_dt"]) * 100.0, np.nan)
    d = d.dropna(subset=["rate_pct_per_h"]).copy()
    d = d[d["rate_pct_per_h"] >= MIN_RATE_PCT_PER_H].copy()
    return d


def _acc(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def build_slices(df: pd.DataFrame, *, min_run_min: int) -> pd.DataFrame:
    """
    Contiguous runs of constant combo_id (O(N), no rolling windows).
    True label is mean local discharge rate (pct/h) within the run.
    """
    g = df.sort_values("time").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()
    combo = g["combo_id"].to_numpy(int)
    cut = np.where(np.diff(combo) != 0)[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(g)]])
    rows = []
    for s, e in zip(starts, ends):
        minutes = int(e - s)
        if minutes < int(min_run_min):
            continue
        gg = g.iloc[s:e]
        g0 = gg.iloc[0]
        y = pd.to_numeric(gg["rate_pct_per_h"], errors="coerce").to_numpy(float)
        y = y[np.isfinite(y) & (y > 0)]
        # require enough rate labels inside the slice
        if len(y) < max(5, int(min_run_min // 2)):
            continue
        true_rate = float(np.mean(y))
        if not np.isfinite(true_rate) or true_rate <= 1e-12:
            continue
        rows.append(
            {
                "combo_id": int(g0["combo_id"]),
                "cpu_bin": int(g0["cpu_bin"]),
                "bri_bin": int(g0["bri_bin"]),
                "net_bin": int(g0["net_bin"]),
                "start_time": str(g0["time"]),
                "end_time": str(gg["time"].iloc[-1]),
                "minutes": minutes,
                "true_rate_pct_per_h": true_rate,
            }
        )
    return pd.DataFrame(rows)


def design_matrix(df: pd.DataFrame, n_cpu: int, n_bri: int, n_net: int) -> Tuple[np.ndarray, List[str]]:
    """
    One-hot main effects + pairwise interactions, excluding baseline level 0 for each factor.
    """
    cpu = df["cpu_bin"].to_numpy(int)
    bri = df["bri_bin"].to_numpy(int)
    net = df["net_bin"].to_numpy(int)
    n = len(df)
    cols = [np.ones(n)]
    names = ["const"]

    def add_dummies(vals: np.ndarray, levels: int, prefix: str) -> List[np.ndarray]:
        mats = []
        for lv in range(1, levels):
            mats.append((vals == lv).astype(float))
            names.append(f"{prefix}{lv}")
        return mats

    cols += add_dummies(cpu, n_cpu, "cpu")
    cols += add_dummies(bri, n_bri, "bri")
    cols += add_dummies(net, n_net, "net")

    # pairwise interactions (exclude baseline-baseline by construction)
    def add_inter(prefix: str, a: np.ndarray, la: int, b: np.ndarray, lb: int) -> None:
        for i in range(1, la):
            for j in range(1, lb):
                cols.append(((a == i) & (b == j)).astype(float))
                names.append(f"{prefix}{i}_{j}")

    add_inter("cpu_bri_", cpu, n_cpu, bri, n_bri)
    add_inter("cpu_net_", cpu, n_cpu, net, n_net)
    add_inter("bri_net_", bri, n_bri, net, n_net)

    X = np.column_stack(cols)
    return X, names


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _ridge_fit_from_sums(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    p = xtx.shape[0]
    A = xtx + lam * np.eye(p)
    return np.linalg.solve(A, xty)


def evaluate_config(
    d: pd.DataFrame,
    cfg: BinConfig,
    *,
    min_run_min: int,
    min_test_total_min: int,
    min_slices_per_combo: int,
    lam: float,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    # bins
    d = d.copy()
    d["cpu_bin"] = cpu_bin(d["cpu_load"], cfg.cpu_edges_pct)
    d["net_bin"] = net_bin(d["net_type_code"])
    d["bri_bin"] = bri_bin(d["screen_on"], d["brightness_state"], cfg.bri_mode)
    d = d.dropna(subset=["cpu_bin"]).copy()
    d["cpu_bin"] = d["cpu_bin"].astype(int)
    d["combo_id"] = d["cpu_bin"] * (int(cfg.bri_mode) * 3) + d["bri_bin"].astype(int) * 3 + d["net_bin"].astype(int)

    n_cpu = len(cfg.cpu_edges_pct) + 1
    n_bri = int(cfg.bri_mode)
    n_net = 3

    # slices (cached per discretization + min_run_min)
    slice_key = (cfg.cpu_edges_pct, cfg.bri_mode, int(min_run_min))
    if slice_key in _SLICE_CACHE:
        slices = _SLICE_CACHE[slice_key]
    else:
        slices = build_slices(d, min_run_min=min_run_min)
        _SLICE_CACHE[slice_key] = slices
    if slices.empty:
        return {"ok": False, "reason": "no_slices"}, pd.DataFrame(), pd.DataFrame()

    # label combo name
    slices["combo_name"] = [
        combo_name(int(r.cpu_bin), int(r.bri_bin), int(r.net_bin), edges_pct=cfg.cpu_edges_pct, bri_mode=cfg.bri_mode)
        for r in slices.itertuples()
    ]

    # filter combos with enough test minutes and enough slices
    combo_stat = slices.groupby(["combo_id", "combo_name", "cpu_bin", "bri_bin", "net_bin"]).agg(
        n_slices=("minutes", "size"),
        test_total_min=("minutes", "sum"),
    ).reset_index()
    eligible = combo_stat[(combo_stat["test_total_min"] >= min_test_total_min) & (combo_stat["n_slices"] >= min_slices_per_combo)].copy()
    if eligible.empty:
        return {"ok": False, "reason": "no_eligible_combos"}, pd.DataFrame(), pd.DataFrame()

    # training labels
    y_rate = pd.to_numeric(d["rate_pct_per_h"], errors="coerce").to_numpy(float)
    m = np.isfinite(y_rate) & (y_rate > 1e-9)
    d2 = d.loc[m].copy()
    d2["log_rate"] = np.log(pd.to_numeric(d2["rate_pct_per_h"], errors="coerce").to_numpy(float))
    if len(d2) < MIN_TRAIN_POINTS:
        return {"ok": False, "reason": "train_too_small"}, pd.DataFrame(), pd.DataFrame()

    # Build global design matrix once
    X_all, _ = design_matrix(d2, n_cpu=n_cpu, n_bri=n_bri, n_net=n_net)
    y_all = d2["log_rate"].to_numpy(float)
    xtx_total = X_all.T @ X_all
    xty_total = X_all.T @ y_all

    # Precompute per-combo sums for exact leave-one-combo-out ridge
    combo_ids = d2["combo_id"].to_numpy(int)
    per_combo_xtx: Dict[int, np.ndarray] = {}
    per_combo_xty: Dict[int, np.ndarray] = {}
    for cid, idx in pd.Series(np.arange(len(d2))).groupby(combo_ids).groups.items():
        Xg = X_all[np.array(idx, dtype=int)]
        yg = y_all[np.array(idx, dtype=int)]
        per_combo_xtx[int(cid)] = Xg.T @ Xg
        per_combo_xty[int(cid)] = Xg.T @ yg

    # per combo leave-out (fast)
    per_combo_rows = []
    per_slice_rows = []
    for _, row in eligible.iterrows():
        cid = int(row["combo_id"])
        # exact leave-one-combo-out ridge using sufficient statistics
        xtx_c = per_combo_xtx.get(cid)
        xty_c = per_combo_xty.get(cid)
        if xtx_c is None or xty_c is None:
            continue
        xtx_excl = xtx_total - xtx_c
        xty_excl = xty_total - xty_c
        # safety: require enough remaining points
        if int(len(d2) - int((combo_ids == cid).sum())) < MIN_TRAIN_POINTS:
            continue
        beta = _ridge_fit_from_sums(xtx_excl, xty_excl, lam)

        # predict combo-level rate from its bins
        df_one = pd.DataFrame({"cpu_bin": [int(row["cpu_bin"])], "bri_bin": [int(row["bri_bin"])], "net_bin": [int(row["net_bin"])]})
        X1, _ = design_matrix(df_one, n_cpu=n_cpu, n_bri=n_bri, n_net=n_net)
        pred_log = float((X1 @ beta).ravel()[0])
        pred_rate = float(np.exp(pred_log))

        gslice = slices[slices["combo_id"] == cid].copy()
        # slice-level accuracy
        accs = []
        for r in gslice.itertuples():
            # We evaluate time-to-drop for a standard drop amount, so only final time matters.
            true_dur_h = float(DROP_STD_PCT) / float(r.true_rate_pct_per_h)
            pred_dur_h = float(DROP_STD_PCT) / pred_rate if pred_rate > 1e-9 else np.nan
            acc = _acc(pred_dur_h, true_dur_h)
            accs.append(acc)
            per_slice_rows.append(
                {
                    "combo_id": cid,
                    "combo_name": str(r.combo_name),
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "minutes": int(r.minutes),
                    "dominance": float(r.dominance),
                    "dom_minutes": int(r.dom_minutes),
                    "true_rate_pct_per_h": float(r.true_rate_pct_per_h),
                    "pred_rate_pct_per_h": pred_rate,
                    "true_dur_h": true_dur_h,
                    "pred_dur_h": pred_dur_h,
                    "acc_time": acc,
                }
            )

        per_combo_rows.append(
            {
                "combo_id": cid,
                "combo_name": str(row["combo_name"]),
                "cpu_bin": int(row["cpu_bin"]),
                "bri_bin": int(row["bri_bin"]),
                "net_bin": int(row["net_bin"]),
                "n_slices": int(row["n_slices"]),
                "test_total_min": int(row["test_total_min"]),
                "pred_rate_pct_per_h": pred_rate,
                "acc_mean": float(np.nanmean(accs)) if len(accs) else np.nan,
                "acc_min": float(np.nanmin(accs)) if len(accs) else np.nan,
            }
        )

    combo_df = pd.DataFrame(per_combo_rows)
    slice_df = pd.DataFrame(per_slice_rows)
    if combo_df.empty:
        return {"ok": False, "reason": "no_combo_eval"}, pd.DataFrame(), pd.DataFrame()

    summary = {
        "ok": True,
        "cpu_edges_pct": cfg.cpu_edges_pct,
        "bri_mode": cfg.bri_mode,
        "min_run_min": int(min_run_min),
        "min_test_total_min": int(min_test_total_min),
        "min_slices_per_combo": int(min_slices_per_combo),
        "lambda": lam,
        "n_combos_evaluated": int(len(combo_df)),
        "acc_mean_over_combos": float(combo_df["acc_mean"].mean()),
        "acc_min_over_combos": float(combo_df["acc_min"].min()),
    }
    return summary, combo_df, slice_df


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["time"]).sort_values("time")
    d = add_local_rate(panel)

    rows = []
    best = None
    best_combo = None
    best_slice = None
    best_key = (-np.inf, -np.inf, -np.inf)

    # Feasibility scan: count eligible combos without training, so we can prune aggressively.
    feasible_rows = 0
    max_eligible_seen = 0
    for cfg, min_run_min, min_test_total_min, min_slices, lam in itertools.product(
        CPU_CONFIGS, MIN_RUN_MINS, MIN_TEST_TOTAL_MINS, MIN_SLICES_PER_COMBO_LIST, LAMBDA_LIST
    ):
        elig_key = (cfg.cpu_edges_pct, cfg.bri_mode, int(min_run_min), int(min_test_total_min), int(min_slices))
        if elig_key in _ELIGIBLE_COUNT_CACHE:
            n_elig = _ELIGIBLE_COUNT_CACHE[elig_key]
        else:
            dd = d.copy()
            dd["cpu_bin"] = cpu_bin(dd["cpu_load"], cfg.cpu_edges_pct)
            dd["net_bin"] = net_bin(dd["net_type_code"])
            dd["bri_bin"] = bri_bin(dd["screen_on"], dd["brightness_state"], cfg.bri_mode)
            dd = dd.dropna(subset=["cpu_bin"]).copy()
            dd["cpu_bin"] = dd["cpu_bin"].astype(int)
            n_bri = int(cfg.bri_mode)
            dd["combo_id"] = dd["cpu_bin"] * (n_bri * 3) + dd["bri_bin"].astype(int) * 3 + dd["net_bin"].astype(int)
            sl = build_slices(dd, min_run_min=int(min_run_min))
            if sl.empty:
                n_elig = 0
            else:
                combo_stat = sl.groupby(["combo_id"]).agg(n_slices=("minutes", "size"), test_total_min=("minutes", "sum")).reset_index()
                n_elig = int(((combo_stat["test_total_min"] >= int(min_test_total_min)) & (combo_stat["n_slices"] >= int(min_slices))).sum())
            _ELIGIBLE_COUNT_CACHE[elig_key] = n_elig

        max_eligible_seen = max(max_eligible_seen, n_elig)
        if n_elig < 10:
            continue

        feasible_rows += 1
        if feasible_rows > MAX_CONFIG_EVAL:
            break

        summ, combo_df, slice_df = evaluate_config(
            d,
            cfg,
            min_run_min=int(min_run_min),
            min_test_total_min=int(min_test_total_min),
            min_slices_per_combo=int(min_slices),
            lam=float(lam),
        )
        if not summ.get("ok"):
            continue
        rows.append(summ)

        # objective: combos>=10 then maximize min accuracy then mean accuracy
        if summ["n_combos_evaluated"] < 10:
            continue
        key = (summ["acc_min_over_combos"], summ["acc_mean_over_combos"], summ["n_combos_evaluated"])
        if key > best_key:
            best_key = key
            best = summ
            best_combo = combo_df
            best_slice = slice_df
            if STOP_IF_FOUND_GOOD and (best["acc_mean_over_combos"] >= 0.90) and (best["acc_min_over_combos"] >= 0.80):
                break

    pd.DataFrame(rows).to_csv(OUT_DIR / "search_summary.csv", index=False, encoding="utf-8")

    if best is None or best_combo is None or best_slice is None:
        (OUT_DIR / "best_config.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "reason": "no_config_meets_>=10_combos",
                    "max_eligible_combos_seen": int(max_eligible_seen),
                    "note": "Search is bounded and pruned; increase coverage by relaxing thresholds if needed.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError("No config meets >=10 evaluated combos. Try relaxing thresholds further.")

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    best_combo.to_csv(OUT_DIR / "best_metrics_by_combo.csv", index=False, encoding="utf-8")
    best_slice.to_csv(OUT_DIR / "best_metrics_by_slice.csv", index=False, encoding="utf-8")
    print("[OK] best:", best)


if __name__ == "__main__":
    main()

