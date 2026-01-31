"""
Fuzzy matching leave-one-combo-out prediction for test_1 (fast version).

We build discharge-rate labels at 1-min resolution using 30-min differencing:
  rate_pct_per_h(t) = -(SOC(t)-SOC(t-30min))/0.5h * 100
Keep rate>=0.5 %/h.

Define combos for *naming / grouping* (coarse scenario):
  - CPU: 3 bins (0-33,34-66,67-100) by cpu_load%
  - Brightness: 4 bins (off,0-33,34-66,67-100) using screen_on + brightness_state
  - Network: 2 effective bins (mobile/wifi) from net_type_code (none not present)

Protocol:
  For each combo c (eligible by total minutes & slices):
    Train = all minutes with combo!=c
    Test slices = contiguous runs of combo==c (min_run_min)
    For each test slice:
      - features = mean cpu_load, mean brightness_pct (off->0), net_bin (majority)
      - predict rate by kernel-weighted KNN over training minutes
      - evaluate time only via DROP_STD_PCT: acc = 1 - |Tpred-Ttrue|/Ttrue

Outputs:
  processed/test1/combo_fuzzy_predict/
    best_config.json
    metrics_by_combo.csv
    metrics_by_slice.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import itertools
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "combo_fuzzy_predict"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_MIN = 30
MIN_RATE = 0.5
DROP_STD_PCT = 10.0

MIN_RUN_MIN = 5
MIN_SLICES_PER_COMBO = 2

# We will evaluate only Top-N combos by coverage so that the table is compact and metrics are meaningful.
TARGET_N_LIST = [6, 4]
MIN_TEST_TOTAL_MIN_LIST = [30, 60, 120, 240]

CPU_EDGES = (33, 66)  # 3 bins

K_LIST = [200, 500, 1000]
SIGMA_LIST = [0.06, 0.08, 0.12]
NET_PENALTY_LIST = [0.0, 0.2, 0.4]
SOC_PENALTY_LIST = [0.0, 0.1, 0.2]  # SOC is observable; helps reduce unexplained drift


def _acc(pred: float, true: float) -> float:
    if not np.isfinite(pred) or not np.isfinite(true) or true <= 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - abs(pred - true) / abs(true)))


def cpu_bin(cpu_load: np.ndarray) -> np.ndarray:
    x = cpu_load * 100.0
    bins = [-np.inf, CPU_EDGES[0], CPU_EDGES[1], np.inf]
    return pd.cut(x, bins=bins, labels=[0, 1, 2], include_lowest=True, right=True).astype("Int64").to_numpy()


def net_bin(net_type_code: np.ndarray) -> np.ndarray:
    n = net_type_code
    out = np.zeros(len(n), dtype="int64")
    out[n == 1] = 0  # mobile -> 0
    out[n == 2] = 1  # wifi -> 1
    # other treated as mobile baseline
    return out


def brightness_pct(screen_on: np.ndarray, brightness_state: np.ndarray) -> np.ndarray:
    scr = screen_on.astype(int)
    b = brightness_state
    off = (scr == 0) | (~np.isfinite(b)) | (b < 0)
    pct = np.clip(b, 0.0, 1.0) * 100.0
    pct = np.where(off, 0.0, pct)
    return pct


def bri_bin(br_pct: np.ndarray) -> np.ndarray:
    # 0=off, 1=0-33, 2=34-66, 3=67-100
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
                "start_time": str(g["time"].iloc[0]),
                "end_time": str(g["time"].iloc[-1]),
                "minutes": minutes,
                "true_rate_pct_per_h": float(np.mean(g["rate_pct_per_h"].to_numpy(float))),
                "cpu_mean": float(np.mean(g["cpu_load"].to_numpy(float))),
                "bri_mean": float(np.mean(g["brightness_pct"].to_numpy(float))),
                "soc_mean": float(np.mean(g["battery_level_pct"].to_numpy(float))),
                "net_major": int(g["net2"].value_counts().idxmax()),
            }
        )
    return pd.DataFrame(rows)


def predict_rate_knn(
    train_cpu: np.ndarray,
    train_bri: np.ndarray,
    train_net: np.ndarray,
    train_soc: np.ndarray,
    train_rate: np.ndarray,
    *,
    cpu_m: float,
    bri_m: float,
    net_m: int,
    soc_m: float,
    k: int,
    sigma: float,
    net_penalty: float,
    soc_penalty: float,
) -> float:
    # distance in cpu/bri (scaled to [0,1] roughly), plus a discrete net penalty
    dc = (train_cpu - cpu_m)
    db = (train_bri - bri_m) / 100.0
    ds = (train_soc - soc_m) / 100.0
    d2 = dc * dc + db * db
    d2 = d2 + (train_net != net_m).astype(float) * (net_penalty**2)
    d2 = d2 + ds * ds * (soc_penalty**2)
    # take k nearest by d2
    k = min(int(k), len(d2))
    idx = np.argpartition(d2, k - 1)[:k]
    d2k = d2[idx]
    wk = np.exp(-d2k / (2 * sigma * sigma))
    rk = train_rate[idx]
    wsum = float(np.sum(wk))
    if wsum <= 1e-12:
        return float(np.mean(rk))
    return float(np.sum(wk * rk) / wsum)


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
    panel["cpu3"] = cpu_bin(cpu)
    panel = panel.dropna(subset=["cpu3"]).copy()
    panel["cpu3"] = panel["cpu3"].astype(int)
    panel["bri4"] = bri_bin(panel["brightness_pct"].to_numpy(float)).astype(int)
    panel["net2"] = net_bin(pd.to_numeric(panel["net_type_code"], errors="coerce").to_numpy(float)).astype(int)

    panel["combo_name"] = [
        combo_name(int(r.cpu3), int(r.bri4), int(r.net2)) for r in panel[["cpu3", "bri4", "net2"]].itertuples(index=False)
    ]
    panel["combo_id"] = panel["cpu3"] * 8 + panel["bri4"] * 2 + panel["net2"]  # 0..23 but some absent

    # build evaluation slices
    slices = build_slices(panel, min_run_min=MIN_RUN_MIN)
    if slices.empty:
        raise RuntimeError("No slices found.")

    stat0 = slices.groupby(["combo_id", "combo_name"]).agg(n_slices=("minutes", "size"), total_min=("minutes", "sum")).reset_index()

    best = None
    best_combo = None
    best_slice = None
    best_key = (-np.inf, -np.inf, -np.inf)  # (min_acc, mean_acc, n_combos)

    # pre-extract training arrays per combo mask efficiently
    all_cpu = pd.to_numeric(panel["cpu_load"], errors="coerce").to_numpy(float)
    all_bri = pd.to_numeric(panel["brightness_pct"], errors="coerce").to_numpy(float)
    all_net = pd.to_numeric(panel["net2"], errors="coerce").to_numpy(int)
    all_soc = pd.to_numeric(panel["battery_level_pct"], errors="coerce").to_numpy(float)
    all_rate = pd.to_numeric(panel["rate_pct_per_h"], errors="coerce").to_numpy(float)
    all_combo = pd.to_numeric(panel["combo_id"], errors="coerce").to_numpy(int)

    for target_n, min_total, k, sigma, net_penalty, soc_penalty in itertools.product(
        TARGET_N_LIST, MIN_TEST_TOTAL_MIN_LIST, K_LIST, SIGMA_LIST, NET_PENALTY_LIST, SOC_PENALTY_LIST
    ):
        stat = stat0[(stat0["total_min"] >= int(min_total)) & (stat0["n_slices"] >= MIN_SLICES_PER_COMBO)].copy()
        if len(stat) < int(target_n):
            continue
        eligible = stat.sort_values(["total_min", "n_slices"], ascending=False).head(int(target_n)).copy()

        combo_rows = []
        slice_rows = []
        for _, er in eligible.iterrows():
            cid = int(er["combo_id"])
            train_mask = all_combo != cid
            train_cpu = all_cpu[train_mask]
            train_bri = all_bri[train_mask]
            train_net = all_net[train_mask]
            train_soc = all_soc[train_mask]
            train_rate = all_rate[train_mask]

            gs = slices[slices["combo_id"] == cid]
            accs = []
            for r in gs.itertuples():
                pred_rate = predict_rate_knn(
                    train_cpu,
                    train_bri,
                    train_net,
                    train_soc,
                    train_rate,
                    cpu_m=float(r.cpu_mean),
                    bri_m=float(r.bri_mean),
                    net_m=int(r.net_major),
                    soc_m=float(r.soc_mean),
                    k=int(k),
                    sigma=float(sigma),
                    net_penalty=float(net_penalty),
                    soc_penalty=float(soc_penalty),
                )
                true_dur_h = DROP_STD_PCT / float(r.true_rate_pct_per_h)
                pred_dur_h = DROP_STD_PCT / pred_rate if pred_rate > 1e-9 else np.nan
                acc = _acc(pred_dur_h, true_dur_h)
                accs.append(acc)
                slice_rows.append(
                    {
                        "combo_id": cid,
                        "combo_name": str(r.combo_name),
                        "minutes": int(r.minutes),
                        "true_rate_pct_per_h": float(r.true_rate_pct_per_h),
                        "pred_rate_pct_per_h": float(pred_rate),
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
        sdf = pd.DataFrame(slice_rows)
        acc_mean = float(cdf["acc_mean"].mean())
        acc_min = float(cdf["acc_mean"].min())
        key = (acc_min, acc_mean, int(len(cdf)))
        if key > best_key:
            best_key = key
            best = {
                "target_n": int(target_n),
                "min_total_min": int(min_total),
                "k": int(k),
                "sigma": float(sigma),
                "net_penalty": float(net_penalty),
                "soc_penalty": float(soc_penalty),
                "n_combos": int(len(cdf)),
                "acc_mean": acc_mean,
                "acc_min": acc_min,
            }
            best_combo = cdf.sort_values("acc_mean")
            best_slice = sdf

    if best is None or best_combo is None or best_slice is None:
        raise RuntimeError("No config produced results.")

    (OUT_DIR / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    best_combo.to_csv(OUT_DIR / "metrics_by_combo.csv", index=False, encoding="utf-8")
    best_slice.to_csv(OUT_DIR / "metrics_by_slice.csv", index=False, encoding="utf-8")
    print("[OK] best", best)


if __name__ == "__main__":
    main()

