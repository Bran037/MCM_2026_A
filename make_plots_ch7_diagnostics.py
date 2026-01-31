from __future__ import annotations

"""
Chapter 7 diagnostics: pick typical discharge segments (test1 x4, main x2),
make rich multi-panel visualizations, and export per-case summary tables.

Outputs:
  MCM_2026_A/figures/diagnostics_cases/
    cases_summary.csv
    test1_epi{e}_int{i}.png
    main_352944080639365_seg{seg}.png
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures" / "diagnostics_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _sanitize_text(s: str) -> str:
    return str(s).replace("\u2011", "-")


def _rate_from_soc_pct(soc_pct: np.ndarray, diff_min: int = 30) -> np.ndarray:
    """Return local discharge rate in %/hour using diff_min lag; NaN for non-discharge or insufficient lag."""
    soc = soc_pct.astype(float)
    lag = np.roll(soc, diff_min)
    lag[:diff_min] = np.nan
    dsoc = soc - lag
    rate = np.where(dsoc < 0, (-dsoc) / (diff_min / 60.0), np.nan)
    return rate


def _pick_distinct(rows: pd.DataFrame, keys: List[str], k: int) -> pd.DataFrame:
    """Pick up to k distinct rows, preferring top of each criterion list, avoiding duplicates by (episode_id,interval_id)."""
    picked: List[Tuple[int, int]] = []
    out_rows = []
    for key in keys:
        d = rows.sort_values(key, ascending=False).copy()
        for r in d.itertuples(index=False):
            eid = int(getattr(r, "episode_id"))
            iid = int(getattr(r, "interval_id"))
            if (eid, iid) in picked:
                continue
            picked.append((eid, iid))
            out_rows.append(r)
            break
        if len(out_rows) >= k:
            break
    # fill remaining by highest rate
    if len(out_rows) < k:
        d = rows.sort_values("rate_pct_per_h", ascending=False)
        for r in d.itertuples(index=False):
            eid = int(getattr(r, "episode_id"))
            iid = int(getattr(r, "interval_id"))
            if (eid, iid) in picked:
                continue
            picked.append((eid, iid))
            out_rows.append(r)
            if len(out_rows) >= k:
                break
    return pd.DataFrame(out_rows)


def summarize_test1_intervals() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pts_path = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_points_1min.csv"
    itv_path = BASE_DIR / "processed" / "test1" / "episodes" / "test1_episode_intervals.csv"
    pts = pd.read_csv(pts_path, parse_dates=["time"])
    itv = pd.read_csv(itv_path, parse_dates=["start_time", "end_time"])
    # keep only detected discharge minutes
    if "is_discharge" in pts.columns:
        pts = pts[pts["is_discharge"] == 1].copy()

    # compute per-interval feature summaries from points
    g = pts.groupby(["episode_id", "interval_id"], as_index=False)
    f = g.agg(
        minutes=("time", "size"),
        soc_start=("battery_level_pct", "first"),
        soc_end=("battery_level_pct", "last"),
        temp_mean_C=("battery_temp_C", "mean"),
        cpu_load_mean=("cpu_load", "mean"),
        cpu_load_p90=("cpu_load", lambda x: float(np.nanpercentile(x, 90))),
        cpu_freq_norm_mean=("cpu_freq_norm", "mean"),
        screen_on_ratio=("screen_on", "mean"),
        brightness_state_mean=(
            "brightness_state",
            lambda x: float(
                np.nanmean(
                    np.where(
                        pd.to_numeric(pts.loc[x.index, "screen_on"], errors="coerce").fillna(0).to_numpy(float) == 1,
                        pd.to_numeric(x, errors="coerce").to_numpy(float),
                        np.nan,
                    )
                )
            ),
        ),
        net_wifi_ratio=("net_type_code", lambda x: float(np.nanmean(np.asarray(x) == 2))),
        net_mobile_ratio=("net_type_code", lambda x: float(np.nanmean(np.asarray(x) == 1))),
    )
    f["drop_pct"] = f["soc_start"] - f["soc_end"]
    f["rate_pct_per_h"] = f["drop_pct"] / (f["minutes"] / 60.0)
    # remove any abnormal "charge" cases
    f = f[(f["drop_pct"] > 0) & (f["rate_pct_per_h"] > 0)].copy()

    # attach global_interval_id from itv
    f = f.merge(itv[["episode_id", "interval_id", "global_interval_id"]], on=["episode_id", "interval_id"], how="left")
    return pts, f


def plot_test1_case(pts: pd.DataFrame, row: pd.Series) -> Path:
    eid = int(row["episode_id"])
    iid = int(row["interval_id"])
    g = pts[(pts["episode_id"] == eid) & (pts["interval_id"] == iid)].copy().sort_values("time")
    t = g["time"]
    soc = pd.to_numeric(g["battery_level_pct"], errors="coerce").to_numpy(float)
    rate = _rate_from_soc_pct(soc, diff_min=30)

    cpu = pd.to_numeric(g["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(g["cpu_freq_norm"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(g["screen_on"], errors="coerce").fillna(0).to_numpy(float)
    bri = pd.to_numeric(g["brightness_state"], errors="coerce").to_numpy(float) * 100.0
    bri = np.where(scr > 0.5, bri, 0.0)
    net = pd.to_numeric(g["net_type_code"], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(g["battery_temp_C"], errors="coerce").to_numpy(float)

    title = (
        f"test1 案例：episode {eid} / interval {iid}（{int(row['minutes'])} min, "
        f"ΔSOC={row['drop_pct']:.1f}%, 平均速率≈{row['rate_pct_per_h']:.2f}%/h）"
    )

    fig, axes = plt.subplots(6, 1, figsize=(11.5, 9.5), dpi=160, sharex=True)
    axes[0].plot(t, soc, color="#4C78A8", linewidth=1.6)
    axes[0].set_ylabel("SOC(%)")
    axes[0].set_title(title)

    axes[1].plot(t, rate, color="#F58518", linewidth=1.4)
    axes[1].set_ylabel("rate\n(%/h)")

    axes[2].plot(t, cpu, color="#54A24B", linewidth=1.2, label="cpu_load")
    axes[2].plot(t, f, color="#B279A2", linewidth=1.0, alpha=0.9, label="cpu_freq_norm")
    axes[2].set_ylabel("CPU")
    axes[2].legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

    axes[3].plot(t, bri, color="#E45756", linewidth=1.2)
    axes[3].set_ylabel("亮度\n(%)")

    # net type: 1 mobile, 2 wifi
    axes[4].step(t, net, where="post", color="#72B7B2", linewidth=1.2)
    axes[4].set_yticks([1, 2])
    axes[4].set_yticklabels(["蜂窝", "Wi-Fi"])
    axes[4].set_ylabel("网络")

    axes[5].plot(t, temp, color="#9C755F", linewidth=1.2)
    axes[5].set_ylabel("温度\n(°C)")
    axes[5].set_xlabel("time")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / f"test1_epi{eid}_int{iid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize_main_device(device_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = BASE_DIR / "processed" / "discharge" / f"{device_id}_discharge_1min.csv"
    segs = BASE_DIR / "processed" / "discharge" / f"{device_id}_segments_summary.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ss = pd.read_csv(segs, parse_dates=["start_utc", "end_utc"])
    # compute avg current/power per segment
    df["I_mA_abs"] = pd.to_numeric(df["battery_current_mA"], errors="coerce").abs()
    df["P_mW"] = df["I_mA_abs"] * pd.to_numeric(df["battery_voltage_mV"], errors="coerce") / 1000.0
    g = df[df["segment_id"] != -1].groupby("segment_id", as_index=False).agg(
        I_mA_mean=("I_mA_abs", "mean"),
        P_mW_mean=("P_mW", "mean"),
        temp_mean_C=("battery_temp_C", "mean"),
        screen_on_ratio=("screen_on", "mean"),
        net_mode=("network_type", lambda x: str(pd.Series(x).mode().iloc[0]) if len(pd.Series(x).mode()) else "unknown"),
        minutes=("timestamp", "size"),
        soc_start=("battery_level_pct", "first"),
        soc_end=("battery_level_pct", "last"),
    )
    g["drop_pct"] = g["soc_start"] - g["soc_end"]
    g["rate_pct_per_h"] = g["drop_pct"] / (g["minutes"] / 60.0)
    g = g.merge(ss[["segment_id", "start_utc", "end_utc"]], on="segment_id", how="left")
    return df, g


def plot_main_case(df: pd.DataFrame, row: pd.Series, device_id: str) -> Path:
    seg = int(row["segment_id"])
    g = df[df["segment_id"] == seg].copy().sort_values("timestamp")
    t = g["timestamp"]
    soc = pd.to_numeric(g["battery_level_pct"], errors="coerce").to_numpy(float)
    rate = _rate_from_soc_pct(soc, diff_min=30)
    I = pd.to_numeric(g["battery_current_mA"], errors="coerce").abs().to_numpy(float)
    V = pd.to_numeric(g["battery_voltage_mV"], errors="coerce").to_numpy(float)
    P = I * V / 1000.0
    temp = pd.to_numeric(g["battery_temp_C"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(g["screen_on"], errors="coerce").fillna(0).astype(int).to_numpy(int)
    net = g["network_type"].astype(str).to_numpy()

    title = (
        f"主数据案例（device {device_id} / segment {seg}，{int(row['minutes'])} min, "
        f"ΔSOC={row['drop_pct']:.1f}%, 平均速率≈{row['rate_pct_per_h']:.2f}%/h）"
    )

    fig, axes = plt.subplots(6, 1, figsize=(11.5, 9.5), dpi=160, sharex=True)
    axes[0].plot(t, soc, color="#4C78A8", linewidth=1.6)
    axes[0].set_ylabel("SOC(%)")
    axes[0].set_title(_sanitize_text(title))
    axes[0].set_title(_sanitize_text(title))

    axes[1].plot(t, rate, color="#F58518", linewidth=1.4)
    axes[1].set_ylabel("rate\n(%/h)")

    axes[2].plot(t, I, color="#54A24B", linewidth=1.2)
    axes[2].set_ylabel("|I|\n(mA)")

    axes[3].plot(t, P, color="#B279A2", linewidth=1.2)
    axes[3].set_ylabel("P\n(mW)")

    # net as categorical: map to code
    cats = pd.Series(net).astype("category")
    codes = cats.cat.codes.to_numpy(int)
    axes[4].step(t, codes, where="post", color="#72B7B2", linewidth=1.2)
    axes[4].set_yticks(np.unique(codes))
    axes[4].set_yticklabels([str(c) for c in cats.cat.categories], fontsize=9)
    axes[4].set_ylabel("网络")

    axes[5].plot(t, temp, color="#9C755F", linewidth=1.2, label="temp")
    axes[5].plot(t, scr * (np.nanmin(temp) if np.isfinite(np.nanmin(temp)) else 0), color="#E45756", linewidth=1.0, alpha=0.5, label="screen_on(示意)")
    axes[5].set_ylabel("温度\n(°C)")
    axes[5].set_xlabel("time")
    axes[5].legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / f"main_{device_id}_seg{seg}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _set_chinese_font()

    # --- test1: pick 4 intervals with diverse drivers ---
    pts, f = summarize_test1_intervals()
    # ensure long-enough intervals (avoid tiny)
    f2 = f[f["minutes"] >= 300].copy()
    # create criteria keys
    # higher rate, higher cpu load, higher brightness (screen on & bright), higher temp
    f2["brightness_mean_pct"] = f2["brightness_state_mean"] * 100.0
    keys = ["rate_pct_per_h", "cpu_load_p90", "brightness_mean_pct", "temp_mean_C"]
    pick4 = _pick_distinct(f2, keys=keys, k=4)

    case_rows = []
    for _, r in pick4.iterrows():
        figpath = plot_test1_case(pts, r)
        case_rows.append(
            {
                "dataset": "test1",
                "case_id": f"test1_epi{int(r['episode_id'])}_int{int(r['interval_id'])}",
                "figure": str(figpath.relative_to(BASE_DIR)).replace("\\", "/"),
                "minutes": int(r["minutes"]),
                "drop_pct": float(r["drop_pct"]),
                "rate_pct_per_h": float(r["rate_pct_per_h"]),
                "cpu_load_mean": float(r["cpu_load_mean"]),
                "cpu_load_p90": float(r["cpu_load_p90"]),
                "cpu_freq_norm_mean": float(r["cpu_freq_norm_mean"]),
                "screen_on_ratio": float(r["screen_on_ratio"]),
                "brightness_mean_pct": float(r["brightness_mean_pct"]),
                "net_mobile_ratio": float(r["net_mobile_ratio"]),
                "net_wifi_ratio": float(r["net_wifi_ratio"]),
                "temp_mean_C": float(r["temp_mean_C"]),
            }
        )

    # --- main dataset: pick 2 segments from one device for clear contrast ---
    device_id = "352944080639365"
    df, segsum = summarize_main_device(device_id)
    segsum2 = segsum[(segsum["minutes"] >= 300) & (segsum["drop_pct"] > 0) & (segsum["rate_pct_per_h"] > 0)].copy()
    # pick one "slow" (lowest rate) and one "fast" (highest rate) among long segments
    # Prefer contrasting net_mode when possible: one 'none' and one 'wi-fi' (common in this device)
    none_seg = segsum2[segsum2["net_mode"].astype(str) == "none"].sort_values("rate_pct_per_h").head(1)
    wifi_seg = segsum2[segsum2["net_mode"].astype(str).isin(["wi-fi", "wifi"])].sort_values("rate_pct_per_h", ascending=False).head(1)
    if len(none_seg) and len(wifi_seg):
        pick_main = pd.concat([none_seg, wifi_seg], ignore_index=True)
    else:
        slow = segsum2.sort_values("rate_pct_per_h").head(1)
        fast = segsum2.sort_values("rate_pct_per_h", ascending=False).head(1)
        pick_main = pd.concat([slow, fast], ignore_index=True).drop_duplicates(subset=["segment_id"])

    for _, r in pick_main.iterrows():
        figpath = plot_main_case(df, r, device_id=device_id)
        case_rows.append(
            {
                "dataset": "main",
                "case_id": f"main_{device_id}_seg{int(r['segment_id'])}",
                "figure": str(figpath.relative_to(BASE_DIR)).replace("\\", "/"),
                "minutes": int(r["minutes"]),
                "drop_pct": float(r["drop_pct"]),
                "rate_pct_per_h": float(r["rate_pct_per_h"]),
                "I_mA_mean": float(r.get("I_mA_mean", np.nan)),
                "P_mW_mean": float(r.get("P_mW_mean", np.nan)),
                "temp_mean_C": float(r.get("temp_mean_C", np.nan)),
                "screen_on_ratio": float(r.get("screen_on_ratio", np.nan)),
                "net_mode": str(r.get("net_mode", "")),
            }
        )

    out = pd.DataFrame(case_rows)
    out.to_csv(OUT_DIR / "cases_summary.csv", index=False, encoding="utf-8")
    print("[OK] wrote", len(out), "cases to", str(OUT_DIR))


if __name__ == "__main__":
    main()

