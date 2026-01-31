from __future__ import annotations

"""
Chapter 7 summary conclusion: "contribution shares" of four drivers under a unified normalized scale.

User-required normalization (interpreted in a data-feasible, auditable way):
  - Network connectivity (connected vs none) is the 0/100 baseline unit:
      net_unit = 0 if no connection, 100 if connected (Wi-Fi or mobile).
  - CPU load, screen brightness, temperature are all mapped onto 0..100 scales,
    and 1% increase in these scales corresponds to 1% of the network 100-unit.

Data reality:
  - test_1 panel has only mobile / wi-fi (no 'none'), so net_unit would be constant there.
    To satisfy "no-connection=0" we pool main dataset (which has network_type=='none') to provide 0 states.
  - main dataset lacks explicit CPU load and brightness; we approximate:
      CPU_equiv_unit from |current| scaled per-device so that median(connected) maps to 100 and low-quantile(idle) maps to 0.
      Brightness_unit from screen_on only (0 if off, 50 if on) as a conservative proxy.

Contribution definition (simple, reproducible; not a causal energy decomposition):
  - Let s_j be the standard deviation of each unit variable over discharge minutes.
  - Contribution share: w_j = s_j / sum_k s_k.

Outputs:
  figures/diagnostics_cases/contribution_share_universal.csv
  figures/diagnostics_cases/contribution_share_universal.png
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures" / "diagnostics_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST1_PATH = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
MAIN_GLOB = BASE_DIR / "processed" / "discharge"


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


def _rate_from_soc_pct(soc_pct: np.ndarray, diff_min: int = 30) -> np.ndarray:
    soc = soc_pct.astype(float)
    lag = np.roll(soc, diff_min)
    lag[:diff_min] = np.nan
    dsoc = soc - lag
    rate = np.where(dsoc < 0, (-dsoc) / (diff_min / 60.0), np.nan)
    return rate


def load_test1_discharge_minutes() -> pd.DataFrame:
    df = pd.read_csv(TEST1_PATH, parse_dates=["time"])
    df["soc_pct"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    rate = _rate_from_soc_pct(df["soc_pct"].to_numpy(float), diff_min=30)
    df["rate_pct_per_h"] = rate
    df = df[np.isfinite(df["rate_pct_per_h"]) & (df["rate_pct_per_h"] >= 0.5)].copy()

    df["net_connected"] = 1  # in test_1 we only see mobile/wi-fi
    df["net_unit"] = 100.0
    df["cpu_unit"] = 100.0 * pd.to_numeric(df["cpu_load"], errors="coerce").to_numpy(float)
    scr = pd.to_numeric(df["screen_on"], errors="coerce").fillna(0).to_numpy(float)
    bri = pd.to_numeric(df["brightness_state"], errors="coerce").to_numpy(float) * 100.0
    df["bri_unit"] = np.where(scr > 0.5, np.clip(bri, 0.0, 100.0), 0.0)
    df["temp_C"] = pd.to_numeric(df["battery_temp_C"], errors="coerce").to_numpy(float)
    df["source"] = "test1"
    return df[["source", "net_unit", "cpu_unit", "bri_unit", "temp_C"]].copy()


def _cpu_equiv_from_current(device_df: pd.DataFrame) -> np.ndarray:
    """
    Map |I| to a 0..100 scale per device:
      0 ~ low-quantile of |I| when net==none & screen_off
      100 ~ median of |I| when net!=none & screen_off
    """
    I = pd.to_numeric(device_df["battery_current_mA"], errors="coerce").abs().to_numpy(float)
    scr = pd.to_numeric(device_df["screen_on"], errors="coerce").fillna(0).astype(int).to_numpy(int)
    net = device_df["network_type"].astype(str).to_numpy()
    none = (net == "none") & (scr == 0)
    conn = (net != "none") & (scr == 0)

    # robust anchors
    I0 = float(np.nanpercentile(I[none], 10)) if np.any(none) else float(np.nanpercentile(I, 10))
    I100 = float(np.nanmedian(I[conn])) if np.any(conn) else float(np.nanmedian(I))
    denom = I100 - I0
    if not np.isfinite(denom) or denom <= 1e-9:
        denom = 1.0
    cpu = 100.0 * (I - I0) / denom
    return np.clip(cpu, 0.0, 200.0)


def load_main_discharge_minutes() -> pd.DataFrame:
    paths = sorted(MAIN_GLOB.glob("*_discharge_1min.csv"))
    parts = []
    for p in paths:
        device_id = p.name.split("_")[0]
        df = pd.read_csv(p, parse_dates=["timestamp"])
        # keep only labeled discharge segments
        df = df[df["segment_id"] != -1].copy()
        if df.empty:
            continue
        df["device_id"] = device_id
        parts.append(df)
    all_df = pd.concat(parts, ignore_index=True)

    all_df["net_connected"] = (all_df["network_type"].astype(str) != "none").astype(int)
    all_df["net_unit"] = 100.0 * all_df["net_connected"].to_numpy(int)

    # CPU equiv per device
    cpu_units = np.full(len(all_df), np.nan, dtype=float)
    for dev, idx in all_df.groupby("device_id").groups.items():
        sub = all_df.loc[idx]
        cpu_units[np.array(idx, dtype=int)] = _cpu_equiv_from_current(sub)
    all_df["cpu_unit"] = cpu_units

    # brightness proxy: only screen_on known; use 0/50 as conservative brightness level proxy
    scr = pd.to_numeric(all_df["screen_on"], errors="coerce").fillna(0).astype(int).to_numpy(int)
    all_df["bri_unit"] = np.where(scr == 1, 50.0, 0.0)

    all_df["temp_C"] = pd.to_numeric(all_df["battery_temp_C"], errors="coerce").to_numpy(float)
    all_df["source"] = "main"
    return all_df[["source", "net_unit", "cpu_unit", "bri_unit", "temp_C"]].copy()


def main() -> None:
    _set_chinese_font()

    t1 = load_test1_discharge_minutes()
    md = load_main_discharge_minutes()
    df = pd.concat([t1, md], ignore_index=True)

    # temperature normalized 0..100 on pooled discharge minutes
    Tmin = float(np.nanmin(df["temp_C"].to_numpy(float)))
    Tmax = float(np.nanmax(df["temp_C"].to_numpy(float)))
    denom = (Tmax - Tmin) if (Tmax - Tmin) > 1e-9 else 1.0
    df["temp_unit"] = 100.0 * (df["temp_C"] - Tmin) / denom

    cols = {
        "网络（连接=100, 无连接=0）": "net_unit",
        "CPU（0-100）": "cpu_unit",
        "屏幕亮度（0-100）": "bri_unit",
        "温度（按观测范围归一化0-100）": "temp_unit",
    }

    s = {k: float(np.nanstd(df[v].to_numpy(float))) for k, v in cols.items()}
    total = float(sum(s.values())) if sum(s.values()) > 1e-12 else 1.0
    w = {k: s[k] / total for k in s}

    out = pd.DataFrame(
        [{"driver": k, "std_unit": s[k], "share": w[k]} for k in cols.keys()]
    ).sort_values("share", ascending=False)
    out["share_pct"] = out["share"] * 100.0
    out["Tmin_C"] = Tmin
    out["Tmax_C"] = Tmax
    out["n_test1_minutes"] = int(len(t1))
    out["n_main_minutes"] = int(len(md))
    out.to_csv(OUT_DIR / "contribution_share_universal.csv", index=False, encoding="utf-8")

    fig = plt.figure(figsize=(8.2, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.bar(out["driver"], out["share_pct"], color="#4C78A8", alpha=0.9)
    ax.set_ylabel("贡献率（%）")
    ax.set_ylim(0, max(5.0, float(out["share_pct"].max()) * 1.15))
    ax.set_title("统一归一化比例尺：四项驱动对高电耗波动的“贡献率”对比（std占比）")
    ax.tick_params(axis="x", rotation=15)
    for x, y in zip(out["driver"], out["share_pct"]):
        ax.text(x, float(y) + 0.6, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contribution_share_universal.png", bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote contribution summary to", str(OUT_DIR))


if __name__ == "__main__":
    main()

