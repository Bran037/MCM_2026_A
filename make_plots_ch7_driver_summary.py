from __future__ import annotations

"""
Create compact "driver summary" visualizations for Chapter 7.

Outputs (under figures/diagnostics_cases/):
  - test1_driver_heatmap.png
  - main_power_compare.png
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
IN_PATH = BASE_DIR / "figures" / "diagnostics_cases" / "cases_summary.csv"
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


def main() -> None:
    _set_chinese_font()
    df = pd.read_csv(IN_PATH)

    # --- test1 heatmap: normalized driver proxies ---
    t1 = df[df["dataset"] == "test1"].copy()
    if len(t1):
        # proxies (dimensionless, for relative comparison only)
        t1["cpu_proxy"] = t1["cpu_load_p90"] * t1["cpu_freq_norm_mean"]
        t1["screen_proxy"] = t1["screen_on_ratio"] * (t1["brightness_mean_pct"] / 100.0)
        t1["net_proxy"] = t1["net_mobile_ratio"]
        t1["temp_proxy"] = np.maximum(0.0, t1["temp_mean_C"] - 30.0) / 10.0

        feats = ["cpu_proxy", "screen_proxy", "net_proxy", "temp_proxy"]
        mat = t1[feats].to_numpy(float)
        # normalize per feature
        mmin = np.nanmin(mat, axis=0)
        mmax = np.nanmax(mat, axis=0)
        denom = np.where((mmax - mmin) > 1e-12, (mmax - mmin), 1.0)
        matn = (mat - mmin) / denom

        fig = plt.figure(figsize=(8.6, 2.8), dpi=160)
        ax = fig.add_subplot(111)
        im = ax.imshow(matn, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(t1)))
        ax.set_yticklabels(t1["case_id"].tolist(), fontsize=9)
        ax.set_xticks(np.arange(len(feats)))
        ax.set_xticklabels(["CPU(代理)", "屏幕(代理)", "网络(代理)", "温度(代理)"], fontsize=9)
        ax.set_title("test1 典型片段：驱动强度（归一化代理指标，0-1）")
        for i in range(matn.shape[0]):
            for j in range(matn.shape[1]):
                ax.text(j, i, f"{matn[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "test1_driver_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    # --- main compare: power/current/rate ---
    m = df[df["dataset"] == "main"].copy()
    if len(m):
        m = m.sort_values("rate_pct_per_h")
        fig = plt.figure(figsize=(8.2, 3.2), dpi=160)
        ax = fig.add_subplot(111)
        x = np.arange(len(m))
        ax.bar(x - 0.2, m["P_mW_mean"].to_numpy(float), width=0.4, label="P_mean (mW)", color="#4C78A8", alpha=0.85)
        ax2 = ax.twinx()
        ax2.plot(x + 0.2, m["rate_pct_per_h"].to_numpy(float), marker="o", linewidth=2, color="#F58518", label="rate (%/h)")
        ax.set_xticks(x)
        ax.set_xticklabels(m["case_id"].tolist(), rotation=0, fontsize=9)
        ax.set_ylabel("平均功率 (mW)")
        ax2.set_ylabel("平均放电速率 (%/h)")
        ax.set_title("主数据典型段：功率与放电速率对照")
        # legends
        ax.legend(loc="upper left", frameon=False)
        ax2.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "main_power_compare.png", bbox_inches="tight")
        plt.close(fig)

    print("[OK] wrote driver summary figures to", str(OUT_DIR))


if __name__ == "__main__":
    main()

