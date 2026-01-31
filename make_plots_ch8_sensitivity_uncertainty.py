from __future__ import annotations

"""
Chapter 8 Sensitivity & Uncertainty figures.

We generate several "pre-solve" uncertainty visualizations based on available data:
  1) Temperature coverage vs "extreme" (no data outside working range) histogram.
  2) SOC differencing noise: 5-min vs 30-min rate label distribution (test1).
  3) Parameter sensitivity (test1 episodes_fit params): tornado chart for time-to-empty proxy.
  4) Coverage/failure mode proxy: KNN neighbor-distance sparsity vs error (from long-interval results, indirect).

Outputs:
  figures/sensitivity_uncertainty/
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "figures" / "sensitivity_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST1_PANEL = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
TEST1_PARAMS = BASE_DIR / "processed" / "test1" / "episodes_fit" / "fit_params.csv"
MAIN_RATE_PANEL = BASE_DIR / "processed" / "discharge" / "rate_panel_logr_with_time.csv"
LONG_METRICS = BASE_DIR / "processed" / "test1" / "fuzzy_predict_long" / "metrics_by_interval.csv"


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


def _rate_from_soc_pct(soc_pct: np.ndarray, diff_min: int) -> np.ndarray:
    soc = soc_pct.astype(float)
    lag = np.roll(soc, diff_min)
    lag[:diff_min] = np.nan
    dsoc = soc - lag
    return np.where(dsoc < 0, (-dsoc) / (diff_min / 60.0), np.nan)


def fig_temp_coverage() -> None:
    t1 = pd.read_csv(TEST1_PANEL)
    t1T = pd.to_numeric(t1["battery_temp_C"], errors="coerce").to_numpy(float)

    main = pd.read_csv(MAIN_RATE_PANEL)
    # main has T0 = T-30
    mT = pd.to_numeric(main["T0"], errors="coerce").to_numpy(float) + 30.0

    fig = plt.figure(figsize=(8.4, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.hist(mT[np.isfinite(mT)], bins=35, alpha=0.55, label="主数据 T(°C)", color="#4C78A8")
    ax.hist(t1T[np.isfinite(t1T)], bins=35, alpha=0.55, label="test1 T(°C)", color="#F58518")
    # "working range" annotation (heuristic): [15, 45]
    ax.axvspan(-20, 10, color="#E45756", alpha=0.10, label="极冷(示意)")
    ax.axvspan(50, 80, color="#E45756", alpha=0.10, label="极热(示意)")
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel("分钟计数")
    ax.set_title("温度覆盖度：数据几乎全部处于工作温度区间（极端两端缺数据）")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "temp_coverage_hist.png", bbox_inches="tight")
    plt.close(fig)


def fig_diff_noise_test1() -> None:
    df = pd.read_csv(TEST1_PANEL, parse_dates=["time"]).sort_values("time")
    soc = pd.to_numeric(df["battery_level_pct"], errors="coerce").to_numpy(float)
    r5 = _rate_from_soc_pct(soc, diff_min=5)
    r30 = _rate_from_soc_pct(soc, diff_min=30)
    # keep positive rates only (discharge labels)
    r5 = r5[np.isfinite(r5) & (r5 > 0)]
    r30 = r30[np.isfinite(r30) & (r30 > 0)]

    fig = plt.figure(figsize=(8.4, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.hist(r5, bins=60, alpha=0.55, label="5min差分 rate(%/h)", color="#4C78A8", density=True)
    ax.hist(r30, bins=60, alpha=0.55, label="30min差分 rate(%/h)", color="#F58518", density=True)
    ax.set_xlabel("rate (%/h)")
    ax.set_ylabel("density")
    ax.set_title("差分窗敏感性：5min 差分噪声更大（SOC量化/平滑导致）")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diff_window_rate_density.png", bbox_inches="tight")
    plt.close(fig)


def _load_params() -> Dict[str, float]:
    p = pd.read_csv(TEST1_PARAMS)
    return {str(r.param): float(r.value) for r in p.itertuples(index=False)}


def _toy_time_to_drop(params: Dict[str, float], *, u: float, f: float, scr_on: float, net: str, T: float, drop_pct: float = 10.0) -> float:
    """
    A simple time-to-drop proxy (hours) using the multiplicative current model.
    Not a full SOC trajectory; used only for local sensitivity ranking.
    """
    I_idle = params["I_idle"]
    alpha_cpu = params["alpha_cpu"]
    gamma = params["gamma"]
    alpha_mob = params["alpha_mob"]
    alpha_wifi = params["alpha_wifi"]
    beta_T = params["beta_T"]
    delta_scr = params["delta_scr"]

    I = I_idle + alpha_cpu * u * (f ** gamma)
    k_scr = 1.0 + delta_scr * scr_on  # crude; scr_on in {0,1}
    k_net = alpha_wifi if net == "wifi" else alpha_mob
    k_T = float(np.exp(beta_T * (T - 30.0)))
    I_eff = I * k_scr * k_net * k_T

    # Convert current to %/h using k0: rate = k0 * I_eff  (k0 fits unit conversion)
    k0 = params["k0"]
    rate_pct_per_h = 100.0 * k0 * I_eff
    if rate_pct_per_h <= 1e-9:
        return np.nan
    return float(drop_pct / rate_pct_per_h)


def fig_param_tornado() -> None:
    params = _load_params()
    base = _toy_time_to_drop(params, u=0.25, f=0.95, scr_on=1.0, net="wifi", T=38.0, drop_pct=10.0)
    # perturb each key param by +/-10%
    keys = ["I_idle", "alpha_cpu", "gamma", "delta_scr", "alpha_wifi", "alpha_mob", "beta_T", "k0"]
    rows = []
    for k in keys:
        for sgn in (-1, +1):
            pp = dict(params)
            pp[k] = params[k] * (1.0 + 0.10 * sgn)
            t = _toy_time_to_drop(pp, u=0.25, f=0.95, scr_on=1.0, net="wifi", T=38.0, drop_pct=10.0)
            rows.append({"param": k, "sgn": sgn, "T_h": t})
    d = pd.DataFrame(rows)
    # compute range in relative change
    out = []
    for k in keys:
        t_lo = float(d[(d.param == k) & (d.sgn == -1)]["T_h"].iloc[0])
        t_hi = float(d[(d.param == k) & (d.sgn == +1)]["T_h"].iloc[0])
        out.append({"param": k, "dT_minus": (t_lo - base) / base, "dT_plus": (t_hi - base) / base})
    o = pd.DataFrame(out)
    o["span"] = o["dT_plus"] - o["dT_minus"]
    o = o.sort_values("span")

    fig = plt.figure(figsize=(8.6, 4.2), dpi=160)
    ax = fig.add_subplot(111)
    y = np.arange(len(o))
    ax.hlines(y, o["dT_minus"], o["dT_plus"], color="#4C78A8", linewidth=5, alpha=0.85)
    ax.vlines(0, -1, len(o), color="#F58518", linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(o["param"].tolist(), fontsize=9)
    ax.set_xlabel("相对变化 ΔT/T（±10% 参数扰动）")
    ax.set_title("参数敏感性（test1拟合参数）：time-to-drop 的 tornado 图（局部近似）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "param_sensitivity_tornado.png", bbox_inches="tight")
    plt.close(fig)


def fig_long_interval_error_bar() -> None:
    df = pd.read_csv(LONG_METRICS)
    df["key"] = df["episode_id"].astype(int).astype(str) + "-" + df["interval_id"].astype(int).astype(str)
    d = df.sort_values("acc_time").copy()
    fig = plt.figure(figsize=(8.6, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.bar(d["key"], d["acc_time"], color="#4C78A8", alpha=0.9)
    ax.axhline(0.90, color="#F58518", linewidth=2, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Acc")
    ax.set_title("失效情形示例：同一算法在不同长段落上误差分化明显（覆盖/分布漂移）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "failure_mode_long_interval_acc.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _set_chinese_font()
    fig_temp_coverage()
    fig_diff_noise_test1()
    fig_param_tornado()
    fig_long_interval_error_bar()
    print("[OK] wrote sensitivity/uncertainty figures to", str(OUT_DIR))


if __name__ == "__main__":
    main()

