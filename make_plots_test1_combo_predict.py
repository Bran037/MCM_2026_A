from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def _set_chinese_font() -> None:
    """
    Ensure Chinese text renders correctly on Windows by selecting an installed CJK font.
    Falls back to default if none found, but this should work on most Win10 setups.
    """
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
    # avoid minus sign being shown as square
    plt.rcParams["axes.unicode_minus"] = False


def _sanitize_text(s: str) -> str:
    # Some fonts lack U+2011 (non-breaking hyphen). Replace with ASCII hyphen.
    return str(s).replace("\u2011", "-")


def _ensure_figdir(out_dir: Path) -> Path:
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    return figdir


def _plot_combo_bars(combo_df: pd.DataFrame, *, title: str, figpath: Path) -> None:
    d = combo_df.sort_values("acc_mean").copy()
    labels = d["combo_name"].tolist()
    acc = d["acc_mean"].to_numpy(float)
    mins = d["total_min"].to_numpy(float)

    fig = plt.figure(figsize=(11, 4.6), dpi=160)
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))
    ax.barh(y, acc, color="#4C78A8", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Acc（按组合的切片平均准确率）")
    ax.set_title(_sanitize_text(title))
    for i, (a, m) in enumerate(zip(acc, mins)):
        ax.text(min(0.98, a + 0.02), i, f"{a:.3f}  (min={int(m)}m)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_box(slice_df: pd.DataFrame, *, title: str, figpath: Path) -> None:
    # keep only combos present
    combos = slice_df["combo_name"].unique().tolist()
    # sort combos by mean acc
    means = slice_df.groupby("combo_name")["acc_time"].mean().sort_values()
    combos = means.index.tolist()
    data = [slice_df.loc[slice_df["combo_name"] == c, "acc_time"].to_numpy(float) for c in combos]

    fig = plt.figure(figsize=(11, 4.8), dpi=160)
    ax = fig.add_subplot(111)
    ax.boxplot(
        data,
        vert=False,
        labels=combos,
        showfliers=False,
        medianprops={"color": "#F58518", "linewidth": 2},
        boxprops={"color": "#4C78A8"},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Acc（切片级）")
    ax.set_title(_sanitize_text(title))
    fig.tight_layout()
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def _plot_true_pred_scatter(slice_df: pd.DataFrame, *, title: str, figpath: Path) -> None:
    d = slice_df.copy()
    d = d[np.isfinite(d["true_rate_pct_per_h"]) & np.isfinite(d["pred_rate_pct_per_h"])].copy()
    x = d["true_rate_pct_per_h"].to_numpy(float)
    y = d["pred_rate_pct_per_h"].to_numpy(float)

    fig = plt.figure(figsize=(5.2, 5.2), dpi=160)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=14, alpha=0.55, color="#4C78A8")
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], color="#F58518", linewidth=2, alpha=0.9)
    ax.set_xlabel("true rate (%/hour)")
    ax.set_ylabel("pred rate (%/hour)")
    ax.set_title(_sanitize_text(title))
    fig.tight_layout()
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def _summarize_combo_table(combo_df: pd.DataFrame, slice_df: pd.DataFrame) -> pd.DataFrame:
    # robust slice stats per combo
    g = slice_df.groupby(["combo_id", "combo_name"])
    q10 = g["acc_time"].quantile(0.1).rename("acc_p10").reset_index()
    q50 = g["acc_time"].quantile(0.5).rename("acc_median").reset_index()
    amin = g["acc_time"].min().rename("acc_min_slice").reset_index()
    amax = g["acc_time"].max().rename("acc_max_slice").reset_index()

    out = combo_df.merge(q10, on=["combo_id", "combo_name"], how="left")
    out = out.merge(q50, on=["combo_id", "combo_name"], how="left")
    out = out.merge(amin, on=["combo_id", "combo_name"], how="left")
    out = out.merge(amax, on=["combo_id", "combo_name"], how="left")
    out = out.sort_values("acc_mean")
    return out


def main() -> None:
    _set_chinese_font()
    jobs = [
        {
            "name": "KNN 模糊匹配（Top‑N 组合留一）",
            "out_dir": BASE_DIR / "processed" / "test1" / "combo_fuzzy_predict",
            "combo_csv": "metrics_by_combo.csv",
            "slice_csv": "metrics_by_slice.csv",
        },
        {
            "name": "Ridge（含SOC项）对照（Top‑N 组合留一）",
            "out_dir": BASE_DIR / "processed" / "test1" / "combo_ridge_predict",
            "combo_csv": "metrics_by_combo.csv",
            "slice_csv": "metrics_by_slice.csv",
        },
    ]

    for j in jobs:
        out_dir = Path(j["out_dir"])
        figdir = _ensure_figdir(out_dir)

        combo_df = pd.read_csv(out_dir / j["combo_csv"])
        slice_df = pd.read_csv(out_dir / j["slice_csv"])

        # Make sure combo_name is treated as utf-8 in downstream writes
        combo_df.to_csv(out_dir / j["combo_csv"], index=False, encoding="utf-8")
        slice_df.to_csv(out_dir / j["slice_csv"], index=False, encoding="utf-8")

        title = j["name"]
        _plot_combo_bars(combo_df, title=f"{title}：组合级准确率（按组合平均）", figpath=figdir / "acc_by_combo_barh.png")
        _plot_slice_box(slice_df, title=f"{title}：切片准确率分布（按组合）", figpath=figdir / "acc_by_combo_box.png")
        _plot_true_pred_scatter(slice_df, title=f"{title}：true vs pred 放电速率（切片级）", figpath=figdir / "true_vs_pred_rate.png")

        table = _summarize_combo_table(combo_df, slice_df)
        table.to_csv(out_dir / "combo_table_detailed.csv", index=False, encoding="utf-8")

        # Avoid Windows console encoding issues (GBK) by printing ASCII only
        print("[OK] wrote figures to", str(figdir))


if __name__ == "__main__":
    main()

