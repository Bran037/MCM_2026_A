from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "processed" / "test1" / "fuzzy_predict_long" / "metrics_by_interval.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "fuzzy_predict_long" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["key"] = df["episode_id"].astype(int).astype(str) + "-" + df["interval_id"].astype(int).astype(str)

    # 1) Acc bar chart
    d = df.sort_values("acc_time").copy()
    fig = plt.figure(figsize=(9.6, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.bar(d["key"], d["acc_time"], color="#4C78A8", alpha=0.9)
    ax.axhline(0.90, color="#F58518", linewidth=2, alpha=0.9, label="0.90 target")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Acc")
    ax.set_xlabel("episode-interval")
    ax.set_title("test_1 long-interval leave-one-out: Acc by interval")
    ax.legend(loc="lower right", frameon=False)
    for x, a in zip(d["key"], d["acc_time"]):
        ax.text(x, min(0.98, float(a) + 0.02), f"{a:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acc_by_interval_bar.png", bbox_inches="tight")
    plt.close(fig)

    # 2) True vs pred duration scatter
    x = df["true_dur_h"].to_numpy(float)
    y = df["pred_dur_h"].to_numpy(float)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    fig = plt.figure(figsize=(4.8, 4.8), dpi=160)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=40, alpha=0.75, color="#4C78A8")
    ax.plot([lo, hi], [lo, hi], color="#F58518", linewidth=2, alpha=0.9)
    for k, xi, yi in zip(df["key"], x, y):
        ax.text(float(xi), float(yi), str(k), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("true duration (h)")
    ax.set_ylabel("pred duration (h)")
    ax.set_title("test_1 long-interval: true vs pred duration")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "true_vs_pred_duration.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Relative error histogram
    rel_err = (df["pred_dur_h"] - df["true_dur_h"]) / df["true_dur_h"]
    fig = plt.figure(figsize=(6.2, 3.6), dpi=160)
    ax = fig.add_subplot(111)
    ax.hist(rel_err.to_numpy(float), bins=10, color="#4C78A8", alpha=0.85)
    ax.axvline(0.0, color="#F58518", linewidth=2, alpha=0.9)
    ax.set_xlabel("(pred-true)/true")
    ax.set_ylabel("count")
    ax.set_title("test_1 long-interval: relative duration error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "relative_error_hist.png", bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote long-interval plots to", str(OUT_DIR))


if __name__ == "__main__":
    main()

