"""
Make overview plots for processed/test1/test1_panel_1min.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
IN_CSV = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "figures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV, parse_dates=["time"])
    df = df.sort_values("time")

    t = df["time"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df["battery_level_pct"], lw=1.2, color="black")
    ax1.set_ylabel("Battery level (%)")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t, df["cpu_load"], lw=1.0, color="#1f77b4", label="CPU load (0-1)")
    ax2b = ax2.twinx()
    ax2b.plot(t, df["cpu_freq_norm"], lw=1.0, color="#ff7f0e", label="CPU freq norm (0-1)")
    ax2.set_ylabel("CPU load")
    ax2b.set_ylabel("CPU freq norm")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t, df["battery_temp_C"], lw=1.0, color="#d62728")
    ax3.set_ylabel("Battery temp (°C)")
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    # Requested brightness encoding:
    #   screen off -> -1, screen on -> [0,1]
    ax4.plot(t, df["brightness_state"], lw=1.0, color="#2ca02c")
    ax4b = ax4.twinx()
    ax4b.step(t, df["net_type_code"], where="post", lw=0.9, color="black", alpha=0.55)
    ax4.set_ylabel("Brightness state (-1 off, 0-1 on)")
    ax4b.set_ylabel("Net type (0 none / 1 mobile / 2 wi-fi)")
    ax4.set_ylim(-1.1, 1.05)
    ax4b.set_ylim(-0.2, 2.2)
    ax4.grid(True, alpha=0.25)

    fig.suptitle("test_1 unified panel (1-min): battery, CPU, temperature, screen/brightness, network", y=0.995)
    out1 = OUT_DIR / "test1_overview.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Missingness summary bar plot
    miss = df.isna().mean().sort_values(ascending=False)
    fig2 = plt.figure(figsize=(14, 5))
    ax = fig2.add_subplot(1, 1, 1)
    ax.bar(miss.index, miss.values)
    ax.set_ylabel("Missing rate")
    ax.set_title("Missingness by column (test1_panel_1min)")
    ax.set_ylim(0, min(1.0, float(miss.max()) * 1.15 + 1e-6))
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    out2 = OUT_DIR / "test1_missingness.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"[OK] wrote {out1}")
    print(f"[OK] wrote {out2}")


if __name__ == "__main__":
    main()

