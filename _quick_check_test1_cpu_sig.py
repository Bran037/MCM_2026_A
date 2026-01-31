from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    rate_path = Path("MCM_2026_A/processed/test1/significance/rate_panel_logr.csv")
    p = pd.read_csv(rate_path, parse_dates=["time"])

    cols = ["log_r", "cpu_load", "cpu_freq_norm", "cpu_x", "scr", "bright", "net_wifi", "net_mobile", "T0"]
    print("n =", len(p))
    print("\nstd (ascending):")
    print(p[cols].std(numeric_only=True).sort_values())

    print("\nCorr with log_r:")
    for c in cols[1:]:
        r = np.corrcoef(p["log_r"], p[c])[0, 1]
        print(f"{c:12s} {r: .4f}")

    print("\nPairwise corr: cpu_load vs others")
    for c in ["cpu_freq_norm", "cpu_x", "scr", "bright", "net_wifi", "net_mobile", "T0"]:
        r = np.corrcoef(p["cpu_load"], p[c])[0, 1]
        print(f"cpu_load vs {c:12s} {r: .4f}")

    print("\nPairwise corr: cpu_x vs others")
    for c in ["scr", "bright", "net_wifi", "net_mobile", "T0"]:
        r = np.corrcoef(p["cpu_x"], p[c])[0, 1]
        print(f"cpu_x    vs {c:12s} {r: .4f}")

    fp = Path("MCM_2026_A/processed/test1/episodes_fit/fit_params.csv")
    if fp.exists():
        params = pd.read_csv(fp).set_index("param")["value"]
        gamma = float(params["gamma"])
        cpu_mech = p["cpu_load"] * np.clip(p["cpu_freq_norm"], 0, 1) ** gamma
        r = np.corrcoef(p["log_r"], cpu_mech)[0, 1]
        print(f"\nmech cpu_load*freq^gamma (gamma={gamma:.3f}) corr with log_r: {r:.4f}")


if __name__ == "__main__":
    main()

