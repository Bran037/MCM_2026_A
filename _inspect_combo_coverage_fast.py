from __future__ import annotations

import pandas as pd

from test1_combo_predict_search import add_local_rate, bri_bin, build_slices, cpu_bin, net_bin, CPU_CONFIGS


def main() -> None:
    panel = pd.read_csv("MCM_2026_A/processed/test1/test1_panel_1min.csv", parse_dates=["time"]).sort_values("time")
    d = add_local_rate(panel)
    for cfg in CPU_CONFIGS:
        dd = d.copy()
        dd["cpu_bin"] = cpu_bin(dd["cpu_load"], cfg.cpu_edges_pct)
        dd["net_bin"] = net_bin(dd["net_type_code"])
        dd["bri_bin"] = bri_bin(dd["screen_on"], dd["brightness_state"], cfg.bri_mode)
        dd = dd.dropna(subset=["cpu_bin"]).copy()
        dd["cpu_bin"] = dd["cpu_bin"].astype(int)
        n_bri = int(cfg.bri_mode)
        dd["combo_id"] = dd["cpu_bin"] * (n_bri * 3) + dd["bri_bin"].astype(int) * 3 + dd["net_bin"].astype(int)
        sl = build_slices(dd, min_run_min=15)
        n_slices = int(len(sl))
        n_combos = int(sl["combo_id"].nunique()) if n_slices else 0
        print("cfg", cfg.cpu_edges_pct, "bri", cfg.bri_mode, "slices", n_slices, "combos", n_combos)


if __name__ == "__main__":
    main()

