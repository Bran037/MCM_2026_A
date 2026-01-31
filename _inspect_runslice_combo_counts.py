from __future__ import annotations

import pandas as pd

from test1_combo_predict_search_runslices import CPU_CONFIGS, bri_bin, build_run_slices, cpu_bin, net_bin


def main() -> None:
    panel = pd.read_csv("MCM_2026_A/processed/test1/test1_panel_1min.csv", parse_dates=["time"]).sort_values("time")
    panel = panel.dropna(subset=["battery_level_pct", "cpu_load", "screen_on", "net_type_code"]).copy()
    for cfg in CPU_CONFIGS:
        d = panel.copy()
        d["cpu_bin"] = cpu_bin(d["cpu_load"], cfg.cpu_edges_pct)
        d["net_bin"] = net_bin(d["net_type_code"])
        d["bri_bin"] = bri_bin(d["screen_on"], d["brightness_state"], cfg.bri_mode)
        d = d.dropna(subset=["cpu_bin"]).copy()
        d["cpu_bin"] = d["cpu_bin"].astype(int)
        n_bri = int(cfg.bri_mode)
        d["combo_id"] = d["cpu_bin"] * (n_bri * 3) + d["bri_bin"].astype(int) * 3 + d["net_bin"].astype(int)
        sl = build_run_slices(d, min_run_min=10, min_drop_pct=2.0)
        print("cfg", cfg.cpu_edges_pct, "bri", cfg.bri_mode, "slices", len(sl), "combos", (sl["combo_id"].nunique() if len(sl) else 0))


if __name__ == "__main__":
    main()

