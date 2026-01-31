"""
Prepare continuous "episodes" for test_1 plots:
- Each episode shows a continuous time span, including intervening charging portions.
- Within each episode we identify 3 discharge intervals (start/end) robustly.
- Model fitting / error metrics will be computed ONLY on discharge intervals.

Outputs:
  processed/test1/episodes/test1_episode_points_1min.csv
  processed/test1/episodes/test1_episode_intervals.csv
  processed/test1/episodes/test1_episode_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
IN_CSV = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "episodes"

# Desired structure: 3 episodes × 3 discharge intervals each
N_EPISODES = 3
INTERVALS_PER_EPISODE = 3

# Discharge detection (minute grid)
SMOOTH_WIN = 11  # rolling median, minutes
FWD_WIN = 60  # minutes; use future net drop to define discharge (robust to SOC quantization plateaus)
FWD_DROP_PCT = 1.0  # if SOC drops by >=1% in next 60 minutes => we consider it "actively discharging"
GAP_FILL_MIN = 20  # fill short gaps inside discharge mask (minutes)

CHARGE_LOOKAHEAD_MIN = 15
CHARGE_RISE_PCT = 1.0  # % within lookahead => charging/plateau-on-charge

MIN_INTERVAL_LEN_MIN = 180  # 3 hours
MIN_INTERVAL_DROP_PCT = 6.0

# refine discharge-interval start (trim only a little; avoid changing interval content)
REFINE_START_WIN = 30      # minutes
REFINE_START_DROP = 0.3    # % drop within next 30min required to accept point as discharge start
REFINE_SLOPE_WIN = 5       # minutes
REFINE_SLOPE_TH = -0.005   # %/min slope over 5min required (slight negative)
REFINE_MAX_SHIFT = 90      # minutes; never move start forward by more than this

# episode selection: keep plots compact (avoid huge "decorative" spans)
# NOTE: too small a cap can leave us with <3 feasible episodes. We cap "extreme" spans but
# still penalize long spans strongly in the score.
MAX_EPISODE_SPAN_MIN = 8000  # minutes (~5.5 days)
SPAN_PENALTY = 0.002  # penalty per minute in episode score

# Interval scoring weights (condition variation inside interval)
W_NET_ENT = 2.0
W_NET_SW = 0.01
W_SCR_SW = 0.01
W_CPU_STD = 2.0
W_F_STD = 0.6
W_TEMP_RANGE = 0.2 / 5.0
W_BR_STD = 0.8
W_DROP = 0.2 / 10.0


def _entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _count_switches(x: np.ndarray) -> int:
    if x.size < 2:
        return 0
    return int(np.sum(x[1:] != x[:-1]))


def _compute_interval_score(g: pd.DataFrame) -> float:
    soc = pd.to_numeric(g["battery_level_pct"], errors="coerce").to_numpy(float)
    cpu = pd.to_numeric(g["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(g["cpu_freq_norm"], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(g["battery_temp_C"], errors="coerce").to_numpy(float)
    br = pd.to_numeric(g["brightness_state"], errors="coerce").to_numpy(float)
    scr = (br >= 0).astype(int)
    net = pd.to_numeric(g["net_type_code"], errors="coerce").fillna(0).astype(int).to_numpy()

    drop = float(soc[0] - soc[-1]) if soc.size else 0.0
    cpu_std = float(np.nanstd(cpu))
    f_std = float(np.nanstd(f))
    temp_range = float(np.nanmax(temp) - np.nanmin(temp)) if temp.size else 0.0
    br_on = np.where(br >= 0, br, np.nan)
    br_std = float(np.nanstd(br_on))
    # guard against NaNs (e.g., no screen-on samples)
    cpu_std = float(np.nan_to_num(cpu_std, nan=0.0))
    f_std = float(np.nan_to_num(f_std, nan=0.0))
    temp_range = float(np.nan_to_num(temp_range, nan=0.0))
    br_std = float(np.nan_to_num(br_std, nan=0.0))

    counts = np.bincount(net.clip(0, 2), minlength=3).astype(float)
    probs = counts / max(1.0, float(counts.sum()))
    net_ent = _entropy_from_probs(probs)
    net_sw = _count_switches(net)
    scr_sw = _count_switches(scr)

    score = (
        W_NET_ENT * net_ent
        + W_NET_SW * net_sw
        + W_SCR_SW * scr_sw
        + W_CPU_STD * cpu_std
        + W_F_STD * f_std
        + W_TEMP_RANGE * temp_range
        + W_BR_STD * br_std
        + W_DROP * max(0.0, drop)
    )
    return float(np.nan_to_num(score, nan=0.0))


def _detect_discharge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    df["soc"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    df["dt_min"] = df["time"].diff().dt.total_seconds().div(60.0).fillna(1.0)
    if "battery_plugged" in df.columns:
        df["plugged"] = pd.to_numeric(df["battery_plugged"], errors="coerce").fillna(0.0)
    else:
        df["plugged"] = 0.0

    # Smooth SOC to make slope more robust to quantization.
    df["soc_smooth"] = df["soc"].rolling(SMOOTH_WIN, center=True, min_periods=1).median()

    # Charging flags: plugged OR future rise
    soc_fwd = df["soc_smooth"].shift(-CHARGE_LOOKAHEAD_MIN)
    df["charge_flag"] = (df["plugged"] > 0.5) | ((soc_fwd - df["soc_smooth"]) >= CHARGE_RISE_PCT)

    # Discharge candidate: in the next FWD_WIN minutes, net SOC decreases enough.
    # This is robust to minute-level SOC quantization plateaus.
    soc_fwd2 = df["soc_smooth"].shift(-FWD_WIN)
    df["fwd_drop"] = df["soc_smooth"] - soc_fwd2  # positive means drop
    discharge_cand = (df["fwd_drop"] >= FWD_DROP_PCT) & (~df["charge_flag"])
    discharge_cand = discharge_cand.fillna(False)

    # Fill short gaps inside discharge (e.g., brief plateaus/measurement artifacts)
    if GAP_FILL_MIN > 0:
        win = 2 * GAP_FILL_MIN + 1
        discharge = discharge_cand.rolling(win, center=True, min_periods=1).max().astype(bool)
    else:
        discharge = discharge_cand.astype(bool)

    # Never allow discharge to extend into charging flags (gap-fill must not bridge charge periods)
    discharge = discharge & (~df["charge_flag"].fillna(False).to_numpy(bool))

    df["is_discharge"] = discharge.astype(int)

    # Extract intervals (start/end indices)
    intervals = []
    i = 0
    while i < len(df):
        if df["is_discharge"].iloc[i] != 1:
            i += 1
            continue
        j = i
        while j < len(df) and df["is_discharge"].iloc[j] == 1:
            j += 1
        g = df.iloc[i:j].copy()

        # --- refine start: move start forward to the first point with clear short-horizon drop ---
        if len(g) > (REFINE_START_WIN + REFINE_SLOPE_WIN + 2):
            # only search within the first REFINE_MAX_SHIFT minutes
            search_n = int(min(len(g) - (REFINE_START_WIN + 1), REFINE_MAX_SHIFT))
            if search_n > 5:
                soc = g["soc_smooth"].to_numpy(float)
                # forward drop over REFINE_START_WIN (use nan padding, no wrap-around)
                fwd = np.full_like(soc, np.nan, dtype=float)
                fwd[: len(soc) - REFINE_START_WIN] = soc[REFINE_START_WIN:]
                fwd_drop = soc - fwd
                # slope over REFINE_SLOPE_WIN (backward diff)
                prev = np.full_like(soc, np.nan, dtype=float)
                prev[REFINE_SLOPE_WIN:] = soc[:-REFINE_SLOPE_WIN]
                slope = (soc - prev) / float(REFINE_SLOPE_WIN)
                ok = (fwd_drop >= REFINE_START_DROP) & (slope <= REFINE_SLOPE_TH)
                ok = ok[:search_n]
                if np.any(ok):
                    k0 = int(np.argmax(ok))  # first True
                    if k0 > 0:
                        g = g.iloc[k0:].copy()
        if len(g) >= MIN_INTERVAL_LEN_MIN:
            drop = float(g["soc"].iloc[0] - g["soc"].iloc[-1])
            if drop >= MIN_INTERVAL_DROP_PCT:
                intervals.append(
                    {
                        "interval_id": len(intervals),
                        "start_time": g["time"].iloc[0],
                        "end_time": g["time"].iloc[-1],
                        "n": int(len(g)),
                        "soc_start": float(g["soc"].iloc[0]),
                        "soc_end": float(g["soc"].iloc[-1]),
                        "drop_pct": float(drop),
                        "score": _compute_interval_score(g),
                    }
                )
        i = j

    return pd.DataFrame(intervals)


def _pick_episode_triples(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Build candidate episodes from consecutive triples of discharge intervals.
    Episode = [i, i+1, i+2] + continuous time span between their start/end (includes charging gaps).
    Pick top N_EPISODES non-overlapping episodes by score.
    """
    if intervals.empty:
        raise RuntimeError("No discharge intervals detected; try relaxing thresholds.")
    intervals = intervals.sort_values("start_time").reset_index(drop=True).copy()

    cand = []
    for i in range(0, len(intervals) - INTERVALS_PER_EPISODE + 1):
        block = intervals.iloc[i : i + INTERVALS_PER_EPISODE].copy()
        start = block["start_time"].min()
        end = block["end_time"].max()
        span_min = float((end - start).total_seconds() / 60.0)
        total_len = float(block["n"].sum())
        total_drop = float(block["drop_pct"].sum())
        # Penalize long spans strongly to avoid huge blank areas.
        score = float(block["score"].sum() + 0.002 * total_len + 0.05 * total_drop - SPAN_PENALTY * span_min)
        cand.append(
            {
                "episode_id": len(cand),
                "i0": int(i),
                "start_time": start,
                "end_time": end,
                "score": score,
                "span_min": span_min,
                "interval_ids": ",".join(str(int(x)) for x in block["interval_id"].tolist()),
            }
        )

    cand_df = pd.DataFrame(cand)
    # prefer compact episodes
    cand_df2 = cand_df[cand_df["span_min"] <= MAX_EPISODE_SPAN_MIN].copy()
    if cand_df2.empty:
        cand_df2 = cand_df.copy()
    cand_df2 = cand_df2.sort_values("score", ascending=False).reset_index(drop=True)

    chosen = []
    used: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    def overlaps(a: Tuple[pd.Timestamp, pd.Timestamp], b: Tuple[pd.Timestamp, pd.Timestamp]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for _, r in cand_df2.iterrows():
        itv = (r["start_time"], r["end_time"])
        if any(overlaps(itv, u) for u in used):
            continue
        chosen.append(r)
        used.append(itv)
        if len(chosen) >= N_EPISODES:
            break

    if len(chosen) < N_EPISODES:
        # fallback: take top-N even if overlaps
        chosen = [r for _, r in cand_df2.head(N_EPISODES).iterrows()]

    out = pd.DataFrame(chosen).reset_index(drop=True)
    out["episode_id"] = np.arange(len(out)).astype(int)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, parse_dates=["time"]).sort_values("time").dropna(subset=["time"])
    intervals = _detect_discharge_intervals(df)
    if intervals.empty:
        raise RuntimeError("No discharge intervals; relax thresholds.")

    episodes = _pick_episode_triples(intervals)
    # diagnostics: keep all detected intervals for debugging / reporting
    (OUT_DIR / "test1_discharge_intervals_detected.csv").write_text(
        intervals.to_csv(index=False), encoding="utf-8"
    )

    # materialize episode points and interval mapping
    point_rows = []
    interval_rows = []
    for epi_id, epi in episodes.iterrows():
        start = epi["start_time"]
        end = epi["end_time"]
        part = df[(df["time"] >= start) & (df["time"] <= end)].copy().sort_values("time").reset_index(drop=True)

        # compute discharge mask on this part by intersecting with chosen intervals
        ids = [int(x) for x in str(epi["interval_ids"]).split(",") if str(x).strip()]
        blocks = intervals[intervals["interval_id"].isin(ids)].sort_values("start_time").reset_index(drop=True)

        part["episode_id"] = int(epi_id)
        part["interval_id"] = -1
        part["is_discharge"] = 0

        for k, b in blocks.iterrows():
            m = (part["time"] >= b["start_time"]) & (part["time"] <= b["end_time"])
            part.loc[m, "interval_id"] = int(k)  # local interval index within episode (0..2)
            part.loc[m, "is_discharge"] = 1
            interval_rows.append(
                {
                    "episode_id": int(epi_id),
                    "interval_id": int(k),
                    "start_time": b["start_time"],
                    "end_time": b["end_time"],
                    "n": int(b["n"]),
                    "drop_pct": float(b["drop_pct"]),
                    "score": float(b["score"]),
                    "global_interval_id": int(b["interval_id"]),
                }
            )

        point_rows.append(part)

    pts = pd.concat(point_rows, ignore_index=True)

    keep_cols = [
        "time",
        "battery_level_pct",
        "battery_temp_C",
        "cpu_load",
        "cpu_freq_norm",
        "screen_on",
        "brightness_state",
        "net_type",
        "net_type_code",
        "episode_id",
        "interval_id",
        "is_discharge",
    ]
    # battery_plugged may not exist in panel; keep if present
    if "battery_plugged" in df.columns:
        keep_cols.insert(3, "battery_plugged")
    pts = pts[keep_cols].copy()

    out_points = OUT_DIR / "test1_episode_points_1min.csv"
    out_intervals = OUT_DIR / "test1_episode_intervals.csv"
    out_summary = OUT_DIR / "test1_episode_summary.csv"
    pts.to_csv(out_points, index=False)
    pd.DataFrame(interval_rows).sort_values(["episode_id", "interval_id"]).to_csv(out_intervals, index=False)
    episodes.to_csv(out_summary, index=False)

    print("[OK] wrote", out_points)
    print("[OK] wrote", out_intervals)
    print("[OK] wrote", out_summary)


if __name__ == "__main__":
    main()

