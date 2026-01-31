"""
Select pure-discharge segments from processed/test1/test1_panel_1min.csv.

Key requirement (from your plots + fit stability):
- NEVER mix charging into discharge segments. In this dataset charging can be gradual
  (e.g. +0.2%/min), so we need a sustained-rise detector, not only 1-min jumps.

Goal for this script:
- Output 9 segments total (3 figures × 3 segments), but prefer *big/continuous discharge*
  segments (visually similar to the main dataset long-segment plots), while still trying
  to maximize *condition variation* within each segment (CPU/net/brightness/temp changes).

Outputs (legacy-compatible filenames):
  processed/test1/segments/test1_segments_1min.csv
  processed/test1/segments/test1_segments_summary.csv
  processed/test1/segments/test1_segments_chosen16.csv   (same as summary; name kept for backward compatibility)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
IN_CSV = BASE_DIR / "processed" / "test1" / "test1_panel_1min.csv"
OUT_DIR = BASE_DIR / "processed" / "test1" / "segments"

# choose 3×3 segments total
TARGET_K = 9
GROUP_SIZE = 3


# 1) discharge detector: break on gaps, artifacts, and charging hints
GAP_BREAK_MIN = 15
SOC_JUMP_ABS_PCT = 4.0  # big per-minute jumps are artifacts/resets

# sustained charging detector (offline, look-ahead)
CHARGE_LOOKAHEAD_MIN = 15
CHARGE_RISE_PCT = 2.0  # if SOC rises by >=2% within next 15 minutes -> charging onset

# Big-segment selection (preferred)
BIG_MIN_LEN_MIN = 600        # >= 10 hours
BIG_MIN_DROP_PCT = 12.0
BIG_MAX_POS_JUMP_PCT = 1.0   # allow 1% bumps (quantization)
BIG_MIN_MONO_RATIO = 0.70

# If not enough "big" segments exist, split the longest discharge episodes into big chunks.
BIG_SPLIT_TARGET_LEN = 900   # 15 hours per chunk (approx)
BIG_SPLIT_MIN_LEN = 600      # don't create chunks shorter than 10 hours

def _segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    df["soc_pct"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    df["dt_min"] = df["time"].diff().dt.total_seconds().div(60.0)
    df["dsoc_pct"] = df["soc_pct"].diff()
    # charging state (may be missing; then treat as not plugged)
    if "battery_plugged" in df.columns:
        df["battery_plugged"] = pd.to_numeric(df["battery_plugged"], errors="coerce").fillna(0.0)
    else:
        df["battery_plugged"] = 0.0

    # break flags
    gap_break = df["dt_min"].fillna(1.0) > GAP_BREAK_MIN
    jump_break = df["dsoc_pct"].abs().fillna(0.0) >= SOC_JUMP_ABS_PCT
    plug_break = df["battery_plugged"].fillna(0.0) > 0.5

    # Charging can be gradual (e.g., +0.2%/min), so detect sustained upcoming increases.
    # We look ahead CHARGE_LOOKAHEAD_MIN and break *at the start* of such a rise.
    soc_fwd = df["soc_pct"].shift(-CHARGE_LOOKAHEAD_MIN)
    charge_break = (soc_fwd - df["soc_pct"]) >= CHARGE_RISE_PCT

    start_break = df.index == df.index.min()
    brk = gap_break | jump_break | plug_break | charge_break | start_break

    seg_id = brk.cumsum().astype(int) - 1  # start from 0
    df["seg_id"] = seg_id
    # IMPORTANT: within-segment diffs (boundary jumps removed)
    df["dsoc_seg_pct"] = df.groupby("seg_id")["soc_pct"].diff()
    return df


def _entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _count_switches(x: np.ndarray) -> int:
    if x.size < 2:
        return 0
    return int(np.sum(x[1:] != x[:-1]))


def _segment_stats(g: pd.DataFrame) -> Dict[str, float]:
    """Compute discharge cleanliness + condition-variation features for one candidate segment."""
    soc = pd.to_numeric(g["soc_pct"], errors="coerce").to_numpy(float)
    ds = pd.to_numeric(g["dsoc_seg_pct"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(g["net_type_code"], errors="coerce").fillna(0).astype(int).to_numpy()
    br = pd.to_numeric(g["brightness_state"], errors="coerce").to_numpy(float)
    scr = (br >= 0).astype(int)
    cpu = pd.to_numeric(g["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(g["cpu_freq_norm"], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(g["battery_temp_C"], errors="coerce").to_numpy(float)
    plugged = pd.to_numeric(g["battery_plugged"], errors="coerce").fillna(0).to_numpy(float)

    # charging flags inside segment (look-ahead, relative to segment local series)
    soc_fwd = pd.Series(soc).shift(-CHARGE_LOOKAHEAD_MIN).to_numpy(float)
    charge_flags = (soc_fwd - soc) >= CHARGE_RISE_PCT
    charge_flag_ratio = float(np.nanmean(charge_flags)) if charge_flags.size else 1.0
    plugged_ratio = float(np.nanmean(plugged > 0.5)) if plugged.size else 0.0

    drop = float(soc[0] - soc[-1]) if soc.size else float("nan")
    mono_ratio = float(np.nanmean(np.nan_to_num(ds, nan=0.0) <= 0.0)) if ds.size else 0.0
    pos_jump_max = float(np.nanmax(np.clip(ds, 0, None))) if ds.size else 0.0
    # avoid "full-battery plateau" segments: require some early/late drop
    if soc.size:
        k = int(min(120, soc.size - 1))  # first 2 hours
        early_drop_120 = float(soc[0] - soc[k])
        late_drop_120 = float(soc[max(0, soc.size - 1 - k)] - soc[-1])
    else:
        early_drop_120 = 0.0
        late_drop_120 = 0.0
    # flat ratio (SOC almost constant) — helps exclude long plateaus
    flat_ratio = float(np.nanmean(np.abs(np.nan_to_num(ds, nan=0.0)) < 0.05)) if ds.size else 1.0

    cpu_std = float(np.nanstd(cpu))
    f_std = float(np.nanstd(f))
    temp_range = float(np.nanmax(temp) - np.nanmin(temp)) if temp.size else 0.0

    counts = np.bincount(net.clip(0, 2), minlength=3).astype(float)
    probs = counts / max(1.0, float(counts.sum()))
    net_entropy = _entropy_from_probs(probs)
    net_switches = _count_switches(net)
    scr_switches = _count_switches(scr)

    br_on = np.where(br >= 0, br, np.nan)
    br_std = float(np.nanstd(br_on))

    # score: favor "big but diverse"
    score = (
        2.0 * net_entropy
        + 0.01 * net_switches
        + 0.01 * scr_switches
        + 2.0 * cpu_std
        + 0.6 * f_std
        + 0.2 * (temp_range / 5.0)
        + 0.8 * br_std
        + 0.2 * (max(0.0, drop) / 10.0)
    )

    return {
        "drop_pct": float(drop),
        "early_drop_120": float(early_drop_120),
        "late_drop_120": float(late_drop_120),
        "flat_ratio": float(flat_ratio),
        "mono_ratio": float(mono_ratio),
        "pos_jump_max": float(pos_jump_max),
        "charge_flag_ratio": float(charge_flag_ratio),
        "plugged_ratio": float(plugged_ratio),
        "cpu_std": float(cpu_std),
        "f_std": float(f_std),
        "temp_range": float(temp_range),
        "net_entropy": float(net_entropy),
        "net_switches": float(net_switches),
        "scr_switches": float(scr_switches),
        "br_std": float(br_std),
        "score": float(score),
    }


def _big_segment_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build big discharge candidates directly from coarse discharge episodes.
    If big episodes are too few, split long episodes into big chunks.
    """
    df = df.sort_values(["seg_id", "time"]).copy()

    # start from coarse discharge episodes (seg_id from _segment)
    eps = []
    for coarse_id, g0 in df.groupby("seg_id"):
        g0 = g0.sort_values("time").reset_index(drop=True)
        n = len(g0)
        if n < BIG_MIN_LEN_MIN:
            continue
        # If very long, split into big chunks
        if n >= 2 * BIG_SPLIT_MIN_LEN:
            starts = list(range(0, n - BIG_SPLIT_MIN_LEN + 1, BIG_SPLIT_TARGET_LEN))
            # ensure at least one window
            if not starts:
                starts = [0]
            for s in starts:
                e = min(n, s + BIG_SPLIT_TARGET_LEN)
                if e - s < BIG_SPLIT_MIN_LEN:
                    continue
                gg = g0.iloc[s:e].copy()
                st = _segment_stats(gg)
                eps.append(
                    {
                        "coarse_seg_id": int(coarse_id),
                        "start_idx": int(s),
                        "end_idx": int(e - 1),
                        "start_time": gg["time"].iloc[0],
                        "end_time": gg["time"].iloc[-1],
                        "n": int(len(gg)),
                        **st,
                    }
                )
        else:
            st = _segment_stats(g0)
            eps.append(
                {
                    "coarse_seg_id": int(coarse_id),
                    "start_idx": 0,
                    "end_idx": int(n - 1),
                    "start_time": g0["time"].iloc[0],
                    "end_time": g0["time"].iloc[-1],
                    "n": int(n),
                    **st,
                }
            )
    return pd.DataFrame(eps)

    soc = pd.to_numeric(g["soc_pct"], errors="coerce").to_numpy(float)
    ds = pd.to_numeric(g["dsoc_seg_pct"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(g["net_type_code"], errors="coerce").fillna(0).astype(int).to_numpy()
    br = pd.to_numeric(g["brightness_state"], errors="coerce").to_numpy(float)
    scr = (br >= 0).astype(int)
    cpu = pd.to_numeric(g["cpu_load"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(g["cpu_freq_norm"], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(g["battery_temp_C"], errors="coerce").to_numpy(float)
    plugged = pd.to_numeric(g["battery_plugged"], errors="coerce").fillna(0).to_numpy(float)

    # charging flags inside window
    soc_fwd = pd.Series(soc).shift(-CHARGE_LOOKAHEAD_MIN).to_numpy(float)
    charge_flags = (soc_fwd - soc) >= CHARGE_RISE_PCT
    charge_flag_ratio = float(np.nanmean(charge_flags)) if charge_flags.size else 1.0
    plugged_ratio = float(np.nanmean(plugged > 0.5)) if plugged.size else 0.0

    drop = float(soc[0] - soc[-1]) if soc.size else float("nan")
    mono_ratio = float(np.nanmean(np.nan_to_num(ds, nan=0.0) <= 0.0)) if ds.size else 0.0
    pos_jump_max = float(np.nanmax(np.clip(ds, 0, None))) if ds.size else 0.0

    # diversity / condition changes
    cpu_std = float(np.nanstd(cpu))
    f_std = float(np.nanstd(f))
    temp_range = float(np.nanmax(temp) - np.nanmin(temp)) if temp.size else 0.0

    # net entropy (0/1/2)
    counts = np.bincount(net.clip(0, 2), minlength=3).astype(float)
    probs = counts / max(1.0, float(counts.sum()))
    net_entropy = _entropy_from_probs(probs)
    net_switches = _count_switches(net)
    scr_switches = _count_switches(scr)

    # brightness variability when screen on
    br_on = np.where(br >= 0, br, np.nan)
    br_std = float(np.nanstd(br_on))

    # score: encourage many switches + entropy + variability + adequate drop
    score = (
        2.0 * net_entropy
        + 0.01 * net_switches
        + 0.01 * scr_switches
        + 2.0 * cpu_std
        + 0.6 * f_std
        + 0.2 * (temp_range / 5.0)
        + 0.8 * br_std
        + 0.2 * (max(0.0, drop) / 10.0)
    )

    return {
        "seg_id": seg_id,
        "start_idx": s,
        "end_idx": e - 1,
        "start_time": g["time"].iloc[0],
        "end_time": g["time"].iloc[-1],
        "n": int(len(g)),
        "soc_start": float(soc[0]),
        "soc_end": float(soc[-1]),
        "drop_pct": drop,
        "mono_ratio": mono_ratio,
        "pos_jump_max": pos_jump_max,
        "charge_flag_ratio": charge_flag_ratio,
        "plugged_ratio": plugged_ratio,
        "cpu_std": cpu_std,
        "temp_range": temp_range,
        "net_entropy": net_entropy,
        "net_switches": net_switches,
        "scr_switches": scr_switches,
        "br_std": br_std,
        "score": float(score),
    }


def _pick_best_nonoverlap(cands: pd.DataFrame, k: int) -> pd.DataFrame:
    cands = cands.sort_values(["score", "drop_pct", "n"], ascending=[False, False, False]).copy()
    chosen: List[Dict[str, object]] = []
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    def overlaps(a: Tuple[pd.Timestamp, pd.Timestamp], b: Tuple[pd.Timestamp, pd.Timestamp]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for _, r in cands.iterrows():
        itv = (r["start_time"], r["end_time"])
        if any(overlaps(itv, j) for j in intervals):
            continue
        chosen.append(r.to_dict())
        intervals.append(itv)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        # fallback: allow overlap (take top-k)
        chosen = [r.to_dict() for _, r in cands.head(k).iterrows()]
    out = pd.DataFrame(chosen).reset_index(drop=True)
    out["chosen_rank"] = np.arange(1, len(out) + 1)
    return out


def _materialize_chosen(df: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for i, (_, w) in enumerate(chosen.iterrows()):
        coarse_id = int(w["coarse_seg_id"])
        s = int(w["start_idx"])
        e = int(w["end_idx"]) + 1
        g = df[df["seg_id"] == coarse_id].sort_values("time").reset_index(drop=True).iloc[s:e].copy()
        # IMPORTANT: each chosen window becomes its own segment id (0..K-1)
        g["orig_seg_id"] = int(coarse_id)
        g["win_start_idx"] = int(s)
        g["win_end_idx"] = int(e - 1)
        g["seg_id"] = int(i)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, parse_dates=["time"])
    df = df.sort_values("time")
    df = df.dropna(subset=["time"])
    df = _segment(df)

    # build big discharge candidates and pick 9 non-overlapping segments
    cand = _big_segment_candidates(df)
    if cand.empty:
        raise RuntimeError("No discharge candidates found; check segmentation thresholds.")

    def filt(
        c: pd.DataFrame,
        *,
        min_len: int,
        min_drop: float,
        min_early: float,
        min_late: float,
        max_flat: float,
    ) -> pd.DataFrame:
        return c[
            (c["n"] >= min_len)
            & (c["drop_pct"] >= min_drop)
            & (c["early_drop_120"] >= min_early)
            & (c["late_drop_120"] >= min_late)
            & (c["flat_ratio"] <= max_flat)
            & (c["mono_ratio"] >= BIG_MIN_MONO_RATIO)
            & (c["pos_jump_max"] <= BIG_MAX_POS_JUMP_PCT)
            & (c["plugged_ratio"] <= 0.001)
            & (c["charge_flag_ratio"] <= 0.02)
        ].copy()

    # staged relaxation to ensure we can always pick 9 "big-ish" segments
    stages = [
        # strict: 10h+, avoid plateaus
        dict(min_len=BIG_MIN_LEN_MIN, min_drop=BIG_MIN_DROP_PCT, min_early=1.0, min_late=1.0, max_flat=0.85),
        # relax plateau a bit
        dict(min_len=BIG_MIN_LEN_MIN, min_drop=BIG_MIN_DROP_PCT, min_early=0.8, min_late=0.8, max_flat=0.90),
        # allow 8h+ segments
        dict(min_len=480, min_drop=max(10.0, BIG_MIN_DROP_PCT * 0.7), min_early=0.8, min_late=0.8, max_flat=0.92),
        # last resort: still big-ish, but keep purity constraints
        dict(min_len=420, min_drop=max(8.0, BIG_MIN_DROP_PCT * 0.6), min_early=0.5, min_late=0.5, max_flat=0.95),
    ]

    cand_f = pd.DataFrame()
    for st in stages:
        cand_f = filt(cand, **st)
        if len(cand_f) >= TARGET_K:
            break

    if len(cand_f) < TARGET_K:
        raise RuntimeError(f"Need {TARGET_K} big discharge segments, got {len(cand_f)} after relaxation.")

    chosen = _pick_best_nonoverlap(cand_f, k=TARGET_K)
    chosen_points = _materialize_chosen(df, chosen).copy()
    # recompute dsoc for safety (now that seg_id is per-window)
    chosen_points["dsoc_seg_pct"] = chosen_points.groupby("seg_id")["soc_pct"].diff()

    # summary per chosen segment_id (0..K-1)
    seg_sum = []
    for sid, g in chosen_points.groupby("seg_id"):
        g = g.sort_values("time")
        soc = pd.to_numeric(g["soc_pct"], errors="coerce").to_numpy(float)
        seg_sum.append(
            {
                "seg_id": int(sid),
                "start_time": g["time"].iloc[0],
                "end_time": g["time"].iloc[-1],
                "n": int(len(g)),
                "soc_start": float(soc[0]),
                "soc_end": float(soc[-1]),
                "drop_pct": float(soc[0] - soc[-1]),
                "orig_seg_id": int(g["orig_seg_id"].iloc[0]),
                "win_start_idx": int(g["win_start_idx"].iloc[0]),
                "win_end_idx": int(g["win_end_idx"].iloc[0]),
            }
        )
    sum2 = pd.DataFrame(seg_sum).sort_values("start_time").reset_index(drop=True)

    # legacy outputs expected by fitter
    out_points = OUT_DIR / "test1_segments_1min.csv"
    out_summary = OUT_DIR / "test1_segments_summary.csv"
    out_chosen = OUT_DIR / "test1_segments_chosen16.csv"
    chosen_points.to_csv(out_points, index=False)
    sum2.to_csv(out_summary, index=False)
    sum2.to_csv(out_chosen, index=False)

    print(f"[OK] wrote {out_points}")
    print(f"[OK] wrote {out_summary}")
    print(f"[OK] wrote {out_chosen}")
    print(f"[OK] chosen segments: {len(sum2)}")


if __name__ == "__main__":
    main()

