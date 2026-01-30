from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IN_DIR = Path("processed") / "discharge"
FIG_DIR = Path("figures") / "four_segment_fits"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NETS = ["none", "wi-fi", "mobile"]
TREF_C = 30.0

# choose 4 segments per device
K = 4
# strict thresholds work for most devices; one device has shorter discharge segments,
# so we will fall back to relaxed thresholds if needed.
MIN_SEG_LEN_MIN = 40
MIN_SEG_DROP_PCT = 8.0
RELAX_MIN_SEG_LEN_MIN = 20
RELAX_MIN_SEG_DROP_PCT = 3.0

# how to build time axis for the combined plot
GAP_BETWEEN_SEGS_MIN = 30.0

# model levels (increasing complexity but still "same params across 4 segs")
# 0: baseline only (time)
# 1: add cumulative screen/net/temp bases
# 2: add interaction bases
LEVELS = [1, 2]

TARGET_R2 = 0.95


@dataclass(frozen=True)
class SegProfile:
    segment_id: int
    minutes: int
    soc_drop_pct: float
    screen_on_ratio: float
    net_mode: str
    temp_mean: float
    temp_range: float
    score: float


def load_device_discharge(device_id: str) -> pd.DataFrame:
    path = IN_DIR / f"{device_id}_discharge_1min.csv"
    df = pd.read_csv(path)
    # timestamp index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    else:
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64")
    df["battery_level_pct"] = pd.to_numeric(df["battery_level_pct"], errors="coerce")
    df["soc"] = pd.to_numeric(df.get("soc", df["battery_level_pct"] / 100.0), errors="coerce")
    if "battery_temp_C" in df.columns:
        df["battery_temp_C"] = pd.to_numeric(df["battery_temp_C"], errors="coerce")
    if "screen_on" in df.columns:
        df["screen_on"] = df["screen_on"].astype("boolean")
    if "network_type" in df.columns:
        df["network_type"] = df["network_type"].astype("string").str.lower()
    return df


def build_profiles(df: pd.DataFrame, *, min_len_min: int, min_drop_pct: float) -> list[SegProfile]:
    profs: list[SegProfile] = []
    for sid, g in df.groupby("segment_id", sort=True):
        if pd.isna(sid):
            continue
        g = g.dropna(subset=["battery_level_pct"]).copy()
        if len(g) < 3:
            continue
        minutes = int((g.index.max() - g.index.min()).total_seconds() // 60) + 1
        soc_drop = float(g["battery_level_pct"].iloc[0] - g["battery_level_pct"].iloc[-1])
        if minutes < min_len_min or soc_drop < min_drop_pct:
            continue
        scr_ratio = float(np.nanmean(g["screen_on"].astype(float))) if "screen_on" in g.columns else 0.0
        if "network_type" in g.columns and g["network_type"].notna().any():
            net_mode = str(g["network_type"].mode().iloc[0])
        else:
            net_mode = "none"
        temp = g["battery_temp_C"] if "battery_temp_C" in g.columns else pd.Series([np.nan] * len(g), index=g.index)
        temp_mean = float(np.nanmean(temp))
        temp_range = float(np.nanmax(temp) - np.nanmin(temp)) if temp.notna().any() else 0.0

        # score encourages long + large drop + within-segment variation (temp_range) + nontrivial screen ratio
        score = soc_drop * (1.0 + 0.5 * temp_range) * (1.0 + 2.0 * (scr_ratio * (1 - scr_ratio)))
        profs.append(
            SegProfile(
                segment_id=int(sid),
                minutes=minutes,
                soc_drop_pct=soc_drop,
                screen_on_ratio=scr_ratio,
                net_mode=net_mode,
                temp_mean=temp_mean,
                temp_range=temp_range,
                score=score,
            )
        )
    profs.sort(key=lambda x: x.score, reverse=True)
    return profs


def pick_diverse_four(profs: list[SegProfile]) -> list[int]:
    """
    Greedy selection to cover:
    - screen_on_ratio: low, mixed, high
    - net_mode: none/wifi/mobile if possible
    - temperature: low/medium/high (by mean)
    """
    if not profs:
        return []

    # binning
    def scr_bin(x: float) -> str:
        if x < 0.2:
            return "scr_low"
        if x > 0.8:
            return "scr_high"
        return "scr_mix"

    temps = [p.temp_mean for p in profs if np.isfinite(p.temp_mean)]
    if temps:
        t_lo, t_hi = float(np.nanpercentile(temps, 33)), float(np.nanpercentile(temps, 66))
    else:
        t_lo, t_hi = 0.0, 0.0

    def t_bin(x: float) -> str:
        if not np.isfinite(x):
            return "t_na"
        if x <= t_lo:
            return "t_low"
        if x >= t_hi:
            return "t_high"
        return "t_mid"

    picked: list[SegProfile] = []
    covered: set[str] = set()

    for _ in range(K):
        best = None
        best_gain = -1.0
        for p in profs:
            if any(p.segment_id == q.segment_id for q in picked):
                continue
            tags = {scr_bin(p.screen_on_ratio), f"net_{p.net_mode}", t_bin(p.temp_mean)}
            gain = len(tags - covered)
            # break ties by score
            if gain > best_gain or (gain == best_gain and (best is None or p.score > best.score)):
                best = p
                best_gain = gain
        if best is None:
            break
        picked.append(best)
        covered |= {scr_bin(best.screen_on_ratio), f"net_{best.net_mode}", t_bin(best.temp_mean)}

    # if still <4 (rare), pad by top scores
    if len(picked) < K:
        for p in profs:
            if len(picked) >= K:
                break
            if any(p.segment_id == q.segment_id for q in picked):
                continue
            picked.append(p)

    return [p.segment_id for p in picked[:K]]


def design_for_segment(g: pd.DataFrame, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build stacked model using initial-condition normalization:
      y_drop(t) = SOC0 - SOC(t) = theta' * C(t)
    where C(t) are cumulative integrals of basis functions.

    Returns: X (n x p), y_drop_pct (n,), soc0_pct (scalar repeated), feature_names
    """
    g = g.copy()
    g = g.dropna(subset=["battery_level_pct"])

    y_pct = g["battery_level_pct"].to_numpy(dtype=float)
    soc0 = float(y_pct[0])
    y_drop = soc0 - y_pct  # starts at 0

    dt_min = (g.index.to_series().diff().dt.total_seconds().fillna(60.0) / 60.0).to_numpy()
    dt_min = np.clip(dt_min, 1.0, 2.0)

    t_min = ((g.index - g.index.min()).total_seconds().astype(float) / 60.0).to_numpy()

    # Level 0 would be just C_time = t
    X = [t_min]
    names = ["C_time"]

    if level >= 1:
        scr = g["screen_on"].fillna(False).astype(int).to_numpy() if "screen_on" in g.columns else np.zeros(len(g))
        net = g["network_type"].fillna("none").astype(str).to_numpy() if "network_type" in g.columns else np.array(["none"] * len(g))
        net_wifi = (net == "wi-fi").astype(int)
        net_mob = (net == "mobile").astype(int)
        T0 = (g["battery_temp_C"] - TREF_C).fillna(0.0).to_numpy() if "battery_temp_C" in g.columns else np.zeros(len(g))

        C_scr = np.cumsum(scr * dt_min)
        C_wifi = np.cumsum(net_wifi * dt_min)
        C_mob = np.cumsum(net_mob * dt_min)
        C_T0 = np.cumsum(T0 * dt_min)

        X += [C_scr, C_wifi, C_mob, C_T0]
        names += ["C_scr", "C_wifi", "C_mob", "C_T0"]

    if level >= 2:
        scr = g["screen_on"].fillna(False).astype(int).to_numpy() if "screen_on" in g.columns else np.zeros(len(g))
        net = g["network_type"].fillna("none").astype(str).to_numpy() if "network_type" in g.columns else np.array(["none"] * len(g))
        net_wifi = (net == "wi-fi").astype(int)
        net_mob = (net == "mobile").astype(int)
        T0 = (g["battery_temp_C"] - TREF_C).fillna(0.0).to_numpy() if "battery_temp_C" in g.columns else np.zeros(len(g))

        C_scr_wifi = np.cumsum((scr * net_wifi) * dt_min)
        C_scr_mob = np.cumsum((scr * net_mob) * dt_min)
        C_scr_T0 = np.cumsum((scr * T0) * dt_min)
        C_wifi_T0 = np.cumsum((net_wifi * T0) * dt_min)
        C_mob_T0 = np.cumsum((net_mob * T0) * dt_min)

        X += [C_scr_wifi, C_scr_mob, C_scr_T0, C_wifi_T0, C_mob_T0]
        names += ["C_scr_wifi", "C_scr_mob", "C_scr_T0", "C_wifi_T0", "C_mob_T0"]

    Xmat = np.column_stack(X)
    return Xmat, y_drop, np.full(len(g), soc0), names


def fit_shared_params(segments: list[pd.DataFrame], level: int) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, list[str]]:
    """
    Fit one theta shared across all segments:
      y_drop = X theta
    Compute R² on SOC% across all points:
      y = SOC%, yhat = SOC0 - X theta
    """
    Xs, ydrops, soc0s = [], [], []
    names = None
    y_all = []
    for g in segments:
        X, y_drop, soc0, nm = design_for_segment(g, level=level)
        if names is None:
            names = nm
        Xs.append(X)
        ydrops.append(y_drop)
        soc0s.append(soc0)
        y_all.append(g["battery_level_pct"].to_numpy(dtype=float))

    X_all = np.vstack(Xs)
    y_drop_all = np.concatenate(ydrops)
    soc0_all = np.concatenate(soc0s)
    y_all = np.concatenate(y_all)

    theta, *_ = np.linalg.lstsq(X_all, y_drop_all, rcond=None)
    yhat = soc0_all - (X_all @ theta)

    resid = y_all - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_all - float(np.mean(y_all))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y_all))) if len(y_all) else np.nan

    assert names is not None
    return theta, r2, rmse, y_all, yhat, names


def concat_time_axis(segments: list[pd.DataFrame]) -> tuple[np.ndarray, list[int]]:
    """Return concatenated x (minutes) and segment boundary indices (start positions)."""
    xs = []
    boundaries = []
    cursor = 0.0
    for g in segments:
        n = len(g)
        boundaries.append(len(xs))
        t = (g.index - g.index.min()).total_seconds().astype(float) / 60.0
        xs.append(cursor + t)
        cursor += float(t.max()) + GAP_BETWEEN_SEGS_MIN if n else cursor + GAP_BETWEEN_SEGS_MIN
    x = np.concatenate(xs) if xs else np.array([])
    return x, boundaries


def plot_device(device_id: str, seg_ids: list[int], segments: list[pd.DataFrame], theta: np.ndarray, r2: float, rmse: float, level: int) -> Path:
    # rebuild yhat in segment order for plotting
    y_all = []
    yhat_all = []
    # also build state overlays
    scr_all = []
    net_all = []
    temp_all = []
    for g in segments:
        X, y_drop, soc0, _ = design_for_segment(g, level=level)
        y_all.append(g["battery_level_pct"].to_numpy(dtype=float))
        yhat_all.append(soc0 - (X @ theta))
        scr_all.append(g["screen_on"].fillna(False).astype(int).to_numpy() if "screen_on" in g.columns else np.zeros(len(g)))
        net_all.append(g["network_type"].fillna("none").astype(str).to_numpy() if "network_type" in g.columns else np.array(["none"] * len(g)))
        temp_all.append(g["battery_temp_C"].to_numpy(dtype=float) if "battery_temp_C" in g.columns else np.full(len(g), np.nan))

    y = np.concatenate(y_all)
    yhat = np.concatenate(yhat_all)
    scr = np.concatenate(scr_all)
    net = np.concatenate(net_all)
    temp = np.concatenate(temp_all)

    x, boundaries = concat_time_axis(segments)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(x, y, lw=1.2, label="SOC(%) observed (4 segments)")
    ax.plot(x, yhat, lw=2.0, label=f"shared-params fit (level{level}), R2={r2:.3f}, RMSE={rmse:.3f}")
    ax.set_title(f"Device {device_id}: 4 discharge segments, shared parameters, segs={seg_ids}")
    ax.set_ylabel("SOC(%)")
    ax.set_xlabel("Concatenated time (min)")
    ax.grid(True, alpha=0.3)

    # vertical separators
    for b in boundaries[1:]:
        ax.axvline(x[b], color="k", alpha=0.12, lw=1)

    # overlays: screen on (blue), mobile (orange), wifi (green-ish)
    y0, y1 = ax.get_ylim()
    ax.fill_between(x, y0, y1, where=(scr == 1), color="tab:blue", alpha=0.05, step="pre")
    ax.fill_between(x, y0, y1, where=(net == "mobile"), color="tab:orange", alpha=0.04, step="pre")
    ax.fill_between(x, y0, y1, where=(net == "wi-fi"), color="tab:green", alpha=0.03, step="pre")
    ax.set_ylim(y0, y1)

    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / f"{device_id}_4seg_shared_level{level}_r2_{r2:.3f}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    meta = pd.read_csv(IN_DIR / "device_meta_summary.csv")
    devices = meta["device_id"].astype(str).tolist()

    rows = []
    for device_id in devices:
        df = load_device_discharge(device_id)
        profs = build_profiles(df, min_len_min=MIN_SEG_LEN_MIN, min_drop_pct=MIN_SEG_DROP_PCT)
        if len(profs) < K:
            profs = build_profiles(df, min_len_min=RELAX_MIN_SEG_LEN_MIN, min_drop_pct=RELAX_MIN_SEG_DROP_PCT)
        seg_ids = pick_diverse_four(profs)
        if len(seg_ids) < 2:
            continue

        segments = []
        for sid in seg_ids:
            g = df[df["segment_id"] == sid].copy()
            g = g.dropna(subset=["battery_level_pct"])
            # keep only valid nets
            if "network_type" in g.columns:
                g = g[g["network_type"].fillna("none").isin(NETS)].copy()
            if len(g) < 10:
                continue
            segments.append(g)

        if len(segments) < 2:
            continue

        best = None
        best_level = None
        best_theta = None
        best_rmse = None
        for level in LEVELS:
            theta, r2, rmse, _, _, _ = fit_shared_params(segments, level=level)
            if best is None or (np.isfinite(r2) and r2 > best):
                best = r2
                best_level = level
                best_theta = theta
                best_rmse = rmse
            if np.isfinite(r2) and r2 >= TARGET_R2:
                break

        assert best is not None and best_level is not None and best_theta is not None and best_rmse is not None
        out_fig = plot_device(device_id, seg_ids, segments, best_theta, best, best_rmse, best_level)
        print("wrote", out_fig)

        rows.append(
            {
                "device_id": device_id,
                "segment_ids": ";".join(map(str, seg_ids)),
                "n_segments": len(segments),
                "level": int(best_level),
                "R2_soc": float(best),
                "RMSE_soc_pct": float(best_rmse),
                "plot": str(out_fig.as_posix()),
            }
        )

    out_csv = IN_DIR / "four_segment_shared_fit_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()

