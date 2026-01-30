from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse segment selection + feature construction from the shared-fit script
import fit_four_segments_shared_params as shared


IN_DIR = Path("processed") / "discharge"
FIG_DIR = Path("figures") / "diagnostics_soc_cv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FOUR = IN_DIR / "four_segment_shared_fit_summary.csv"

GAP_BETWEEN_SEGS_MIN = 30.0
LEVEL = 1  # start from level1 (screen/net/temp cumulative bases); can upgrade later if needed
NETS = ["none", "wi-fi", "mobile"]


@dataclass(frozen=True)
class SplitResult:
    device_id: str
    split: str
    seg_ids_ordered: str
    train_seg_ids: str
    test_seg_ids: str
    n_train: int
    n_test: int
    r2_test: float
    rmse_test: float
    level: int
    plot: str


def _fit_theta(train_segments: list[pd.DataFrame], level: int) -> np.ndarray:
    Xs = []
    ys = []
    for g in train_segments:
        X, y_drop, _, _ = shared.design_for_segment(g, level=level)
        Xs.append(X)
        ys.append(y_drop)
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    theta, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    return theta


def _predict_segment(g: pd.DataFrame, theta: np.ndarray, level: int) -> np.ndarray:
    X, _, soc0, _ = shared.design_for_segment(g, level=level)
    return soc0 - (X @ theta)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else np.nan


def _concat_time_axis(segments: list[pd.DataFrame]) -> tuple[np.ndarray, list[int]]:
    xs = []
    boundaries = []
    cursor = 0.0
    for g in segments:
        boundaries.append(len(xs))
        t = (g.index - g.index.min()).total_seconds().astype(float) / 60.0
        xs.append(cursor + t)
        cursor += float(t.max()) + GAP_BETWEEN_SEGS_MIN if len(t) else cursor + GAP_BETWEEN_SEGS_MIN
    x = np.concatenate(xs) if xs else np.array([])
    return x, boundaries


def plot_cv(
    device_id: str,
    seg_ids: list[int],
    segments: list[pd.DataFrame],
    train_set: set[int],
    test_set: set[int],
    theta: np.ndarray,
    level: int,
    r2_test: float,
    rmse_test: float,
    split_name: str,
) -> Path:
    # build series in concatenated order
    y_all, yhat_all, scr_all, net_all = [], [], [], []
    seg_tag = []
    for sid, g in zip(seg_ids, segments):
        y = g["battery_level_pct"].to_numpy(dtype=float)
        yhat = _predict_segment(g, theta, level=level)
        y_all.append(y)
        yhat_all.append(yhat)
        scr_all.append(g["screen_on"].fillna(False).astype(int).to_numpy() if "screen_on" in g.columns else np.zeros(len(g)))
        net_all.append(g["network_type"].fillna("none").astype(str).to_numpy() if "network_type" in g.columns else np.array(["none"] * len(g)))
        seg_tag.append(np.full(len(g), sid))

    y_all = np.concatenate(y_all)
    yhat_all = np.concatenate(yhat_all)
    scr = np.concatenate(scr_all)
    net = np.concatenate(net_all)
    seg_tag = np.concatenate(seg_tag)

    x, boundaries = _concat_time_axis(segments)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x, y_all, lw=1.2, label="SOC(%) observed (4 segments)")
    ax.plot(x, yhat_all, lw=2.0, label=f"prediction (shared params), level{level}")
    ax.set_title(
        f"{device_id} | {split_name} | test R2={r2_test:.3f}, RMSE={rmse_test:.3f} | segs={seg_ids}"
    )
    ax.set_xlabel("Concatenated time (min)")
    ax.set_ylabel("SOC(%)")
    ax.grid(True, alpha=0.3)

    # separators + train/test shading by segment
    y0, y1 = ax.get_ylim()
    for i, sid in enumerate(seg_ids):
        # segment span in x
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(x)
        x0 = float(x[start_idx])
        x1 = float(x[end_idx - 1]) if end_idx - 1 >= start_idx else x0
        if sid in test_set:
            ax.axvspan(x0, x1, color="tab:red", alpha=0.05)
        else:
            ax.axvspan(x0, x1, color="tab:gray", alpha=0.03)
        if i > 0:
            ax.axvline(x0, color="k", alpha=0.12, lw=1)

    # overlays: screen on / network states
    ax.fill_between(x, y0, y1, where=(scr == 1), color="tab:blue", alpha=0.04, step="pre")
    ax.fill_between(x, y0, y1, where=(net == "mobile"), color="tab:orange", alpha=0.03, step="pre")
    ax.fill_between(x, y0, y1, where=(net == "wi-fi"), color="tab:green", alpha=0.025, step="pre")
    ax.set_ylim(y0, y1)

    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / f"{device_id}_{split_name}_level{level}_r2_{r2_test:.3f}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    four = pd.read_csv(FOUR)
    meta = pd.read_csv(IN_DIR / "device_meta_summary.csv")
    devices = meta["device_id"].astype(str).tolist()

    out_rows: list[dict] = []

    for device_id in devices:
        row = four[four["device_id"].astype(str) == str(device_id)]
        if row.empty:
            continue
        seg_ids = [int(s) for s in str(row["segment_ids"].iloc[0]).split(";") if s.strip() != ""]

        df = shared.load_device_discharge(str(device_id))
        # order segments by actual start time
        seg_start = df[df["segment_id"].isin(seg_ids)].groupby("segment_id").apply(lambda g: g.index.min())
        seg_ordered = seg_start.sort_values().index.astype(int).tolist()
        if len(seg_ordered) < 4:
            continue

        # build segment frames in that order
        seg_frames = []
        for sid in seg_ordered[:4]:
            g = df[df["segment_id"] == sid].copy()
            g = g.dropna(subset=["battery_level_pct"])
            if "network_type" in g.columns:
                g = g[g["network_type"].fillna("none").isin(NETS)].copy()
            if len(g) < 10:
                continue
            seg_frames.append(g)

        if len(seg_frames) < 4:
            continue

        # split A: train 3 test 1
        train3 = seg_frames[:3]
        test1 = seg_frames[3:]
        train_set_31 = set(seg_ordered[:3])
        test_set_31 = {seg_ordered[3]}
        theta31 = _fit_theta(train3, level=LEVEL)

        y_test = np.concatenate([g["battery_level_pct"].to_numpy(dtype=float) for g in test1])
        yhat_test = np.concatenate([_predict_segment(g, theta31, level=LEVEL) for g in test1])
        r2_test = _r2(y_test, yhat_test)
        rmse_test = _rmse(y_test, yhat_test)
        plot_path = plot_cv(
            str(device_id),
            seg_ordered[:4],
            seg_frames,
            train_set_31,
            test_set_31,
            theta31,
            LEVEL,
            r2_test,
            rmse_test,
            "soc_train3_test1",
        )
        out_rows.append(
            dict(
                device_id=str(device_id),
                split="soc_train3_test1",
                seg_ids_ordered=";".join(map(str, seg_ordered[:4])),
                train_seg_ids=";".join(map(str, seg_ordered[:3])),
                test_seg_ids=str(seg_ordered[3]),
                n_train=int(sum(len(g) for g in train3)),
                n_test=int(sum(len(g) for g in test1)),
                r2_test=float(r2_test),
                rmse_test=float(rmse_test),
                level=int(LEVEL),
                plot=str(plot_path.as_posix()),
            )
        )
        print("wrote", plot_path)

        # split B: train 2 test 2
        train2 = seg_frames[:2]
        test2 = seg_frames[2:]
        train_set_22 = set(seg_ordered[:2])
        test_set_22 = set(seg_ordered[2:4])
        theta22 = _fit_theta(train2, level=LEVEL)

        y_test = np.concatenate([g["battery_level_pct"].to_numpy(dtype=float) for g in test2])
        yhat_test = np.concatenate([_predict_segment(g, theta22, level=LEVEL) for g in test2])
        r2_test = _r2(y_test, yhat_test)
        rmse_test = _rmse(y_test, yhat_test)
        plot_path = plot_cv(
            str(device_id),
            seg_ordered[:4],
            seg_frames,
            train_set_22,
            test_set_22,
            theta22,
            LEVEL,
            r2_test,
            rmse_test,
            "soc_train2_test2",
        )
        out_rows.append(
            dict(
                device_id=str(device_id),
                split="soc_train2_test2",
                seg_ids_ordered=";".join(map(str, seg_ordered[:4])),
                train_seg_ids=";".join(map(str, seg_ordered[:2])),
                test_seg_ids=";".join(map(str, seg_ordered[2:4])),
                n_train=int(sum(len(g) for g in train2)),
                n_test=int(sum(len(g) for g in test2)),
                r2_test=float(r2_test),
                rmse_test=float(rmse_test),
                level=int(LEVEL),
                plot=str(plot_path.as_posix()),
            )
        )
        print("wrote", plot_path)

    out = pd.DataFrame(out_rows)
    out.to_csv(IN_DIR / "cv_soc_four_segment_holdout.csv", index=False, encoding="utf-8")
    print("wrote", IN_DIR / "cv_soc_four_segment_holdout.csv")


if __name__ == "__main__":
    main()

