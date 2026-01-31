from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


TEST1_DIR = Path("Database") / "test_1"
OUT_DIR = Path("processed") / "test1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "T4.csv",
    "T0.csv",
    "ScreenOn.csv",
    "UserPresent.csv",
    "Wifi.csv",
    "Bluetooth.csv",
    "AllBroadcasts.csv",
    "Calls.csv",
    "SMS.csv",
    "Moriarty.csv",
    "AppPackages.csv",
]


@dataclass
class Stat:
    file: str
    n_rows: int
    user_col: Optional[str]
    uuid_col: Optional[str]
    ts_col: Optional[str]
    uuid_min: Optional[int]
    uuid_max: Optional[int]
    uuid_unique: Optional[int]
    uuid_unique_sampled: Optional[int]
    ts_min: Optional[str]
    ts_max: Optional[str]


def _first_row_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _detect_cols(cols: list[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    lower = [c.strip().lower() for c in cols]
    user = None
    uuid = None
    ts = None
    # user id
    for cand in ["userid", "user_id", "user", "userId".lower()]:
        if cand in lower:
            user = cols[lower.index(cand)]
            break
    # uuid
    if "uuid" in lower:
        uuid = cols[lower.index("uuid")]
    # timestamp string column (if any)
    for cand in ["timestamp", "timestemp", "battery_timestamp", "traffic_timestamp", "googleplayloc_timestamp"]:
        if cand in lower:
            ts = cols[lower.index(cand)]
            break
    return user, uuid, ts


def _summarize_file(path: Path, max_rows_for_unique: int = 200_000) -> Stat:
    cols = _first_row_header(path)
    user_col, uuid_col, ts_col = _detect_cols(cols)

    # Count rows quickly (stream) + gather uuid min/max and sampled uniques
    n_rows = 0
    uuid_min = None
    uuid_max = None
    uuid_sample = set()
    ts_min = None
    ts_max = None

    usecols = []
    if user_col:
        usecols.append(user_col)
    if uuid_col:
        usecols.append(uuid_col)
    if ts_col:
        usecols.append(ts_col)

    # If no uuid, just count rows
    if not uuid_col:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            # skip header
            next(f)
            for _ in f:
                n_rows += 1
        return Stat(
            file=path.name,
            n_rows=n_rows,
            user_col=user_col,
            uuid_col=uuid_col,
            ts_col=ts_col,
            uuid_min=None,
            uuid_max=None,
            uuid_unique=None,
            uuid_unique_sampled=None,
            ts_min=None,
            ts_max=None,
        )

    # Chunked read for safety
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False):
        n_rows += len(chunk)
        u = pd.to_numeric(chunk[uuid_col], errors="coerce").dropna().astype("int64")
        if len(u):
            mn = int(u.min())
            mx = int(u.max())
            uuid_min = mn if uuid_min is None else min(uuid_min, mn)
            uuid_max = mx if uuid_max is None else max(uuid_max, mx)

            # sample uniques up to max_rows_for_unique rows scanned
            if len(uuid_sample) < 200_000 and n_rows <= max_rows_for_unique:
                uuid_sample.update(u.head(50_000).tolist())

        if ts_col and ts_col in chunk.columns:
            s = chunk[ts_col].astype("string")
            s = s.dropna()
            if len(s):
                smin = str(s.min())
                smax = str(s.max())
                ts_min = smin if ts_min is None else min(ts_min, smin)
                ts_max = smax if ts_max is None else max(ts_max, smax)

    # estimate uniques by full scan if file not too big
    uuid_unique = None
    if n_rows <= max_rows_for_unique:
        # re-scan only uuid col
        uniq = set()
        for chunk in pd.read_csv(path, usecols=[uuid_col], chunksize=200_000, low_memory=False):
            u = pd.to_numeric(chunk[uuid_col], errors="coerce").dropna().astype("int64")
            uniq.update(u.tolist())
        uuid_unique = len(uniq)

    return Stat(
        file=path.name,
        n_rows=n_rows,
        user_col=user_col,
        uuid_col=uuid_col,
        ts_col=ts_col,
        uuid_min=uuid_min,
        uuid_max=uuid_max,
        uuid_unique=uuid_unique,
        uuid_unique_sampled=len(uuid_sample) if uuid_sample else None,
        ts_min=ts_min,
        ts_max=ts_max,
    )


def _uuid_set_from_file(path: Path, uuid_col: str) -> set[int]:
    s: set[int] = set()
    for chunk in pd.read_csv(path, usecols=[uuid_col], chunksize=200_000, low_memory=False):
        u = pd.to_numeric(chunk[uuid_col], errors="coerce").dropna().astype("int64")
        s.update(u.tolist())
    return s


def main() -> None:
    rows: list[dict] = []
    stats: list[Stat] = []
    for fn in FILES:
        p = TEST1_DIR / fn
        if not p.exists():
            continue
        st = _summarize_file(p)
        stats.append(st)
        rows.append(st.__dict__)

    df = pd.DataFrame(rows).sort_values("file")
    df.to_csv(OUT_DIR / "test1_file_summary.csv", index=False, encoding="utf-8")

    # Alignment with T4 (as backbone, because it contains Battery_level)
    t4 = next((s for s in stats if s.file == "T4.csv"), None)
    if t4 and t4.uuid_col:
        t4_set = _uuid_set_from_file(TEST1_DIR / "T4.csv", t4.uuid_col)
        align_rows = []
        for st in stats:
            if st.file == "T4.csv" or not st.uuid_col:
                continue
            other_set = _uuid_set_from_file(TEST1_DIR / st.file, st.uuid_col)
            inter = len(t4_set & other_set)
            align_rows.append(
                {
                    "file": st.file,
                    "uuid_unique_other": len(other_set),
                    "uuid_unique_T4": len(t4_set),
                    "uuid_intersection_with_T4": inter,
                    "share_of_other_in_T4": inter / len(other_set) if other_set else None,
                    "share_of_T4_covered_by_other": inter / len(t4_set) if t4_set else None,
                }
            )
        pd.DataFrame(align_rows).sort_values("uuid_intersection_with_T4", ascending=False).to_csv(
            OUT_DIR / "test1_uuid_alignment_with_T4.csv", index=False, encoding="utf-8"
        )


if __name__ == "__main__":
    main()

