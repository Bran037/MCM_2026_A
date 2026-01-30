import pandas as pd
from collections import Counter

files = [
    "352944080639365.csv",
    "358057080902248.csv",
    "866558034754119.csv",
    "866817034014258.csv",
    "867194030192184.csv",
]

def summarize_file(path: str, sample_rows: int = 2000, chunk_rows: int = 200000):
    sample50 = pd.read_csv(path, header=None, nrows=50)
    ncol = sample50.shape[1]

    # 缁忛獙浣嶆锛?-based锛?    ts_col = 5          # epoch ms
    charging_col = 6    # true/false
    batt_pct_col = 7    # 鐢甸噺鐧惧垎姣?    pkg_col = 8         # 鍖呭悕锛堝 com.zopper.batteryage锛?    temp_col = 10       # 娓╁害锛堟憚姘忓害锛?    volt_col = 11       # 鐢靛帇锛坢V锛?    curr_col = 12       # 鐢垫祦/鐢垫祦鐩稿叧锛堝崟浣嶅緟纭锛?    net_col = 13        # none/wi-fi/mobile
    screen_col = 14     # true/false锛堢枒浼?screen on锛?
    # 鍙栨牱鏈?    sample = pd.read_csv(path, header=None, nrows=sample_rows)

    # 鏃堕棿鎴充笌閲囨牱闂撮殧
    ts = pd.to_numeric(sample.iloc[:, ts_col], errors="coerce")
    null_ts_sample = int(ts.isna().sum())
    ts_sorted = ts.dropna().astype("int64").sort_values()
    deltas_ms = ts_sorted.diff().dropna().astype("int64")
    deltas_ms = deltas_ms[(deltas_ms >= 0) & (deltas_ms <= 60000)]
    delta_counts = Counter((deltas_ms // 1000).astype(int).tolist())
    top_deltas = delta_counts.most_common(5)

    # chunk 鎵弿鍏ㄦ枃浠舵椂闂磋寖鍥达紙閬垮厤涓€娆℃€ц鐖嗗唴瀛橈級
    ts_min = None
    ts_max = None
    total = 0
    null_ts_total = 0
    for chunk in pd.read_csv(path, header=None, chunksize=chunk_rows):
        total += len(chunk)
        t = pd.to_numeric(chunk.iloc[:, ts_col], errors="coerce")
        null_ts_total += int(t.isna().sum())
        tmin = t.min()
        tmax = t.max()
        if pd.notna(tmin):
            ts_min = tmin if ts_min is None else min(ts_min, tmin)
        if pd.notna(tmax):
            ts_max = tmax if ts_max is None else max(ts_max, tmax)

    def rng(idx: int):
        x = pd.to_numeric(sample.iloc[:, idx], errors="coerce")
        return (float(x.min()), float(x.max()))

    out = {
        "file": path,
        "rows": total,
        "cols": ncol,
        "start": pd.to_datetime(ts_min, unit="ms") if ts_min is not None else None,
        "end": pd.to_datetime(ts_max, unit="ms") if ts_max is not None else None,
        "null_ts_sample": null_ts_sample,
        "null_ts_total": null_ts_total,
        "battery_pct_range": rng(batt_pct_col),
        "temp_range": rng(temp_col),
        "volt_range": rng(volt_col),
        "curr_like_range": rng(curr_col),
        "charging_counts": sample.iloc[:, charging_col].astype(str).value_counts().head(5).to_dict(),
        "pkg_top": sample.iloc[:, pkg_col].astype(str).value_counts().head(5).to_dict(),
        "net_types_top": sample.iloc[:, net_col].astype(str).value_counts().head(5).to_dict(),
        "screen_like_top": sample.iloc[:, screen_col].astype(str).value_counts().head(5).to_dict(),
        "top_delta_seconds_le_60": top_deltas,
    }
    return out

for f in files:
    s = summarize_file(f)
    print("\n==", s["file"], "==")
    print("rows:", s["rows"], "cols:", s["cols"])
    print("time:", s["start"], "->", s["end"], "null_ts(total):", s["null_ts_total"], ")")
    print("battery% range:", s["battery_pct_range"], "temp range:", s["temp_range"], "volt range:", s["volt_range"], "curr-like range:", s["curr_like_range"])
    print("charging (sample):", s["charging_counts"])
    print("net (sample):", s["net_types_top"], "screen-like (sample):", s["screen_like_top"])
    print("top delta seconds (<=60s):", s["top_delta_seconds_le_60"])
