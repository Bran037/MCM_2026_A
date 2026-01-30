import pandas as pd
from pathlib import Path

files = [
    "352944080639365.csv",
    "358057080902248.csv",
    "866558034754119.csv",
    "866817034014258.csv",
    "867194030192184.csv",
]

RAW = list(range(18))


def summarize_col(s: pd.Series):
    # normalize empty strings
    s = s.replace({"": pd.NA})
    out = {
        "non_empty": int(s.notna().sum()),
        "empty": int(s.isna().sum()),
    }

    # Force string for categorical checks
    # (avoid pandas auto bool/int inference)
    s_str = s.astype("string")

    # numeric attempt
    sn = pd.to_numeric(s_str, errors="coerce").astype("float64")
    num_ratio = float(sn.notna().mean())
    out["num_ratio"] = round(num_ratio, 4)

    if num_ratio >= 0.95:
        sn2 = sn.dropna()
        out.update({
            "min": float(sn2.min()),
            "p05": float(sn2.quantile(0.05)),
            "median": float(sn2.median()),
            "p95": float(sn2.quantile(0.95)),
            "max": float(sn2.max()),
        })
    else:
        vc = s_str.value_counts(dropna=True)
        out["unique"] = int(vc.shape[0])
        out["top_values"] = vc.head(10).to_dict()

    return out


for f in files:
    p = Path(f)
    # sample for speed (enough to infer types/patterns)
    df = pd.read_csv(p, header=None, names=RAW, engine="c", dtype="string", nrows=200000)

    print("\n===", f, "(sample 200k)===")
    print("sample_rows:", len(df))

    # Column summaries
    for c in RAW:
        info = summarize_col(df[c])
        print(f"col{c:02d}:", info)

    # Timestamp plausibility (col5)
    ts = pd.to_numeric(df[5], errors="coerce")
    dt = pd.to_datetime(ts, unit="ms", errors="coerce")
    print("timestamp(col05) range:", dt.min(), "->", dt.max())

    # Quick relationship checks
    lvl = pd.to_numeric(df[7], errors="coerce")
    v = pd.to_numeric(df[11], errors="coerce")
    i = pd.to_numeric(df[12], errors="coerce")
    tmp = pd.to_numeric(df[10], errors="coerce")

    for name, x in [("voltage(col11)", v), ("current(col12)", i), ("temp(col10)", tmp)]:
        print(f"corr(level,col07) with {name}:", float(lvl.corr(x)))

    # col09 hypothesis: maybe brightness/usage intensity? check relation with screen_on
    c9 = pd.to_numeric(df[9], errors="coerce")
    scr = df[14].str.lower().map({"true": 1, "false": 0})
    if c9.notna().mean() > 0.9 and scr.notna().mean() > 0.9:
        print("col09 mean when screen_on=1:", float(c9[scr==1].mean()))
        print("col09 mean when screen_on=0:", float(c9[scr==0].mean()))

    # tail columns non-empty rate
    for c in [15,16,17]:
        ne = int(df[c].replace({"": pd.NA}).notna().sum())
        print(f"non-empty count col{c:02d}:", ne)
