import pandas as pd

def analyze(f, n=200000):
    df = pd.read_csv(f, header=None, names=list(range(18)), dtype="string", nrows=n)
    c9 = pd.to_numeric(df[9], errors="coerce")
    cur = pd.to_numeric(df[12], errors="coerce")
    tmp = pd.to_numeric(df[10], errors="coerce")
    lvl = pd.to_numeric(df[7], errors="coerce")
    scr = df[14].str.lower().map({"true": 1, "false": 0})
    net = df[13].fillna("NA")

    print("\n==", f, "==")
    print("c9 range:", float(c9.min()), float(c9.max()), "non-na ratio:", float(c9.notna().mean()))
    print("corr(c9,current):", float(c9.corr(cur)))
    print("corr(c9,temp):", float(c9.corr(tmp)))
    print("corr(c9,level):", float(c9.corr(lvl)))
    if scr.notna().mean() > 0.9:
        print("c9 mean screen_on=1:", float(c9[scr == 1].mean()), "screen_on=0:", float(c9[scr == 0].mean()))

    means = pd.DataFrame({"c9": c9, "net": net}).groupby("net")["c9"].mean().sort_values(ascending=False)
    print("c9 mean by net:", means.to_dict())

for f in ["358057080902248.csv", "866558034754119.csv"]:
    analyze(f)
