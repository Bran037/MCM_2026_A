import pandas as pd

f = "user_behavior_dataset.csv"
df = pd.read_csv(f)

print("rows:", len(df), "cols:", len(df.columns))
print("columns:", list(df.columns))
print("nulls:\n", df.isna().sum().to_string())

num = df.select_dtypes(include="number")
print("\nNumeric describe:\n", num.describe().T[["count","mean","std","min","25%","50%","75%","max"]].to_string())

for c in ["Device Model","Operating System","Gender","User Behavior Class"]:
    if c in df.columns:
        print(f"\n{c} value_counts:\n", df[c].value_counts(dropna=False).to_string())
