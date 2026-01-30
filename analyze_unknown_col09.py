import pandas as pd
from pathlib import Path

IN_DIR = Path('processed')
files = sorted(IN_DIR.glob('*_clean.csv'))

rows = []

for f in files:
    df = pd.read_csv(f)
    device = str(df['device_id'].iloc[0])

    c9 = pd.to_numeric(df['col09_unknown'], errors='coerce')
    cur = pd.to_numeric(df['battery_current_mA'], errors='coerce')
    tmp = pd.to_numeric(df['battery_temp_C'], errors='coerce')
    lvl = pd.to_numeric(df['battery_level_pct'], errors='coerce')

    scr = df['screen_on'].astype('boolean')
    chg = df['is_charging'].astype('boolean')
    net = df['network_type'].fillna('NA')

    # discharge-only subset
    mask = (chg == False)

    c9_d = c9[mask]
    cur_d = cur[mask]
    tmp_d = tmp[mask]
    lvl_d = lvl[mask]
    scr_d = scr[mask]
    net_d = net[mask]

    top = c9_d.value_counts(dropna=True).head(10)
    top_dict = {str(k): int(v) for k, v in top.items()}

    mean_by_net = pd.DataFrame({'c9': c9_d, 'net': net_d}).groupby('net')['c9'].mean().to_dict()

    rows.append({
        'device_id': device,
        'rows': int(len(df)),
        'discharge_rows': int(mask.sum()),
        'c9_min': float(c9_d.min()) if c9_d.notna().any() else None,
        'c9_max': float(c9_d.max()) if c9_d.notna().any() else None,
        'c9_unique': int(c9_d.nunique(dropna=True)),
        'c9_top10': top_dict,
        'corr_c9_current': float(c9_d.corr(cur_d)),
        'corr_c9_temp': float(c9_d.corr(tmp_d)),
        'corr_c9_level': float(c9_d.corr(lvl_d)),
        'mean_c9_screen_on': float(c9_d[scr_d == True].mean()) if (scr_d == True).any() else None,
        'mean_c9_screen_off': float(c9_d[scr_d == False].mean()) if (scr_d == False).any() else None,
        'mean_c9_by_net': mean_by_net,
    })

out = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', 200)
print(out[['device_id','rows','discharge_rows','c9_min','c9_max','c9_unique','corr_c9_current','corr_c9_temp','corr_c9_level','mean_c9_screen_on','mean_c9_screen_off']].to_string(index=False))

print('\n--- c9 top10 values (discharge-only) ---')
for r in rows:
    print(r['device_id'], r['c9_top10'])

print('\n--- mean c9 by network (discharge-only) ---')
for r in rows:
    m = r['mean_c9_by_net']
    keys = sorted(m.keys())
    print(r['device_id'], {k: m[k] for k in keys})
