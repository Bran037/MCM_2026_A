# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = Path('processed')
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)

clean_files = sorted(IN_DIR.glob('*_clean.csv'))


def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce')
    d = d.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    num_cols = ['battery_level_pct','battery_temp_C','battery_voltage_mV','battery_current_mA','col09_unknown']
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    d['is_charging'] = d['is_charging'].astype('boolean')
    d['screen_on'] = d['screen_on'].astype('boolean')

    agg = {
        'battery_level_pct': 'median',
        'battery_temp_C': 'median',
        'battery_voltage_mV': 'median',
        'battery_current_mA': 'median',
        'col09_unknown': 'median',
        'is_charging': 'max',
        'screen_on': 'max',
    }

    out = d.resample('1min').agg(agg)
    return out


def plot_device(df1m: pd.DataFrame, device_id: str) -> Path:
    t = df1m.index
    soc = df1m['battery_level_pct']
    temp = df1m['battery_temp_C']
    volt = df1m['battery_voltage_mV']
    curr = df1m['battery_current_mA']
    chg = df1m['is_charging'].fillna(False).astype(bool)
    scr = df1m['screen_on'].fillna(False).astype(bool)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, soc, lw=1)
    axes[0].set_ylabel('SOC(%)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, curr, lw=1)
    axes[1].set_ylabel('Current(mA)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, volt, lw=1)
    axes[2].set_ylabel('Voltage(mV)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, temp, lw=1)
    axes[3].set_ylabel('Temp(C)')
    axes[3].grid(True, alpha=0.3)

    # Shade charging/screen-on periods
    for ax in axes:
        y0, y1 = ax.get_ylim()
        ax.fill_between(t, y0, y1, where=chg.values, color='tab:green', alpha=0.08, step='pre')
        ax.fill_between(t, y0, y1, where=scr.values, color='tab:blue', alpha=0.05, step='pre')
        ax.set_ylim(y0, y1)

    fig.suptitle(f'Device {device_id}: 1-min resampled signals (green=charging, blue=screen-on)')
    axes[-1].set_xlabel('Time (UTC)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = OUT_DIR / f'{device_id}_overview.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# Per-device plots
for f in clean_files:
    df = pd.read_csv(f)
    device_id = str(df['device_id'].iloc[0]) if 'device_id' in df.columns else f.stem.replace('_clean','')
    df1m = resample_1min(df)
    out = plot_device(df1m, device_id)
    print('wrote', out)

# Combined SOC comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
for f in clean_files:
    df = pd.read_csv(f)
    device_id = str(df['device_id'].iloc[0])
    df1m = resample_1min(df)
    soc = df1m['battery_level_pct']
    ax.plot(soc.index, soc.values, lw=1, label=device_id)

ax.set_title('SOC(%) comparison across devices (1-min resampled, UTC)')
ax.set_ylabel('SOC(%)')
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
out = OUT_DIR / 'soc_comparison.png'
fig.savefig(out, dpi=160)
plt.close(fig)
print('wrote', out)
