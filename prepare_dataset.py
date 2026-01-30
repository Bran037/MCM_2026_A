import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path('Database')
OUT_DIR = Path('processed')
OUT_DIR.mkdir(exist_ok=True)

# Inferred schema (18 columns; last 3 are empty in all checked files)
COLS = [
    'device_id',
    'device_model',
    'android_version',
    'battery_chemistry',
    'battery_capacity_mAh',
    'timestamp_ms',
    'is_charging',
    'battery_level_pct',
    'logger_package',
    'col09_unknown',
    'battery_temp_C',
    'battery_voltage_mV',
    'battery_current_mA',
    'network_type',
    'screen_on',
    'unused_15',
    'unused_16',
    'unused_17',
]

DEVICE_FILES = sorted([p for p in RAW_DIR.glob('*.csv') if p.stem.isdigit()])


def to_bool(series: pd.Series) -> pd.Series:
    s = series.astype('string').str.strip().str.lower()
    return s.map({'true': True, 'false': False}).astype('boolean')


def clean_one(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, header=None, names=COLS, dtype='string')

    # Normalize empties
    df = df.replace({'': pd.NA})

    # Drop always-empty columns
    df = df.drop(columns=['unused_15', 'unused_16', 'unused_17'], errors='ignore')

    # Types
    df['device_id'] = pd.to_numeric(df['device_id'], errors='coerce').astype('Int64')
    df['battery_capacity_mAh'] = pd.to_numeric(df['battery_capacity_mAh'], errors='coerce')
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df['battery_level_pct'] = pd.to_numeric(df['battery_level_pct'], errors='coerce')
    df['col09_unknown'] = pd.to_numeric(df['col09_unknown'], errors='coerce')
    df['battery_temp_C'] = pd.to_numeric(df['battery_temp_C'], errors='coerce')
    df['battery_voltage_mV'] = pd.to_numeric(df['battery_voltage_mV'], errors='coerce')
    df['battery_current_mA'] = pd.to_numeric(df['battery_current_mA'], errors='coerce')

    df['is_charging'] = to_bool(df['is_charging'])
    df['screen_on'] = to_bool(df['screen_on'])

    # Datetime (keep both)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce', utc=True)

    # Sort + dedupe by timestamp_ms
    df = df.sort_values(['timestamp_ms'], kind='mergesort')
    before = len(df)
    df = df.drop_duplicates(subset=['timestamp_ms'], keep='last')
    after = len(df)

    # Sampling interval stats
    ts = df['timestamp'].dropna().sort_values()
    dts = ts.diff().dropna().dt.total_seconds()

    def _vc_top(series: pd.Series, n: int = 10) -> dict:
        vc = series.value_counts(dropna=False).head(n)
        # JSON requires keys to be basic python types; normalize to strings
        return {str(k): int(v) for k, v in vc.items()}

    meta = {
        'file': str(path.name),
        'rows_raw': int(before),
        'rows_dedup': int(after),
        'dedup_removed': int(before - after),
        'time_start_utc': str(ts.min()) if len(ts) else None,
        'time_end_utc': str(ts.max()) if len(ts) else None,
        'dt_median_s': float(np.nanmedian(dts)) if len(dts) else None,
        'dt_p05_s': float(np.nanpercentile(dts, 5)) if len(dts) else None,
        'dt_p95_s': float(np.nanpercentile(dts, 95)) if len(dts) else None,
        'null_counts': df.isna().sum().to_dict(),
        'ranges': {
            'battery_level_pct': [float(df['battery_level_pct'].min()), float(df['battery_level_pct'].max())],
            'battery_temp_C': [float(df['battery_temp_C'].min()), float(df['battery_temp_C'].max())],
            'battery_voltage_mV': [float(df['battery_voltage_mV'].min()), float(df['battery_voltage_mV'].max())],
            'battery_current_mA': [float(df['battery_current_mA'].min()), float(df['battery_current_mA'].max())],
            'col09_unknown': [float(df['col09_unknown'].min()), float(df['col09_unknown'].max())],
        },
        'value_counts': {
            'network_type_top': _vc_top(df['network_type'], 10),
            'screen_on': _vc_top(df['screen_on'], 10),
            'is_charging': _vc_top(df['is_charging'], 10),
            'logger_package': _vc_top(df['logger_package'], 10),
        }
    }

    return df, meta


all_meta = []
for p in DEVICE_FILES:
    df, meta = clean_one(p)
    out_csv = OUT_DIR / f"{p.stem}_clean.csv"
    # Keep a consistent column order
    col_order = [
        'device_id','device_model','android_version','battery_chemistry','battery_capacity_mAh',
        'timestamp_ms','timestamp','is_charging','screen_on','network_type','logger_package','col09_unknown',
        'battery_level_pct','battery_temp_C','battery_voltage_mV','battery_current_mA'
    ]
    df = df[col_order]
    df.to_csv(out_csv, index=False, encoding='utf-8')
    all_meta.append(meta)
    print('wrote', out_csv, 'rows', len(df))

# Write summary
summary_path = OUT_DIR / 'cleaning_summary.json'
summary_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', summary_path)

# Also write a compact table
rows = []
for m in all_meta:
    rows.append({
        'file': m['file'],
        'rows_raw': m['rows_raw'],
        'rows_dedup': m['rows_dedup'],
        'dedup_removed': m['dedup_removed'],
        'time_start_utc': m['time_start_utc'],
        'time_end_utc': m['time_end_utc'],
        'dt_median_s': m['dt_median_s'],
        'level_min': m['ranges']['battery_level_pct'][0],
        'level_max': m['ranges']['battery_level_pct'][1],
        'temp_min': m['ranges']['battery_temp_C'][0],
        'temp_max': m['ranges']['battery_temp_C'][1],
        'volt_min': m['ranges']['battery_voltage_mV'][0],
        'volt_max': m['ranges']['battery_voltage_mV'][1],
        'curr_min': m['ranges']['battery_current_mA'][0],
        'curr_max': m['ranges']['battery_current_mA'][1],
        'col09_min': m['ranges']['col09_unknown'][0],
        'col09_max': m['ranges']['col09_unknown'][1],
    })

df_sum = pd.DataFrame(rows)
df_sum.to_csv(OUT_DIR / 'cleaning_summary.csv', index=False, encoding='utf-8')
print('wrote', OUT_DIR / 'cleaning_summary.csv')
