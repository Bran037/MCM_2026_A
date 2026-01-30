import numpy as np
import pandas as pd
from pathlib import Path

IN_DIR = Path('processed')
files = sorted(IN_DIR.glob('*_clean.csv'))

# Use 5-minute differencing on 1-min resampled SOC to reduce quantization noise

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce')
    d = d.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    for c in ['battery_level_pct','battery_temp_C','battery_voltage_mV','battery_current_mA']:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    d['is_charging'] = d['is_charging'].astype('boolean')
    d['screen_on'] = d['screen_on'].astype('boolean')

    agg = {
        'battery_level_pct': 'median',
        'battery_temp_C': 'median',
        'battery_voltage_mV': 'median',
        'battery_current_mA': 'median',
        'is_charging': 'max',
        'screen_on': 'max',
        'network_type': lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan,
    }

    return d.resample('1min').agg(agg)


def build_panel() -> pd.DataFrame:
    out = []
    for f in files:
        df = pd.read_csv(f)
        dev = str(df['device_id'].iloc[0])
        d1 = resample_1min(df)

        # discharge only
        d1 = d1[d1['is_charging'] == False].copy()

        # 5-min differencing
        d1['soc'] = d1['battery_level_pct'] / 100.0
        d1['soc_lag'] = d1['soc'].shift(5)
        d1['t_lag'] = d1.index.shift(5, freq='1min')

        dt_h = 5.0/60.0
        d1['dsoc_dt'] = (d1['soc'] - d1['soc_lag']) / dt_h  # per hour
        d1 = d1.dropna(subset=['dsoc_dt','battery_temp_C','screen_on','network_type'])

        # keep only discharging steps (SOC decreasing)
        d1 = d1[d1['dsoc_dt'] < 0].copy()
        d1['r'] = -d1['dsoc_dt']  # discharge rate (1/hour)
        d1['device_id'] = dev
        out.append(d1[['device_id','r','battery_temp_C','screen_on','network_type']])

    return pd.concat(out, ignore_index=True)


panel = build_panel()
print('panel rows:', len(panel))
print(panel[['r','battery_temp_C']].describe().to_string())

# Encode screen and network
panel['screen_on'] = panel['screen_on'].astype(bool)

# baseline categories
nets = ['none','wi-fi','mobile']
panel['network_type'] = panel['network_type'].astype('string').str.lower()
panel = panel[panel['network_type'].isin(nets)]

# Linear temperature model: r = a0 + a1*screen + a2_w*I(net=wifi)+a2_m*I(net=mobile) + aT*(T-Tref)
Tref = 30.0
panel['T0'] = panel['battery_temp_C'] - Tref
panel['scr'] = panel['screen_on'].astype(int)
panel['net_wifi'] = (panel['network_type'] == 'wi-fi').astype(int)
panel['net_mobile'] = (panel['network_type'] == 'mobile').astype(int)

X = np.column_stack([
    np.ones(len(panel)),
    panel['scr'].to_numpy(),
    panel['net_wifi'].to_numpy(),
    panel['net_mobile'].to_numpy(),
    panel['T0'].to_numpy(),
])
y = panel['r'].to_numpy()

# OLS
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
ss_res = float(np.sum((y - yhat)**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

print('\nOLS (linear T) coefficients:')
print('r = b0 + b_scr*scr + b_wifi*net_wifi + b_mob*net_mobile + b_T*(T-30)')
print(beta)
print('R2:', r2)

# Exponential temperature model (log-linear): log r = c0 + ... + cT*(T-Tref)
# (drop non-positive r just in case)
panel2 = panel[panel['r'] > 0].copy()
X2 = np.column_stack([
    np.ones(len(panel2)),
    panel2['scr'].to_numpy(),
    panel2['net_wifi'].to_numpy(),
    panel2['net_mobile'].to_numpy(),
    panel2['T0'].to_numpy(),
])
y2 = np.log(panel2['r'].to_numpy())
coef, *_ = np.linalg.lstsq(X2, y2, rcond=None)
y2hat = X2 @ coef
ss_res2 = float(np.sum((y2 - y2hat)**2))
ss_tot2 = float(np.sum((y2 - np.mean(y2))**2))
r2_2 = 1 - ss_res2/ss_tot2 if ss_tot2>0 else np.nan
print('\nOLS on log r (exp T) coefficients:')
print('log r = c0 + c_scr*scr + c_wifi*net_wifi + c_mob*net_mobile + c_T*(T-30)')
print(coef)
print('R2_log:', r2_2)

# Save a small CSV of fitted coefficients
out = Path('processed')/'discharge_rate_fit_coeffs.csv'
pd.DataFrame({
    'model':['linear_r','log_r'],
    'b0':[beta[0], coef[0]],
    'b_scr':[beta[1], coef[1]],
    'b_wifi':[beta[2], coef[2]],
    'b_mobile':[beta[3], coef[3]],
    'b_T':[beta[4], coef[4]],
    'Tref':[Tref, Tref],
    'R2':[r2, r2_2],
}).to_csv(out, index=False)
print('\nwrote', out)
