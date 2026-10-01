"""
SkyGuard AI — synthetic anomaly injection

Injects known, labeled faults into clean data so we have ground truth to
train and evaluate against. Tracks WHICH column was corrupted (anomaly_col)
so evaluation checks the right feature per row.

Usage:
    from src.injection import inject_anomalies
    injected_df = inject_anomalies(clean_df)
"""

import numpy as np
import pandas as pd
from src.config import VALUE_COLS


def inject_anomalies(df: pd.DataFrame, value_cols=VALUE_COLS, fraction: float = 0.04, seed: int = 42) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'normal'
    df['anomaly_col'] = ''

    n_inject = int(len(df) * fraction)
    inject_idx = rng.choice(df.index, size=n_inject, replace=False)
    types = ['spike', 'jump', 'frozen', 'drift']

    for idx in inject_idx:
        if df.at[idx, 'is_anomaly'] == 1:
            continue  # already touched by an earlier injection

        col = rng.choice(value_cols)
        atype = rng.choice(types)
        sid = df.at[idx, 'station_id']

        if atype == 'spike':
            df.at[idx, col] += rng.choice([-1, 1]) * rng.uniform(15, 30)
            df.at[idx, 'is_anomaly'] = 1
            df.at[idx, 'anomaly_type'] = 'spike'
            df.at[idx, 'anomaly_col'] = col

        elif atype == 'jump':
            df.at[idx, col] += rng.choice([-1, 1]) * rng.uniform(8, 15)
            df.at[idx, 'is_anomaly'] = 1
            df.at[idx, 'anomaly_type'] = 'jump'
            df.at[idx, 'anomaly_col'] = col

        elif atype == 'frozen':
            station_rows = df[df['station_id'] == sid].sort_values('date')
            pos = list(station_rows.index).index(idx)
            freeze_len = rng.integers(3, 7)
            freeze_idx = station_rows.index[pos:pos + freeze_len]
            frozen_value = df.at[idx, col]
            df.loc[freeze_idx, col] = frozen_value
            df.loc[freeze_idx, 'is_anomaly'] = 1
            df.loc[freeze_idx, 'anomaly_type'] = 'frozen'
            df.loc[freeze_idx, 'anomaly_col'] = col

        elif atype == 'drift':
            station_rows = df[df['station_id'] == sid].sort_values('date')
            pos = list(station_rows.index).index(idx)
            drift_len = rng.integers(5, 10)
            drift_idx = station_rows.index[pos:pos + drift_len]
            drift_amounts = np.linspace(0, rng.uniform(8, 15), len(drift_idx))
            df.loc[drift_idx, col] = df.loc[drift_idx, col].values + drift_amounts
            df.loc[drift_idx, 'is_anomaly'] = 1
            df.loc[drift_idx, 'anomaly_type'] = 'drift'
            df.loc[drift_idx, 'anomaly_col'] = col

    return df