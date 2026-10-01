"""
SkyGuard AI — feature engineering

Usage:
    from src.features import build_neighbor_lookup, add_all_features
    nearest_neighbors, station_ids = build_neighbor_lookup(clean_df)
    featured_df = add_all_features(clean_df, nearest_neighbors, station_ids)
"""

import numpy as np
import pandas as pd
from src.config import VALUE_COLS, K_NEIGHBORS


# ---------------------------------------------------------------------------
# Spatial: distance matrix + nearest neighbors
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_neighbor_lookup(df: pd.DataFrame, k: int = K_NEIGHBORS):
    """Returns (nearest_neighbors dict, station_ids array) for spatial comparison."""
    station_coords = df[['station_id', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    station_ids = station_coords['station_id'].values
    n = len(station_ids)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine(
                    station_coords.loc[i, 'lat'], station_coords.loc[i, 'lon'],
                    station_coords.loc[j, 'lat'], station_coords.loc[j, 'lon'],
                )
    dist_df = pd.DataFrame(dist_matrix, index=station_ids, columns=station_ids)

    nearest_neighbors = {
        sid: dist_df.loc[sid].sort_values().iloc[1:k + 1].index.tolist()
        for sid in station_ids
    }
    return nearest_neighbors, station_ids


# ---------------------------------------------------------------------------
# Temporal features (lagged, so a value never contaminates its own baseline)
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame, value_cols=VALUE_COLS) -> pd.DataFrame:
    df = df.sort_values(['station_id', 'date']).reset_index(drop=True)
    grouped = df.groupby('station_id')

    for col in value_cols:
        # shift(1) BEFORE rolling: baseline uses only days strictly before today,
        # otherwise an injected spike inflates its own mean/std and hides itself
        df[f'{col}_roll_mean7'] = grouped[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
        df[f'{col}_roll_std7'] = grouped[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
        df[f'{col}_roll_mean14'] = grouped[col].transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
        df[f'{col}_roll_std3'] = grouped[col].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
        df[f'{col}_diff1'] = grouped[col].transform(lambda s: s.diff())

    for col in value_cols:
        df[f'{col}_zscore'] = (df[col] - df[f'{col}_roll_mean7']) / df[f'{col}_roll_std7']
        df[f'{col}_mean_gap'] = df[f'{col}_roll_mean7'] - df[f'{col}_roll_mean14']

    for col in value_cols:
        same = grouped[col].transform(lambda s: s.diff() == 0)
        df[f'{col}_unchanged_streak'] = same.groupby(df['station_id']).transform(
            lambda s: s.groupby((~s).cumsum()).cumcount()
        )
    return df


# ---------------------------------------------------------------------------
# Multivariate features (temp/humidity/pressure relationships within one reading)
# ---------------------------------------------------------------------------
def add_multivariate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['temp_avg'] = df[['temp_9am', 'temp_3pm']].mean(axis=1)
    df['humidity_avg'] = df[['humidity_9am', 'humidity_3pm']].mean(axis=1)
    df['pressure_avg'] = df[['pressure_9am', 'pressure_3pm']].mean(axis=1)
    df['temp_swing'] = df['temp_3pm'] - df['temp_9am']
    df['humidity_swing'] = df['humidity_3pm'] - df['humidity_9am']
    df['pressure_swing'] = df['pressure_3pm'] - df['pressure_9am']
    df['temp_humidity_ratio'] = df['temp_avg'] / (df['humidity_avg'] + 1e-6)
    df['temp_range'] = df['temp_max'] - df['temp_min']
    return df


# ---------------------------------------------------------------------------
# Spatial features (deviation from nearby stations, normalized per-station)
# ---------------------------------------------------------------------------
def add_spatial_features(df, nearest_neighbors, station_ids,
                          spatial_cols=('temp_avg', 'humidity_avg', 'pressure_avg')) -> pd.DataFrame:
    df = df.sort_values(['station_id', 'date']).reset_index(drop=True)
    pivots = {col: df.pivot(index='date', columns='station_id', values=col) for col in spatial_cols}

    for col in spatial_cols:
        pivot = pivots[col]
        df[f'{col}_neighbor_avg'] = np.nan
        for sid in station_ids:
            neighbor_series = pivot[nearest_neighbors[sid]].mean(axis=1)
            mask = df['station_id'] == sid
            df.loc[mask, f'{col}_neighbor_avg'] = df.loc[mask, 'date'].map(neighbor_series)

        df[f'{col}_spatial_dev'] = df[col] - df[f'{col}_neighbor_avg']

        # normalize away each station's permanent climate offset from its neighbors
        grouped_dev = df.groupby('station_id')[f'{col}_spatial_dev']
        dev_mean = grouped_dev.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
        dev_std = grouped_dev.transform(lambda s: s.shift(1).rolling(14, min_periods=5).std())
        df[f'{col}_spatial_dev_zscore'] = (df[f'{col}_spatial_dev'] - dev_mean) / dev_std

    return df


# ---------------------------------------------------------------------------
# Aggregate features (the "worst signal across any variable" summary)
# ---------------------------------------------------------------------------
def get_contextual_feature_cols(all_cols):
    """Selects only the relative/contextual features — never raw absolute values,
    which confound the model with each station's baseline climate."""
    keywords = ['zscore', 'diff1', 'spatial_dev', 'swing', 'ratio', 'range',
                'std3', 'mean_gap', 'unchanged_streak']
    return [c for c in all_cols if any(k in c for k in keywords)]


def add_aggregate_features(df: pd.DataFrame, contextual_cols) -> pd.DataFrame:
    df = df.copy()
    zscore_cols = [c for c in contextual_cols if 'zscore' in c and 'spatial' not in c]
    diff_cols = [c for c in contextual_cols if 'diff1' in c]
    spatial_zscore_cols = [c for c in contextual_cols if 'spatial_dev_zscore' in c]
    streak_cols = [c for c in contextual_cols if 'unchanged_streak' in c]
    std3_cols = [c for c in contextual_cols if 'std3' in c]

    df['max_abs_zscore'] = df[zscore_cols].abs().max(axis=1)
    df['max_abs_diff'] = df[diff_cols].abs().max(axis=1)
    df['max_abs_spatial_zscore'] = df[spatial_zscore_cols].abs().max(axis=1)
    df['max_unchanged_streak'] = df[streak_cols].max(axis=1)
    df['min_roll_std3'] = df[std3_cols].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def add_all_features(df: pd.DataFrame, nearest_neighbors, station_ids, value_cols=VALUE_COLS):
    """Runs temporal -> multivariate -> spatial -> aggregate, in the correct order."""
    df = add_temporal_features(df, value_cols)
    df = add_multivariate_features(df)
    df = add_spatial_features(df, nearest_neighbors, station_ids)
    df = df.replace([np.inf, -np.inf], np.nan)

    exclude_cols = ['station_id', 'date', 'lat', 'lon', 'is_anomaly',
                     'anomaly_type', 'anomaly_col'] + list(value_cols)
    contextual_cols = get_contextual_feature_cols(
        [c for c in df.columns if c not in exclude_cols]
    )
    df = add_aggregate_features(df, contextual_cols)

    feature_cols = contextual_cols + ['max_abs_zscore', 'max_abs_diff',
                                        'max_abs_spatial_zscore', 'max_unchanged_streak',
                                        'min_roll_std3']
    return df, feature_cols