"""
SkyGuard AI — data loading and cleaning

Usage:
    from src.data_prep import load_and_clean
    clean_df = load_and_clean()
"""

import pandas as pd
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH, VALUE_COLS, CITY_COORDS

CORE_COLS = {
    'Date': 'date',
    'Location': 'station_id',
    'Temp9am': 'temp_9am',
    'Temp3pm': 'temp_3pm',
    'MinTemp': 'temp_min',
    'MaxTemp': 'temp_max',
    'Humidity9am': 'humidity_9am',
    'Humidity3pm': 'humidity_3pm',
    'Pressure9am': 'pressure_9am',
    'Pressure3pm': 'pressure_3pm',
}


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV and select + rename only the columns we need."""
    df = pd.read_csv(path)
    core_df = df[list(CORE_COLS.keys())].rename(columns=CORE_COLS)
    return core_df


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lat/lon to every row based on station_id (needed for spatial features)."""
    df = df.copy()
    df['lat'] = df['station_id'].map(lambda x: CITY_COORDS[x][0])
    df['lon'] = df['station_id'].map(lambda x: CITY_COORDS[x][1])
    return df


def interpolate_missing(df: pd.DataFrame, value_cols=VALUE_COLS) -> pd.DataFrame:
    """
    Fill missing values per-station using linear interpolation on that station's
    own time series. Uses .transform(), NOT .apply(), so station_id is never lost
    (this bit us repeatedly during development — see notebook history).
    """
    df = df.sort_values(['station_id', 'date']).reset_index(drop=True)
    df[value_cols] = df.groupby('station_id')[value_cols].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both')
    )
    return df


def drop_stations_with_no_data(df: pd.DataFrame, value_cols=VALUE_COLS) -> pd.DataFrame:
    """
    Drop any station where a column is missing for ALL its rows — interpolation
    can't fill a column that was never recorded at all (e.g. no pressure sensor).
    """
    missing_by_station = df.groupby('station_id')[value_cols].apply(
        lambda g: g.isnull().sum(), include_groups=False
    )
    stations_to_drop = missing_by_station[(missing_by_station > 0).any(axis=1)].index.tolist()
    if stations_to_drop:
        print(f"Dropping stations with no data for some variable: {stations_to_drop}")
    df = df[~df['station_id'].isin(stations_to_drop)].reset_index(drop=True)
    return df


def load_and_clean(raw_path: str = RAW_DATA_PATH, save_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Full cleaning pipeline: load -> rename -> coords -> interpolate -> drop bad stations -> save."""
    df = load_raw(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df = add_coordinates(df)
    df = interpolate_missing(df)
    df = drop_stations_with_no_data(df)

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved clean data: {df.shape} -> {save_path}")
    return df


if __name__ == "__main__":
    load_and_clean()