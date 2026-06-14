from deltalake import DeltaTable, write_deltalake
import pandas as pd
import os
import numpy as np

DIR_PATH_SILVER = "data/silver"

def interpolate_value(series):

    s = series.copy()

    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(',', '.', regex=False)
        s = pd.to_numeric(s, errors='coerce')

    valid_indices = s.index[s.notna()]
    if valid_indices.empty:
        return s

    nan_indices = s.index[s.isna()]

    for idx in nan_indices:

        prev_valid_indices = valid_indices[valid_indices < idx]
        prev_val = s.loc[prev_valid_indices[-1]] if not prev_valid_indices.empty else None
        next_valid_indices = valid_indices[valid_indices > idx]
        next_val = s.loc[next_valid_indices[0]] if not next_valid_indices.empty else None

        if prev_val is not None and next_val is not None:
            s.loc[idx] = (prev_val + next_val) / 2

        elif prev_val is not None:
            s.loc[idx] = prev_val
        elif next_val is not None:
            s.loc[idx] = next_val

    return s

def save_silver_delta(bronze_table_path):

    try:
        working_path = bronze_table_path

        if not os.path.exists(working_path) and os.path.exists(os.path.join("WeatherReport", working_path)):
            working_path = os.path.join("WeatherReport", working_path)
        silver_dir = DIR_PATH_SILVER

        if not os.path.exists(silver_dir) and os.path.exists(os.path.join("WeatherReport", silver_dir)):
            silver_dir = os.path.join("WeatherReport", silver_dir)
        elif not os.path.exists(silver_dir):
            if os.path.exists("WeatherReport"):
                silver_dir = os.path.join("WeatherReport", "data", "silver")
            os.makedirs(silver_dir, exist_ok=True)

        dt = DeltaTable(working_path)
        df = dt.to_pandas().tail(100).reset_index(drop=True)
        skip_cols = ['Data Medicao', 'injection_timestamp']

        for col in df.columns:

            if col not in skip_cols:
                df[col] = interpolate_value(df[col])

        write_deltalake(silver_dir, df, mode="append")
        return f"Success! Silver data appended to Delta table at {silver_dir}"

    except Exception as e:
        return f"Error in silver processing: {str(e)}"

