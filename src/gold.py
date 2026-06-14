from deltalake import DeltaTable, write_deltalake
import pandas as pd
import os

DIR_PATH_GOLD = "data/gold"
DIR_PATH_SILVER = "data/silver"

def save_gold_delta(silver_table_path):

    try:

        working_path = silver_table_path
        if not os.path.exists(working_path) and os.path.exists(os.path.join("WeatherReport", working_path)):
            working_path = os.path.join("WeatherReport", working_path)

        gold_dir = DIR_PATH_GOLD
        if not os.path.exists(gold_dir) and os.path.exists(os.path.join("WeatherReport", gold_dir)):
            gold_dir = os.path.join("WeatherReport", gold_dir)
        elif not os.path.exists(gold_dir):

            if os.path.exists("WeatherReport"):

                gold_dir = os.path.join("WeatherReport", "data", "gold")

            os.makedirs(gold_dir, exist_ok=True)
    
        dt_silver = DeltaTable(working_path)
        df_full_silver = dt_silver.to_pandas()

        df_full_silver['Data Medicao'] = pd.to_datetime(df_full_silver['Data Medicao'])

        df_full_silver = df_full_silver.sort_values('Data Medicao').reset_index(drop=True)

        new_batch_start = max(0, len(df_full_silver) - 100)

        gold_records = []
        for i in range(new_batch_start, len(df_full_silver)):

            start_idx = max(0, i - 6)
            window = df_full_silver.iloc[start_idx : i + 1].copy()
            window['Window_Reference_Day'] = df_full_silver.iloc[i]['Data Medicao']
            gold_records.append(window)

        if not gold_records:
            return "No new data to process for Gold."

        df_gold = pd.concat(gold_records, ignore_index=True)

        write_deltalake(gold_dir, df_gold, mode="append")

        dt_gold = DeltaTable(gold_dir)
        version = dt_gold.version()

        print(f"\n[Retraining Trigger] New gold packet reached (Delta version {version}). Updating LSTM model...")

        try:

            from src.models.training import train
            train()

        except Exception as train_error:
            print(f"Failed to trigger retraining: {train_error}")
        return f"Success! Gold data (windows) appended to Delta table at {gold_dir}"
    except Exception as e:
        return f"Error in gold processing: {str(e)}"

