from deltalake import write_deltalake

import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PATH_BRONZE = os.path.join(BASE_DIR, "data", "bronze")

def save_bronze_delta(new_data):

    try:
        if not os.path.exists(DIR_PATH_BRONZE):
            os.makedirs(DIR_PATH_BRONZE, exist_ok=True)
        if len(new_data) < 100:
            return(f"Error, wrong size: {len(new_data)}")

        elif not isinstance(new_data, pd.DataFrame):
            return ("Error, new_data is not a pandas DataFrame")

        new_data["injection_timestamp"] = datetime.now()

        if 'Unnamed: 11' in new_data.columns:
            new_data = new_data.drop(columns=['Unnamed: 11'])

        write_deltalake(DIR_PATH_BRONZE, new_data, mode="append")
        return "Success!"

    except Exception as e:
        return (f"Error in bronze: {e}")
