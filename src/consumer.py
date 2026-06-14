import json
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from kafka import KafkaConsumer
from src.bronze import save_bronze_delta
from src.silver import save_silver_delta
from src.gold import save_gold_delta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")

def start_consumer(topic="weather"):
    try:

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='weather-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        print(f"Consumer started, waiting for messages on topic '{topic}'...")

        batch = []

        for message in consumer:

            data = message.value
            batch.append(data)

            if len(batch) >= 100:
                print(f"\n--- Batch of 100 records reached. Triggering Delta Pipeline ---")

                df_batch = pd.DataFrame(batch)

                bronze_status = save_bronze_delta(df_batch)

                print(f"Bronze: {bronze_status}")

                if "Success" in bronze_status:

                    silver_status = save_silver_delta(BRONZE_DIR)
                    print(f"Silver: {silver_status}")

                    if "Success" in silver_status:
                        gold_status = save_gold_delta(SILVER_DIR)
                        print(f"Gold: {gold_status}")

                batch = []

                print("--- Batch Processed. Waiting for next batch ---\n")

    except Exception as e:

        print(f"Consumer Error: {e}")

if __name__ == "__main__":

    start_consumer()

