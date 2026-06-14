import pandas as pd
import time
import json
from kafka import KafkaProducer

def start_producer(csv_path="data/dados_A315_D_2010-01-01_2026-05-31.csv", topic="weather"):

    try:

        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')

        )

        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path, sep=";")
        print(f"Starting stream to topic '{topic}'...")

        for index, row in df.iterrows():
            message = row.to_dict()
            producer.send(topic, message)
            if index % 10 == 0:
                print(f"Sent {index} records...")
            time.sleep(1)

    except Exception as e:
        print(f"Producer Error: {e}")

if __name__ == "__main__":
    start_producer()

