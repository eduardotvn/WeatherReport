import multiprocessing
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.producer import start_producer
from src.consumer import start_consumer

def main():
    print("=== WeatherReport Real-time Simulation ===")
    print("Instructions:")
    print("1. Ensure Kafka is running: 'docker-compose up -d'")
    print("2. The simulation will start a Producer (1 row/sec) and a Consumer (batch 100).")
    print("-" * 40)

    consumer_process = multiprocessing.Process(target=start_consumer)
    consumer_process.start()
    time.sleep(2)

    producer_process = multiprocessing.Process(target=start_producer)
    producer_process.start()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print("\nStopping simulation...")
        producer_process.terminate()
        consumer_process.terminate()
        print("Simulation stopped.")


if __name__ == "__main__":

    main()

