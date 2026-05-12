# generates fake cloud data for training the model
# run this once to create data/sample_dataset.csv

import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

NUM_RECORDS = 2000
ANOMALY_RATE = 0.05  # 5% of data will be anomalous
OUTPUT_PATH = "data/sample_dataset.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dataset():
    random.seed(42)
    np.random.seed(42)

    records = []
    start_time = datetime.now()

    for i in range(NUM_RECORDS):
        ts = start_time + timedelta(seconds=i)
        server = random.randint(1, 5)

        if random.random() < ANOMALY_RATE:
            # anomaly - everything goes crazy
            cpu = round(np.random.uniform(85, 100), 2)
            mem = round(np.random.uniform(85, 100), 2)
            disk = round(np.random.uniform(300, 500), 2)
            net = round(np.random.uniform(800, 1200), 2)
            flag = 1
        else:
            # normal behavior
            cpu = round(np.random.normal(50, 10), 2)
            mem = round(np.random.normal(60, 8), 2)
            disk = round(np.random.normal(100, 20), 2)
            net = round(np.random.normal(300, 50), 2)
            flag = 0

        records.append((ts, server, cpu, mem, disk, net, flag))

    df = pd.DataFrame(records, columns=[
        "timestamp", "server_id", "cpu_usage", "memory_usage",
        "disk_io", "network_traffic", "anomaly_flag"
    ])

    # save it
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    anomaly_count = df["anomaly_flag"].sum()
    logger.info(f"generated {len(df)} rows ({anomaly_count} anomalies) -> {OUTPUT_PATH}")
    return True


if __name__ == "__main__":
    generate_dataset()
    print("dataset ready!")
