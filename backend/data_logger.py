import csv
import os
from datetime import datetime

# Store newly-labeled rows separately so we don't corrupt the original training CSV,
# which contains many extra columns and a different column order.
DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../dataset/irrigation_new_labels.csv"
)

HEADERS = [
    "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
    "air_temp", "air_humidity", "rainfall", "ph",
    "nitrogen", "phosphorus", "potassium", "needs_water"
]

def append_labeled_row(features, label):
    if len(features) != 12:
        raise ValueError("Expected 12 features")

    row = features + [label]

    file_exists = os.path.exists(DATASET_PATH)

    with open(DATASET_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(HEADERS)

        writer.writerow(row)
