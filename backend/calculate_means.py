import os
import pandas as pd
import json
import numpy as np

# 1. SETUP PATHS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset_augmented.csv")
OUTPUT_PATH = os.path.join(CUR_DIR, "feature_means.json")

# Features (Order matters!)
FEATURES = [
    "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
    "air_temp", "air_humidity", "rainfall", "ph",
    "nitrogen", "phosphorus", "potassium"
]

def calculate_means():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Calculate means
    means = {}
    for feature in FEATURES:
        if feature in df.columns:
            means[feature] = float(df[feature].mean())
        else:
            print(f"Warning: Feature '{feature}' not found in dataset. Using 0.0 as default.")
            means[feature] = 0.0

    # Save to JSON
    print(f"Saving feature means to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(means, f, indent=4)

    print("✅ Feature means calculated successfully:")
    print(json.dumps(means, indent=4))

if __name__ == "__main__":
    calculate_means()
