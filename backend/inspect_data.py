import pandas as pd
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset.csv")

if not os.path.exists(DATASET_PATH):
    print("Dataset not found!")
    exit(1)

df = pd.read_csv(DATASET_PATH)
target = "needs_water"

print("Dataset Head:")
print(df.head())

print("\nClass Distribution:")
print(df[target].value_counts())

print("\nBasic Statistics:")
print(df.describe())

# Check a few specific 'dry' rows if possible
print("\nSample 'Dry' Rows (Soil Moisture < 20):")
print(df[df['soil_moisture'] < 20][[target, 'soil_moisture', 'temperature', 'rainfall']].head(10))
