import os
import joblib
import numpy as np
import pandas as pd
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)

from fed_afta.models import SimpleTorchEncoder
from sklearn.preprocessing import MinMaxScaler

# Load model
MODEL_PATH = os.path.join(CUR_DIR, "final_model.pkl")
artifact = joblib.load(MODEL_PATH)

# Load scaler and encoder
scaler = artifact['scaler']
encoder_state = artifact['encoder_state']
encoder = SimpleTorchEncoder(input_dim=12, embedding_dim=32)
encoder.load_state_dict(encoder_state)
encoder.eval()

# Test case: Needs Water (Dry soil, high temp, no rain)
test_input = np.array([[10, 35, 20, 14, 200, 38, 30, 0, 6.5, 100, 50, 50]], dtype=np.float32)
print("Test Input (RAW):", test_input)

# Scale it
test_scaled = scaler.transform(test_input)
print("Test Input (SCALED):", test_scaled)

# Predict
pred = encoder.predict(test_scaled)
prob = encoder.predict_proba(test_scaled)

print(f"Prediction: {pred}")
print(f"Probability: {prob}")

# Check what the dataset looks like for similar cases
DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset.csv")
df = pd.read_csv(DATASET_PATH)

print("\nDataset samples with LOW soil moisture (<15) and LOW rainfall (<10):")
dry_samples = df[(df['soil_moisture'] < 15) & (df['rainfall'] < 10)][['soil_moisture', 'temperature', 'rainfall', 'needs_water']].head(20)
print(dry_samples)
print(f"\nneeds_water distribution in dry samples: {dry_samples['needs_water'].value_counts()}")
