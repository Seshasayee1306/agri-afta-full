
import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. SETUP PATHS & IMPORTS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)

from fed_afta.models import SimpleTorchEncoder

DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset_augmented.csv")
MODEL_PATH = os.path.join(CUR_DIR, "final_model.pkl")

# 2. LOAD & PREPARE DATA
print(f"Loading data from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

FEATURES = [
    "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
    "air_temp", "air_humidity", "rainfall", "ph",
    "nitrogen", "phosphorus", "potassium"
]
TARGET = "needs_water"

# Drop NaNs
df = df.dropna(subset=FEATURES + [TARGET])

X_raw = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# Scale Data (Required for Neural Networks)
print("Scaling data...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# 3. TRAIN PURE AFTA MODEL
print("Training Pure AFTA Model (SimpleTorchEncoder)...")
input_dim = 12
# Initialize model
model = SimpleTorchEncoder(input_dim=input_dim, embedding_dim=32, device='cpu')

# Train using the built-in fit method
# Increasing epochs since we are relying solely on this network now
model.fit(X, y, epochs=100, batch_size=16, lr=0.001, verbose=True)

# 4. EVALUATE
print("Evaluating...")
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"✅ Pure AFTA Training Complete!")
print(f"🔹 Accuracy: {acc * 100:.2f}%")

# 5. SAVE ARTIFACT
# We only save the state dict and scaler. No external head.
print(f"Saving pure model to {MODEL_PATH}...")
artifact = {
    "encoder_state": model.state_dict(),
    "scaler": scaler,
    "metadata": {
        "architecture": "Pure_AFTA_Original",
        "n_features": 12,
        "embedding_dim": 32
    }
}
joblib.dump(artifact, MODEL_PATH)
print("Done.")
