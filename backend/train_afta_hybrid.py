
import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. SETUP PATHS & IMPORTS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)

from fed_afta.models import SimpleTorchEncoder

DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset.csv")
MODEL_PATH = os.path.join(CUR_DIR, "final_model.pkl")

# 2. LOAD DATA
print(f"Loading data from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

FEATURES = [
    "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
    "air_temp", "air_humidity", "rainfall", "ph",
    "nitrogen", "phosphorus", "potassium"
]
TARGET = "needs_water"

from sklearn.preprocessing import MinMaxScaler

# Drop NaNs
df = df.dropna(subset=FEATURES + [TARGET])

X_raw = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# 🚨 AFTA (Neural Net) NEEDS SCALED DATA!
print("Scaling data for AFTA Encoder...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# 3. TRAIN AFTA ENCODER
print("Training AFTA Encoder (Neural Network)...")
input_dim = 12
encoder = SimpleTorchEncoder(input_dim=input_dim, embedding_dim=32) # Increased to 32 for better features

# We train the encoder longer to ensure it learns well
encoder.fit(X, y, epochs=200, batch_size=16, lr=0.005, verbose=True) # Increased epochs, smaller batch

# 4. EXTRACT EMBEDDINGS
print("Extracting AFTA Embeddings...")
embeddings = encoder.get_embeddings(X)

# 5. TRAIN RANDOM FOREST HEAD (ON EMBEDDINGS)
print("Training Random Forest Head on Embeddings...")
# Optimized for Accuracy & Robustness
rf = RandomForestClassifier(
    n_estimators=200,       # More trees for stability
    max_depth=None,         # Allow full depth to capture complex patterns
    min_samples_leaf=2,     # Small leaf size to capture specific cases (like dry soil)
    class_weight='balanced', # 🚨 CRITICAL: Handle class imbalance (Needs Water is minority)
    random_state=42
)
y_int = y.astype(int) 
rf.fit(embeddings, y_int)

# 6. VERIFY ACCURACY
y_pred = rf.predict(embeddings)
acc = accuracy_score(y_int, y_pred)
print(f"✅ Hybrid Model Training Complete!")
print(f"🔹 Accuracy on Training Data: {acc * 100:.2f}%")

# 7. SAVE HYBRID ARTIFACT
print(f"Saving hybrid model to {MODEL_PATH}...")
artifact = {
    "encoder_state": encoder.state_dict(),
    "head": rf,
    "scaler": scaler,  # ✅ SAVE SCALER
    "metadata": {
        "architecture": "AFTA_Hybrid_RF",
        "n_features": 12,
        "embedding_dim": 32
    }
}
joblib.dump(artifact, MODEL_PATH)
print("Done.")
