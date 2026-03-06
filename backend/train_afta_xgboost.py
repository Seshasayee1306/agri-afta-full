
import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
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

# Drop NaNs
df = df.dropna(subset=FEATURES + [TARGET])

X_raw = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# Scale (Essential for AFTA to work)
print("Scaling data...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# 3. TRAIN AFTA ENCODER
print("Training AFTA Encoder (Neural Network)...")
input_dim = 12
encoder = SimpleTorchEncoder(input_dim=input_dim, embedding_dim=32, device='cpu')
encoder.fit(X, y, epochs=100, batch_size=16, lr=0.005, verbose=True)

# 4. EXTRACT EMBEDDINGS
print("Extracting Embeddings for XGBoost...")
embeddings = encoder.get_embeddings(X)

# 5. TRAIN XGBOOST HEAD
# This is the original intended architecture (Fed-TabNet usually uses XGBoost)
print("Training XGBoost Classifier...")
xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
y_int = y.astype(int)
xgb.fit(embeddings, y_int)

# 6. EVALUATE
y_pred = xgb.predict(embeddings)
acc = accuracy_score(y_int, y_pred)
print(f"✅ Original AFTA+XGBoost Training Complete!")
print(f"🔹 Accuracy: {acc * 100:.2f}%")

# 7. SAVE ARTIFACT
print(f"Saving model to {MODEL_PATH}...")
artifact = {
    "encoder_state": encoder.state_dict(),
    "head": xgb,     # XGBoost
    "scaler": scaler,
    "metadata": {
        "architecture": "AFTA_Original_XGBoost",
        "n_features": 12,
        "embedding_dim": 32
    }
}
joblib.dump(artifact, MODEL_PATH)
print("Done.")
