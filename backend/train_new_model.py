
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. SETUP PATHS
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset.csv")
MODEL_PATH = os.path.join(CUR_DIR, "final_model.pkl")

# 2. LOAD DATA
print(f"Loading data from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# Features (Order matters!)
FEATURES = [
    "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
    "air_temp", "air_humidity", "rainfall", "ph",
    "nitrogen", "phosphorus", "potassium"
]
TARGET = "needs_water"

# Drop missing values
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# 3. TRAIN RANDOM FOREST (High Accuracy Config)
print("Training Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 4. MEASURE ACCURACY
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"✅ Model Training Complete!")
print(f"🔹 Accuracy on Training Data: {acc * 100:.2f}%")

# 5. SAVE WHOLE MODEL
print(f"Saving model to {MODEL_PATH}...")
joblib.dump(clf, MODEL_PATH)
print("Done.")
