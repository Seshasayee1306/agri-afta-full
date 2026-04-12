import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from sklearn.metrics import accuracy_score, confusion_matrix
from backend.model_loader import ModelWrapper

print("Evaluation script started...")

# -----------------------------------------
# PATHS (run from repo root via `python -m backend.evaluate_models`)
# -----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "dataset" / "irrigation_stage_dataset.csv"
MAIN_MODEL_PATH = REPO_ROOT / "backend" / "final_model.pkl"
STAGE_AFTA_DIR = REPO_ROOT / "backend" / "stage_afta_models"
STATIC_DIR = REPO_ROOT / "backend" / "static"

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
df = pd.read_csv(DATASET_PATH)
df = df.fillna(df.mean(numeric_only=True))

# -----------------------------------------
# LOAD MAIN AFTA MODEL
# -----------------------------------------
model = ModelWrapper(str(MAIN_MODEL_PATH))

# -----------------------------------------
# PRELOAD STAGE AFTA MODELS
# -----------------------------------------
unique_stages = df["growth_stage"].unique()
stage_models = {}

for s in unique_stages:
    stage_model_path = STAGE_AFTA_DIR / f"{s}_afta.pkl"
    stage_models[s] = ModelWrapper(str(stage_model_path))

print("Stage AFTA models loaded.")

# -----------------------------------------
# PREPARE BATCH INPUT FOR AFTA
# -----------------------------------------
# final_model.pkl was trained with this raw feature order.
X_batch = df[
    ["soil_moisture", "temperature", "soil_humidity",
     "hour", "dayofyear", "air_temp",
     "air_humidity", "rainfall", "ph",
     "nitrogen", "phosphorus", "potassium"]
].values.astype(np.float32)

# Batch predict AFTA (main model)
afta_probs = model.predict_proba(X_batch)
afta_threshold = float(getattr(model, "threshold", 0.5))
afta_preds_batch = [int(p >= afta_threshold) for p in afta_probs]

print("Main AFTA predictions completed.")

# -----------------------------------------
# EVALUATION LOOP
# -----------------------------------------
stage_preds = []
afta_preds = []
ensemble_preds = []
true_labels = []

for i, row in df.iterrows():

    if i % 500 == 0:
        print(f"Processing row {i}")

    stage = row["growth_stage"]

    # ---- Stage AFTA prediction ----
    stage_features = np.array([
        row["soil_moisture"], row["temperature"],
        row["soil_humidity"], row["air_temp"],
        row["air_humidity"], row["rainfall"],
        row["ph"], row["nitrogen"],
        row["phosphorus"], row["potassium"],
        row["hour"], row["dayofyear"]
    ]).reshape(1, -1)

    stage_pred = int(stage_models[stage].predict(stage_features))

    # ---- Main AFTA prediction (from batch) ----
    afta_pred = afta_preds_batch[i]

    # ---- Ensemble ----
    ensemble_pred = 1 if (stage_pred + afta_pred) >= 1 else 0

    # ---- Store results ----
    stage_preds.append(int(stage_pred))
    afta_preds.append(int(afta_pred))
    ensemble_preds.append(int(ensemble_pred))
    true_labels.append(int(row["needs_water"]))

print("Prediction loop completed.")

# -----------------------------------------
# METRICS
# -----------------------------------------
stage_acc = accuracy_score(true_labels, stage_preds)
afta_acc = accuracy_score(true_labels, afta_preds)
ensemble_acc = accuracy_score(true_labels, ensemble_preds)

print("\nModel Evaluation Results")
print("------------------------")
print(f"Stage AFTA Accuracy: {stage_acc:.4f}")
print(f"Main AFTA Accuracy: {afta_acc:.4f}")
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# -----------------------------------------
# CREATE STATIC FOLDER IF NOT EXISTS
# -----------------------------------------
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# SAVE ACCURACY BAR GRAPH
# -----------------------------------------
plt.figure()
plt.bar(
    ["Stage AFTA", "Main AFTA", "Ensemble"],
    [stage_acc, afta_acc, ensemble_acc]
)
plt.title("AFTA Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(STATIC_DIR / "accuracy_comparison.png")
plt.close()

# -----------------------------------------
# SAVE CONFUSION MATRIX
# -----------------------------------------
cm = confusion_matrix(true_labels, ensemble_preds)

plt.figure()
if sns is not None:
    sns.heatmap(cm, annot=True, fmt="d")
else:
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(STATIC_DIR / "confusion_matrix.png")
plt.close()

print("\nGraphs saved inside backend/static/")
print("Evaluation completed successfully.")
