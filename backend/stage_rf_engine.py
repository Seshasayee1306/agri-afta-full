import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "stage_models")

def load_stage_model(stage):
    model_path = os.path.join(MODEL_DIR, f"{stage}_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    global_path = os.path.join(MODEL_DIR, "global_model.pkl")
    if os.path.exists(global_path):
        return joblib.load(global_path)
    raise FileNotFoundError(f"No model found for stage={stage} and no global fallback")

def predict_stage(stage, feature_dict):
    artifact = load_stage_model(stage)
    X = np.array([[
        feature_dict["soil_moisture"],
        feature_dict["temperature"],
        feature_dict["soil_humidity"],
        feature_dict["air_temp"],
        feature_dict["air_humidity"],
        feature_dict["rainfall"],
        feature_dict["ph"],
        feature_dict["nitrogen"],
        feature_dict["phosphorus"],
        feature_dict["potassium"]
    ]], dtype=float)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.5))
        prob = float(model.predict_proba(X)[0, 1])
        return int(prob >= threshold)

    return int(artifact.predict(X)[0])
