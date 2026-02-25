import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "stage_models")


def load_stage_model(stage):
    model_path = os.path.join(MODEL_DIR, f"{stage}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stage model not found: {model_path}")

    return joblib.load(model_path)


def predict_stage(stage, feature_dict):
    model = load_stage_model(stage)

    features = np.array([[
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
    ]])

    return int(model.predict(features)[0])