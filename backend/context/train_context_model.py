# backend/context/train_context_model.py

import os
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from backend.context.preprocess import preprocess_context_data

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATASET_PATH = "/app/dataset/Smart_Farming_Crop_Yield_2024.csv"
MODEL_PATH = "/models/context_model.pkl"
FEATURES_PATH = "/models/context_feature_names.npy"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-svc:5000"
)

# -------------------------------------------------
# TRAIN FUNCTION
# -------------------------------------------------
def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Context_Agronomy_Model")

    # Load + preprocess
    X, y = preprocess_context_data(DATASET_PATH)

    # ✅ Normalize target → context risk score
    y = (y - y.min()) / (y.max() - y.min())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save feature names
    np.save(FEATURES_PATH, X_train.columns.to_list())

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dtest = xgb.DMatrix(X_test.values, label=y_test.values)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "verbosity": 0
    }

    with mlflow.start_run():
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300
        )

        preds = model.predict(dtest)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.xgboost.log_model(model, artifact_path="model")

        model.save_model(MODEL_PATH)

        print("✅ Context model trained & saved successfully")
        print("📦 Model path:", MODEL_PATH)
        print("📄 Feature names saved:", FEATURES_PATH)


if __name__ == "__main__":
    train()
