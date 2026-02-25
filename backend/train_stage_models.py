import os
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from fed_afta.models import SimpleTorchEncoder

print("Starting Stage-wise AFTA training...")

df = pd.read_csv("dataset/irrigation_stage_dataset.csv")

feature_cols = [
    "soil_moisture", "temperature",
    "soil_humidity", "air_temp",
    "air_humidity", "rainfall",
    "ph", "nitrogen",
    "phosphorus", "potassium",
    "hour", "dayofyear"
]

stages = df["growth_stage"].unique()

os.makedirs("backend/stage_afta_models", exist_ok=True)

for stage in stages:
    print(f"\nTraining AFTA model for stage: {stage}")

    stage_df = df[df["growth_stage"] == stage]

    X = stage_df[feature_cols].values.astype(np.float32)
    y = stage_df["needs_water"].values.astype(np.float32)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = SimpleTorchEncoder(
        input_dim=X.shape[1],
        embedding_dim=16,
        device="cpu"
    )

    encoder.eval()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    with torch.no_grad():
        emb_train = encoder.get_embeddings(X_tensor)

        if isinstance(emb_train, torch.Tensor):
            emb_train = emb_train.detach().cpu().numpy()
        else:
            emb_train = np.asarray(emb_train)

    dtrain = xgb.DMatrix(emb_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4
    }

    head = xgb.train(params, dtrain, num_boost_round=100)

    artifact = {
        "encoder_state": encoder.state_dict(),
        "head": head,
        "metadata": {
            "n_features": X.shape[1]
        }
    }

    save_path = f"backend/stage_afta_models/{stage}_afta.pkl"
    joblib.dump(artifact, save_path)

    print(f"Saved: {save_path}")

print("\nStage-wise AFTA training completed successfully.")