import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from fed_afta.models import SimpleTorchEncoder

print("Starting Stage-wise AFTA training...")

# ---------------- MLflow Setup ----------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("AFTA_Stage_Specific_Training")
# ---------------------------------------------


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

# sanitize once
df = df.replace([np.inf, -np.inf], np.nan)
for c in feature_cols + ["needs_water"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["needs_water", "growth_stage"])
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
df["needs_water"] = df["needs_water"].astype(int).clip(0, 1)

for stage in stages:
    print(f"\nTraining AFTA model for stage: {stage}")

    stage_df = df[df["growth_stage"] == stage]
    if len(stage_df) < 200:
        print(f"Skipping {stage}: too few rows ({len(stage_df)})")
        continue

    X = stage_df[feature_cols].values.astype(np.float32)
    y = stage_df["needs_water"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    with mlflow.start_run(run_name=f"stage_{stage}_{datetime.now().strftime('%m%d_%H%M')}"):
        mlflow.log_param("growth_stage", stage)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", len(stage_df))

        encoder = SimpleTorchEncoder(
            input_dim=X.shape[1],
            embedding_dim=16,
            device="cpu"
        )

        # Train encoder (previous code used random untrained embeddings).
        encoder.fit(X_train, y_train, epochs=15, batch_size=256, lr=1e-3, verbose=False)
        emb_train = np.asarray(encoder.get_embeddings(X_train), dtype=np.float32)
        emb_val = np.asarray(encoder.get_embeddings(X_val), dtype=np.float32)

        dtrain = xgb.DMatrix(emb_train, label=y_train)
        dval = xgb.DMatrix(emb_val, label=y_val)

        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        spw = (neg / max(pos, 1.0)) if pos > 0 else 1.0

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr"],
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "scale_pos_weight": spw,
            "verbosity": 0,
        }

        # Log params to MLflow
        mlflow.log_params(params)

        head = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Calibrate threshold on validation split for better F1.
        val_probs = head.predict(dval)
        thresholds = np.linspace(0.1, 0.9, 161)
        best_threshold = 0.5
        best_f1 = -1.0
        for t in thresholds:
            f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_threshold = float(t)

        artifact = {
            "encoder_state": encoder.state_dict(),
            "head": head,
            "threshold": best_threshold,
            "metadata": {
                "n_features": X.shape[1],
                "stage": str(stage),
                "val_f1": best_f1,
                "best_iteration": int(getattr(head, "best_iteration", -1)),
            }
        }

        save_path = f"backend/stage_afta_models/{stage}_afta.pkl"
        joblib.dump(artifact, save_path)

        # Log metrics and artifacts to MLflow
        mlflow.log_metric("val_f1", best_f1)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_iteration", int(getattr(head, "best_iteration", -1)))
        mlflow.log_artifact(save_path, artifact_path="models")

        print(f"Saved: {save_path} | val_f1={best_f1:.4f} | threshold={best_threshold:.3f}")


print("\nStage-wise AFTA training completed successfully.")
