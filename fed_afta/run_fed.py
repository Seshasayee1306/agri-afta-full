# fed_afta/run_fed.py

import os
import pandas as pd
import numpy as np
from fed_afta.models import SimpleTorchEncoder
from fed_afta.server import Server

def main():
    # Base directory
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load dataset
    data_path = os.path.join(BASE, "dataset", "irrigation_dataset.csv")
    print("Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    print("Rows:", len(df))

    # Define features and target
    features = [
        "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
        "air_temp", "air_humidity", "rainfall", "ph",
        "nitrogen", "phosphorus", "potassium"
    ]
    config = {"features": features, "target": "needs_water", "active_k": 200}

    # Check if target exists
    if config['target'] not in df.columns:
        raise ValueError(f"Target column '{config['target']}' not found in dataset!")

    # ----------------------------
    # Dataset sanitization
    # ----------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure client_id exists and is clean
    if "client_id" not in df.columns:
        print("Warning: 'client_id' column not found. Creating default client assignment.")
        df["client_id"] = 0
    df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce").fillna(0).astype(int)

    # Ensure valid target (0/1) and no NaNs
    df[config["target"]] = pd.to_numeric(df[config["target"]], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[config["target"]])
    df[config["target"]] = df[config["target"]].astype(int)
    df = df[df[config["target"]].isin([0, 1])]
    print(f"Target sanitization: {before} -> {len(df)} rows")

    # Ensure required features present and numeric.
    # Do NOT drop rows for missing sensor columns; keep rows and fill missing with 0
    # (consistent with the AFTA pipeline elsewhere in this repo).
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nan_counts = df[features].isna().sum().sum()
    if nan_counts:
        print(f"Feature sanitization: filling {int(nan_counts)} NaNs with 0")
        df[features] = df[features].fillna(0)

    # Initialize encoder
    input_dim = len(features)
    encoder = SimpleTorchEncoder(input_dim=input_dim, embedding_dim=16, device='cpu')

    # Pretrain encoder on the global dataset (small number of epochs for stability)
    X_all = df[features].fillna(0).values
    y_all = df[config['target']].values.astype(np.float32)
    print("Pretraining encoder on global sample (small)...")
    try:
        y_all = np.clip(y_all, 0.0, 1.0)
        encoder.fit(X_all, y_all, epochs=5, batch_size=256, lr=1e-3, verbose=True)
    except Exception as e:
        print("Pretrain failed:", e)

    # Initialize server
    server = Server(df, encoder, config)

    client_dfs = {cid: df[df.client_id == cid].reset_index(drop=True) for cid in sorted(df.client_id.unique())}
    server.register_clients(client_dfs)

    # Run federated rounds
    server.run_rounds(rounds=3)

    print(f"[Server] Final model artifact saved to: {os.path.join(BASE, 'backend', 'final_model.pkl')}")

if __name__ == "__main__":
    main()
