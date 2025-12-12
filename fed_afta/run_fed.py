# fed_afta/run_fed.py

import os
import pandas as pd
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

    # Initialize encoder
    input_dim = len(features)
    encoder = SimpleTorchEncoder(input_dim=input_dim, embedding_dim=16, device='cpu')

    # Pretrain encoder on the global dataset (small number of epochs for stability)
    X_all = df[features].fillna(0).values
    y_all = df[config['target']].values
    print("Pretraining encoder on global sample (small)...")
    try:
        encoder.fit(X_all, y_all, epochs=5, batch_size=256, lr=1e-3, verbose=True)
    except Exception as e:
        print("Pretrain failed:", e)

    # Initialize server
    server = Server(df, encoder, config)

    # Prepare clients (ensure `client_id` exists)
    if 'client_id' not in df.columns:
        print("Warning: 'client_id' column not found. Creating default client assignment.")
        df['client_id'] = 0  # all data assigned to a single client

    client_dfs = {cid: df[df.client_id == cid].reset_index(drop=True) for cid in sorted(df.client_id.unique())}
    server.register_clients(client_dfs)

    # Run federated rounds
    server.run_rounds(rounds=3)

    print(f"[Server] Final model artifact saved to: {os.path.join(BASE, 'backend', 'final_model.pkl')}")

if __name__ == "__main__":
    main()
