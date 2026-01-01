import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------------------------------
# RETRAIN CONTROL CONFIG
# -----------------------------------------------------
STATE_FILE = "/app/retrain_state/retrain_state.json"
MIN_NEW_ROWS = 50   # retrain only if at least these many new rows exist

# -----------------------------------------------------
# PATH FIXES (Docker / K8s safe)
# -----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fed_afta.models import SimpleTorchEncoder
from fed_afta.server import Server

# -----------------------------------------------------
# MODEL CONFIG (MOVED UP — REQUIRED)
# -----------------------------------------------------
features = [
    "soil_moisture",
    "temperature",
    "soil_humidity",
    "hour",
    "dayofyear",
    "air_temp",
    "air_humidity",
    "rainfall",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium"
]

config = {
    "features": features,
    "target": "needs_water",
    "active_k": 500
}

# -----------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_trained_rows": 0}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(row_count):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_trained_rows": row_count}, f)

# -----------------------------------------------------
# MAIN RETRAIN FUNCTION
# -----------------------------------------------------
def run_retrain():
    print(f"[{datetime.now()}] Starting federated retraining job")

    # Dataset path (Docker/K8s absolute)
    dataset_path = "/app/dataset/irrigation_dataset.csv"
    model_output_path = os.path.join(os.path.dirname(__file__), "../final_model.pkl")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # -------------------------------------------------
    # FEATURE SANITIZATION
    # -------------------------------------------------
    feature_cols = config["features"]

    before_feat = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    print("Feature sanitization:")
    print(f" - Rows before: {before_feat}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after feature sanitization")

    # -------------------------------------------------
    # TARGET SANITIZATION
    # -------------------------------------------------
    target_col = config["target"]

    before_target = len(df)
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    df = df[df[target_col].isin([0, 1])]

    print("Target sanitization:")
    print(f" - Rows before: {before_target}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after target sanitization")

    # -------------------------------------------------
    # CLIENT ID CHECK (REQUIRED FOR FEDERATION)
    # -------------------------------------------------
    if "client_id" not in df.columns:
        raise ValueError("client_id column missing from dataset")

    # -------------------------------------------------
    # RETRAIN GATING LOGIC
    # -------------------------------------------------
    state = load_state()
    current_rows = len(df)
    new_rows = current_rows - state["last_trained_rows"]

    print(f"Current rows: {current_rows}")
    print(f"Rows since last retrain: {new_rows}")

    if new_rows < MIN_NEW_ROWS:
        print("Not enough new data. Skipping retraining.")
        return

    # -------------------------------------------------
    # FEDERATED TRAINING
    # -------------------------------------------------
    print("Initializing SimpleTorch encoder...")
    encoder = SimpleTorchEncoder(
        input_dim=len(features),
        embedding_dim=16,
        device="cpu"
    )

    print("Initializing federated server...")
    srv = Server(df, encoder, config)

    print("Registering federated clients...")
    client_dfs = {
        cid: df[df.client_id == cid].reset_index(drop=True)
        for cid in sorted(df.client_id.unique())
    }
    srv.register_clients(client_dfs)

    print("Running 3 federated learning rounds...")
    srv.run_rounds(rounds=3)

    # -------------------------------------------------
    # SAVE MODEL + UPDATE STATE
    # -------------------------------------------------
    print(f"Saving trained model to: {model_output_path}")
    joblib.dump(srv.global_model, model_output_path)

    save_state(current_rows)

    print(f"[{datetime.now()}] Retraining completed successfully")
    print("Retrain state updated")

# -----------------------------------------------------
# ENTRYPOINT (K8s Job compatible)
# -----------------------------------------------------
if __name__ == "__main__":
    try:
        run_retrain()
        print("✓ Retraining job completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
