import os
import sys
import pandas as pd
import joblib
from datetime import datetime
import json

# -----------------------------------------------------
# RETRAIN CONTROL CONFIG
# -----------------------------------------------------
STATE_FILE = "/app/retrain_state/retrain_state.json"

MIN_NEW_ROWS = 50   # retrain only if at least these many new rows exist

# -----------------------------------------------------
# PATH FIXES (for Docker / K8s execution)
# -----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fed_afta.models import SimpleTorchEncoder
from fed_afta.server import Server

# -----------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_trained_rows": 0}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(row_count):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_trained_rows": row_count}, f)

# -----------------------------------------------------
# MAIN RETRAIN FUNCTION
# -----------------------------------------------------
def run_retrain():
    print(f"[{datetime.now()}] Starting federated retraining job")

    # Paths relative to /app in Docker
    dataset_path = "/app/dataset/irrigation_dataset.csv"

    model_output_path = os.path.join(
        os.path.dirname(__file__),
        "../final_model.pkl"
    )

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)


    # -------------------------------------------------
    # FEATURE SANITIZATION (CRITICAL)
    # -------------------------------------------------
    feature_cols = config["features"]

    # Drop rows with NaN or inf in features
    before_feat = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    print(f"Feature sanitization:")
    print(f" - Rows before: {before_feat}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after feature sanitization")


    # -------------------------------------------------
    # DATA SANITY CHECKS (CRITICAL)
    # -------------------------------------------------
    target_col = "needs_water"

# Drop rows where target is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])

# Force target to binary int {0,1}
    df[target_col] = df[target_col].astype(int)

# Ensure valid label range
    df = df[df[target_col].isin([0, 1])]

    print(f"Sanitized dataset:")
    print(f" - Rows before: {initial_rows}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after sanitization — check labels")


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
    # MODEL CONFIG
    # -------------------------------------------------
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
