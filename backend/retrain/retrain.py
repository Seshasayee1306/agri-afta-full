import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

# -----------------------------------------------------
# OPTIONAL: S3 LABELED DATA LOADER
# -----------------------------------------------------
def _s3_client():
    import boto3
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))


def _load_new_labeled_rows_from_s3():
    """
    Loads newly-arrived labeled rows from S3 objects under {S3_PREFIX}/labeled/.

    Each object is expected to be JSON like:
      { "kind":"label", "features":[12 floats], "label":0/1, ... }

    Returns: (df_new, last_key_processed)
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return pd.DataFrame(), None

    base = os.getenv("S3_PREFIX", "agri").strip().strip("/")
    prefix = f"{base}/labeled/"

    state = load_state()
    last_key = state.get("last_s3_key")

    s3 = _s3_client()
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj.get("Key")
            if not k:
                continue
            if last_key and k <= last_key:
                continue
            keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    if not keys:
        return pd.DataFrame(), last_key

    keys = sorted(set(keys))
    rows = []
    for k in keys:
        try:
            body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
            rec = json.loads(body.decode("utf-8"))
            if rec.get("kind") != "label":
                continue
            feats = rec.get("features")
            label = rec.get("label")
            if not isinstance(feats, list) or len(feats) != 12:
                continue
            if label not in (0, 1):
                continue
            rows.append([float(x) for x in feats] + [int(label)])
        except Exception as e:
            print("Skipping S3 key due to parse error:", k, e)

    if not rows:
        return pd.DataFrame(), keys[-1]

    cols = features + ["needs_water"]
    df_new = pd.DataFrame(rows, columns=cols)
    return df_new, keys[-1]

# -----------------------------------------------------
# RETRAIN CONTROL CONFIG
# -----------------------------------------------------
STATE_FILE = "/app/retrain_state/retrain_state.json"
MIN_NEW_ROWS = 50

# -----------------------------------------------------
# PATH FIXES (Docker / K8s safe)
# -----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fed_afta.models import SimpleTorchEncoder
from fed_afta.server import Server

# -----------------------------------------------------
# MODEL CONFIG (UNCHANGED)
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
        return {"last_trained_rows": 0, "last_s3_key": None}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(row_count, last_s3_key=None):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_trained_rows": row_count, "last_s3_key": last_s3_key}, f)

# -----------------------------------------------------
# MAIN RETRAIN FUNCTION
# -----------------------------------------------------
def run_retrain():

    # ---------------- MLflow ----------------
    mlflow.set_tracking_uri("http://mlflow-svc:5000")
    mlflow.set_experiment("AFTA_Federated_Retrain")

    print(f"[{datetime.now()}] Starting federated retraining job")

    dataset_path = os.getenv("BASE_TRAINING_DATASET", "/app/dataset/irrigation_dataset.csv")
    model_output_path = os.path.join(os.path.dirname(__file__), "../final_model.pkl")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Append newly labeled rows from S3 (if configured)
    df_new, last_s3_key = _load_new_labeled_rows_from_s3()
    if len(df_new) > 0:
        print(f"Appending {len(df_new)} new labeled rows from S3.")
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
    else:
        print("No new labeled rows found in S3 (or S3 not configured).")

    # -------------------------------------------------
    # FEATURE SANITIZATION
    # -------------------------------------------------
    before_feat = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep rows; fill missing feature values with 0 (consistent with AFTA pipeline)
    for c in config["features"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[config["features"]] = df[config["features"]].fillna(0)

    print("Feature sanitization:")
    print(f" - Rows before: {before_feat}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after feature sanitization")

    # -------------------------------------------------
    # TARGET SANITIZATION
    # -------------------------------------------------
    before_target = len(df)
    df[config["target"]] = pd.to_numeric(df[config["target"]], errors="coerce")
    df = df.dropna(subset=[config["target"]])
    df[config["target"]] = df[config["target"]].astype(int)
    df = df[df[config["target"]].isin([0, 1])]

    print("Target sanitization:")
    print(f" - Rows before: {before_target}")
    print(f" - Rows after : {len(df)}")

    if len(df) == 0:
        raise ValueError("Dataset empty after target sanitization")

    # -------------------------------------------------
    # CLIENT ID CHECK
    # -------------------------------------------------
    if "client_id" not in df.columns:
        print("client_id column missing from dataset; assigning client_id=0")
        df["client_id"] = 0
    df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce").fillna(0).astype(int)

    # -------------------------------------------------
    # RETRAIN GATING
    # -------------------------------------------------
    state = load_state()
    current_rows = len(df)
    new_rows = current_rows - int(state.get("last_trained_rows", 0))

    print(f"Current rows: {current_rows}")
    print(f"Rows since last retrain: {new_rows}")

    force_retrain = os.getenv("FORCE_RETRAIN", "0") == "1"
    if force_retrain:
        print("Force retrain flag detected. Proceeding regardless of new data count.")

    if not force_retrain and new_rows < MIN_NEW_ROWS and len(df_new) == 0:
        print("Not enough new data. Skipping retraining.")
        return

    # -------------------------------------------------
    # FEDERATED TRAINING + MLFLOW
    # -------------------------------------------------
    with mlflow.start_run(run_name=f"retrain_{datetime.now().isoformat()}"):

        mlflow.log_param("model_type", "AFTA")
        mlflow.log_param("rounds", 3)
        mlflow.log_param("features", len(features))
        mlflow.log_metric("rows_used", current_rows)

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
        # METRICS (AFTA-CORRECT)
        # -------------------------------------------------
        print("Logging federated performance metrics...")

        # Accuracy already computed inside Server
        # AFTA-correct accuracy extraction
        if hasattr(srv, "round_metrics") and len(srv.round_metrics) > 0:
            final_accuracy = float(srv.round_metrics[-1])
        else:
            final_accuracy = 0.0



        mlflow.log_metric("accuracy", final_accuracy)

        print(f"Final Federated Accuracy: {final_accuracy:.4f}")

        # -------------------------------------------------
        # SAVE MODEL ARTIFACT
        # -------------------------------------------------
        # Server.run_rounds already writes a valid artifact to backend/final_model.pkl.
        print(f"Model artifact written to: {model_output_path}")
        if os.path.exists(model_output_path):
            mlflow.log_artifact(model_output_path, artifact_path="model")

            # Optional: upload model to S3 so serving can fetch latest
            bucket = os.getenv("S3_BUCKET")
            if bucket:
                try:
                    base = os.getenv("S3_PREFIX", "agri").strip().strip("/")
                    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    key = f"{base}/models/final_model_{ts}.pkl"
                    latest_key = f"{base}/models/final_model_latest.pkl"
                    s3 = _s3_client()
                    s3.upload_file(model_output_path, bucket, key)
                    s3.upload_file(model_output_path, bucket, latest_key)
                    print("Uploaded model to S3:", key)
                except Exception as e:
                    print("⚠️ S3 model upload failed:", e)

        save_state(current_rows, last_s3_key or state.get("last_s3_key"))

        print(f"[{datetime.now()}] Retraining completed successfully")
        print("Retrain state updated")

# -----------------------------------------------------
# ENTRYPOINT
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
