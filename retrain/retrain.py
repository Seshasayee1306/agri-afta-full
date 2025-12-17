import os
import sys
import pandas as pd
import joblib
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fed_afta.models import TabNetEncoder
from fed_afta.server import Server

def run_retrain():
    print(f"[{datetime.now()}] Starting federated retraining...")
    
    # Paths relative to /app in Docker
    dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/irrigation_dataset.csv")
    model_output_path = os.path.join(os.path.dirname(__file__), "../final_model.pkl")
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    features = [
        "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
        "air_temp", "air_humidity", "rainfall", "ph", "nitrogen", 
        "phosphorus", "potassium"
    ]
    
    config = {
        'features': features,
        'target': 'needs_water',
        'active_k': 500
    }
    
    print("Initializing TabNet encoder...")
    encoder = TabNetEncoder(device='cpu')
    
    print("Initializing federated server...")
    srv = Server(df, encoder, config)
    
    print("Registering clients...")
    client_dfs = {
        cid: df[df.client_id == cid].reset_index(drop=True)
        for cid in sorted(df.client_id.unique())
    }
    srv.register_clients(client_dfs)
    
    print(f"Running 3 federated learning rounds...")
    srv.run_rounds(rounds=3)
    
    print(f"Model saved to: {model_output_path}")
    print(f"[{datetime.now()}] Retraining completed successfully!")

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