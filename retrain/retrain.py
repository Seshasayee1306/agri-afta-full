# retrain/retrain.py
import pandas as pd
from fed_afta.models import TabNetEncoder
from fed_afta.server import Server
import joblib

def run_retrain():
    df = pd.read_csv("../dataset/irrigation_dataset.csv")
    features = ["soil_moisture","temperature","soil_humidity","hour","dayofyear","air_temp","air_humidity","rainfall","ph","nitrogen","phosphorus","potassium"]
    config = {'features':features, 'target':'needs_water', 'active_k':500}
    encoder = TabNetEncoder(device='cpu')
    srv = Server(df, encoder, config)
    client_dfs = {cid: df[df.client_id==cid].reset_index(drop=True) for cid in sorted(df.client_id.unique())}
    srv.register_clients(client_dfs)
    srv.run_rounds(rounds=3)
    # final_model written by server to ../backend/final_model.pkl

if __name__ == "__main__":
    run_retrain()
