
import os
import sys
import joblib
import numpy as np
import torch
import xgboost as xgb

# Ensure we can import from fed_afta
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fed_afta.models import SimpleTorchEncoder

class ModelWrapper:
    def __init__(self, model_path):
        print(f"Loading Original AFTA+XGBoost model from {model_path}...")
        
        # Load feature means for imputation
        self.feature_means = {}
        means_path = os.path.join(os.path.dirname(model_path), "feature_means.json")
        if os.path.exists(means_path):
            import json
            with open(means_path, "r") as f:
                self.feature_means = json.load(f)
            print("✅ Loaded feature imputation means.")
        else:
            print("⚠️ Warning: feature_means.json not found. Imputation will use 0.0.")

        self.feature_order = [
            "soil_moisture", "temperature", "soil_humidity", "hour", "dayofyear",
            "air_temp", "air_humidity", "rainfall", "ph",
            "nitrogen", "phosphorus", "potassium"
        ]
        
        # Load artifact
        artifact = joblib.load(model_path)
        
        # 1. LOAD ENCODER
        self.device = "cpu"
        self.input_dim = 12
        self.embedding_dim = 32
        
        meta = artifact.get("metadata", {})
        if "n_features" in meta:
            self.input_dim = meta["n_features"]
        if "embedding_dim" in meta:
            self.embedding_dim = meta["embedding_dim"]

        self.encoder = SimpleTorchEncoder(
            input_dim=self.input_dim, 
            embedding_dim=self.embedding_dim, 
            device=self.device
        )
        
        encoder_state = artifact.get("encoder_state")
        if encoder_state:
            self.encoder.load_state_dict(encoder_state)
        self.encoder.eval()

        # 2. LOAD HEAD (XGBoost or None)
        self.head = artifact.get("head")
        self.architecture = meta.get("architecture", "Unknown")
        print(f"🔹 Model Architecture: {self.architecture}")

        # 3. LOAD SCALER
        self.scaler = artifact.get("scaler")
        
        print("✅ AFTA Model loaded successfully!")

    def _impute(self, input_data):
        """
        Handles missing values.
        input_data: list (12 vals) OR dict (partial vals)
        Returns: numpy array of shape (1, 12)
        """
        # If input is a list/array with valid length, use it directly (assuming valid)
        if isinstance(input_data, (list, np.ndarray)):
            arr = np.array(input_data, dtype=np.float32)
            if arr.size == 12:
                return arr.reshape(1, -1)
            
            # Helper: If short list, map to first N features
            if arr.size < 12:
                row = []
                for i, feat in enumerate(self.feature_order):
                    if i < arr.size:
                        row.append(float(arr.flat[i]))
                    else:
                        row.append(self.feature_means.get(feat, 0.0))
                return np.array(row, dtype=np.float32).reshape(1, -1)
 

        # If input is dictionary, fill missing
        if isinstance(input_data, dict):
            row = []
            for feat in self.feature_order:
                val = input_data.get(feat)
                if val is None or val == "":
                    val = self.feature_means.get(feat, 0.0)
                row.append(float(val))
            return np.array(row, dtype=np.float32).reshape(1, -1)
            
        return np.asarray(input_data, dtype=np.float32).reshape(1, -1)

    def _embed(self, X):
        # 1. Handle Missing / Formatting
        X = self._impute(X)
            
        # 2. Apply Scaler
        if hasattr(self, 'scaler') and self.scaler:
            X = self.scaler.transform(X)
            
        return X # Return processed X (for direct prediction) or embeddings?
                 # SimpleTorchEncoder expects Tensor or numpy X and does internal logic.

    def predict(self, input_data):
        X = self._embed(input_data)
        
        if self.head:
            # Hybrid with XGBoost head
            # For hybrid, we need embeddings first
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                emb = self.encoder.network_(X_tensor).cpu().numpy()
            return self.head.predict(emb)[0]
        else:
            # Pure AFTA (Encoder is the classifier)
            return self.encoder.predict(X)[0]

    def predict_proba(self, input_data):
        X = self._embed(input_data)
        
        if self.head:
            # Hybrid with XGBoost head
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                emb = self.encoder.network_(X_tensor).cpu().numpy()
            probs = self.head.predict_proba(emb)
            return probs[0][1]
        else:
            # Pure AFTA (Encoder is the classifier)
            probs = self.encoder.predict_proba(X)
            return probs[0]

    def get_embeddings_and_pred(self, input_data):
        X = self._embed(input_data)
        
        # Calculate embeddings for SHAP/Viz regardless of model type
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            emb = self.encoder.network_(X_tensor).cpu().numpy()

        if self.head:
            probs = self.head.predict_proba(emb)
            prob = probs[0][1]
        else:
            prob = self.encoder.predict_proba(X)[0]
            
        pred = int(prob >= 0.5)
        return emb, prob, pred