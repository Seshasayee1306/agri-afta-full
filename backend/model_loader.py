import os
import sys
import joblib
import numpy as np
import torch
import xgboost as xgb
from scipy.special import expit as sigmoid   # For logitraw models

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fed_afta.models import SimpleTorchEncoder


def _infer_input_dim_from_state(encoder_state):
    if not encoder_state:
        return None
    for k, v in encoder_state.items():
        if "weight" in k:
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if arr.ndim == 2:
                return arr.shape[1]
    return None


class ModelWrapper:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(CUR_DIR, "final_model.pkl")
        elif not os.path.isabs(model_path):
            model_path = os.path.join(CUR_DIR, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load artifact
        artifact = joblib.load(model_path)
        encoder_state = artifact.get("encoder_state")
        head = artifact.get("head")

        if encoder_state is None or head is None:
            raise RuntimeError("Model artifact missing encoder or head.")

        # Infer feature dimension
        input_dim = _infer_input_dim_from_state(encoder_state)
        if input_dim is None:
            meta = artifact.get("metadata", {})
            input_dim = meta.get("n_features", artifact.get("n_features", None))

        if input_dim is None:
            raise RuntimeError("Unable to infer input_dim for encoder.")

        # Build encoder
        self.encoder = SimpleTorchEncoder(
            input_dim=input_dim,
            embedding_dim=16,
            device="cpu"
        )
        self.encoder.load_state_dict(encoder_state)
        self.encoder.eval()

        # XGBoost head
        self.head: xgb.Booster = head

        # Detect XGB output type
        self.is_logitraw = False
        params = self.head.attributes()
        if "objective" in params and "logitraw" in params["objective"]:
            self.is_logitraw = True

    # -----------------------------------------------------
    # FIXED _embed FUNCTION (no indentation issues)
    # -----------------------------------------------------
    def _embed(self, X):
        # Ensure numpy
        X = np.asarray(X, dtype=np.float32)

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Get embeddings from encoder
        emb = self.encoder.get_embeddings(X_tensor)

        # -------- Normalize output type --------
        # If already numpy
        if isinstance(emb, np.ndarray):
            return emb

        # List → numpy
        if isinstance(emb, list):
            return np.array(emb, dtype=np.float32)

        # Tensor → numpy
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().numpy()

        # Fallback
        return np.asarray(emb, dtype=np.float32)

    # -----------------------------------------------------

    def _predict_xgb(self, emb):
        dmat = xgb.DMatrix(emb)
        raw = self.head.predict(dmat)

        # Convert margin → probability
        if self.is_logitraw:
            prob = sigmoid(raw)
        else:
            prob = raw

        return prob

    def predict_proba(self, X):
        emb = self._embed(X)
        prob = self._predict_xgb(emb)
        return prob.tolist()

    def predict(self, X):
        prob = self.predict_proba(X)[0]
        return int(prob >= 0.5)

    def get_embeddings_and_pred(self, X):
        emb = self._embed(X)
        prob = self._predict_xgb(emb)[0]
        pred = int(prob >= 0.5)
        return emb, prob, pred
