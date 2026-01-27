import xgboost as xgb
import numpy as np
import os

# -------------------------------------------------
# MODEL PATH (mounted PVC)
# -------------------------------------------------
MODEL_PATH = "/models/context_model.pkl"

# -------------------------------------------------
# Disease → numeric severity mapping
# -------------------------------------------------
DISEASE_MAP = {
    "None": 0.0,
    "Mild": 0.3,
    "Moderate": 0.6,
    "Severe": 1.0
}


class ContextModel:
    """
    Context model using numeric agronomy signals.
    Crop type & region are applied as calibrated rule-based weights.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    # -------------------------------------------------
    # Lazy model loader (safe for Flask & Kubernetes)
    # -------------------------------------------------
    def _load(self):
        if self.loaded:
            return

        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Context model not found at /models/context_model.pkl"
            )

        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)
        self.loaded = True

    # -------------------------------------------------
    # Context score prediction
    # -------------------------------------------------
    def predict_context_score(
        self,
        region,
        crop_type,
        ndvi,
        disease_status,
        temperature,
        rainfall,
        humidity
    ):
        self._load()

        # -------------------------
        # Numeric feature encoding
        # -------------------------
        disease_score = DISEASE_MAP.get(disease_status, 0.0)
        rainfall_inverse = 1.0 / (rainfall + 1.0)

        X = np.array([[
            ndvi,
            disease_score,
            temperature,
            rainfall_inverse,
            humidity
        ]], dtype=np.float32)

        dmat = xgb.DMatrix(X)

        # -------------------------
        # Model prediction
        # -------------------------
        base_score = float(self.model.predict(dmat)[0])

        # 🔑 NORMALIZE BEFORE WEIGHTING
        base_score = float(np.clip(base_score, 0.0, 1.0))

        # -------------------------
        # Region-based weighting
        # -------------------------
        region_weight = {
            "Tamil Nadu": 1.10,
            "Karnataka": 1.05,
            "Punjab": 0.90,
            "Rajasthan": 0.85
        }.get(region, 1.0)

        # -------------------------
        # Crop-based weighting
        # -------------------------
        crop_weight = {
            "Rice": 1.20,
            "Wheat": 1.00,
            "Maize": 1.05,
            "Cotton": 0.95
        }.get(crop_type, 1.0)

        # -------------------------
        # Final calibrated score
        # -------------------------
        final_score = base_score * region_weight * crop_weight

        return float(np.clip(final_score, 0.0, 1.0))


# -------------------------------------------------
# SAFE SINGLETON (no crash on import)
# -------------------------------------------------
context_model = ContextModel()
