import xgboost as xgb
import numpy as np
import os
import pandas as pd
import threading

# -------------------------------------------------
# MODEL PATH (mounted PVC)
# -------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_UBJ = os.path.join(BASE_DIR, "context_model.ubj")
MODEL_PATH_LEGACY_PKL = os.path.join(BASE_DIR, "context_model.pkl")
_TRAIN_LOCK = threading.Lock()

# -------------------------------------------------
# Disease → numeric severity mapping
# -------------------------------------------------
DISEASE_MAP = {
    "None": 0.0,
    "Mild": 0.3,
    "Moderate": 0.6,
    "Severe": 1.0
}

def _repo_root() -> str:
    # backend/context -> backend -> repo root
    return os.path.dirname(os.path.dirname(BASE_DIR))


def _smart_farming_dataset_path() -> str:
    return os.path.join(_repo_root(), "dataset", "Smart_Farming_Crop_Yield_2024.csv")


def _normalize_yield(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if y_max <= y_min:
        return np.zeros_like(y, dtype=np.float32)
    return (y - y_min) / (y_max - y_min)


def _compute_group_weights(df: pd.DataFrame):
    """
    Deterministic weights derived from the Smart Farming training dataset.
    We scale by the mean normalized yield per group, relative to global mean.
    """
    weights = {
        "region": {},
        "crop_type": {},
    }

    if df is None or df.shape[0] == 0:
        return weights

    if "yield_kg_per_hectare" not in df.columns:
        return weights

    y = pd.to_numeric(df["yield_kg_per_hectare"], errors="coerce").values
    y_norm = _normalize_yield(y)
    global_mean = float(np.nanmean(y_norm))
    if not np.isfinite(global_mean) or global_mean <= 0:
        return weights

    df2 = df.copy()
    df2["_y_norm"] = y_norm

    if "region" in df2.columns:
        by_region = df2.groupby("region")["_y_norm"].mean()
        for k, v in by_region.items():
            if not isinstance(k, str):
                continue
            ratio = float(v) / global_mean if np.isfinite(v) else 1.0
            weights["region"][k] = float(np.clip(ratio, 0.85, 1.15))

    if "crop_type" in df2.columns:
        by_crop = df2.groupby("crop_type")["_y_norm"].mean()
        for k, v in by_crop.items():
            if not isinstance(k, str):
                continue
            ratio = float(v) / global_mean if np.isfinite(v) else 1.0
            weights["crop_type"][k] = float(np.clip(ratio, 0.85, 1.15))

    return weights


class ContextModel:
    """
    Context model using numeric agronomy signals.
    Crop type & region are applied as calibrated rule-based weights.
    """

    def __init__(self):
        self.model = None
        self.loaded = False
        self._weights = {"region": {}, "crop_type": {}}

    # -------------------------------------------------
    # Lazy model loader (safe for Flask & Kubernetes)
    # -------------------------------------------------
    def _load(self):
        if self.loaded:
            return

        model_path = MODEL_PATH_UBJ if os.path.exists(MODEL_PATH_UBJ) else MODEL_PATH_LEGACY_PKL

        if not os.path.exists(model_path):
            # If missing locally, train a compatible model deterministically from the
            # repo dataset so region/crop context is actually used.
            with _TRAIN_LOCK:
                model_path = MODEL_PATH_UBJ if os.path.exists(MODEL_PATH_UBJ) else MODEL_PATH_LEGACY_PKL
                if not os.path.exists(model_path):
                    self._train_and_save_from_repo_dataset()
                    model_path = MODEL_PATH_UBJ if os.path.exists(MODEL_PATH_UBJ) else MODEL_PATH_LEGACY_PKL

        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.loaded = True
        if not self._weights["region"] and not self._weights["crop_type"]:
            self._load_weights_from_repo_dataset()

    def _load_weights_from_repo_dataset(self):
        try:
            ds = _smart_farming_dataset_path()
            if not os.path.exists(ds):
                return
            df = pd.read_csv(ds)
            self._weights = _compute_group_weights(df)
        except Exception as e:
            print("⚠️ Context weights load failed:", e)

    def _train_and_save_from_repo_dataset(self):
        ds = _smart_farming_dataset_path()
        if not os.path.exists(ds):
            raise RuntimeError(f"Context dataset not found at {ds}")

        df = pd.read_csv(ds)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[
            "NDVI_index",
            "temperature_C",
            "rainfall_mm",
            "humidity_%",
            "yield_kg_per_hectare",
        ])

        if df.shape[0] == 0:
            raise RuntimeError("Context dataset empty after filtering")

        disease = df.get("crop_disease_status", pd.Series(["None"] * len(df)))
        disease = disease.fillna("None").astype(str).str.strip().str.capitalize()
        disease_score = disease.map(lambda s: DISEASE_MAP.get(s, 0.0)).astype(np.float32).values

        ndvi = pd.to_numeric(df["NDVI_index"], errors="coerce").astype(np.float32).values
        temp = pd.to_numeric(df["temperature_C"], errors="coerce").astype(np.float32).values
        rain = pd.to_numeric(df["rainfall_mm"], errors="coerce").astype(np.float32).values
        humid = pd.to_numeric(df["humidity_%"], errors="coerce").astype(np.float32).values
        rain_inv = 1.0 / (rain + 1.0)

        X = np.stack([ndvi, disease_score, temp, rain_inv, humid], axis=1).astype(np.float32)
        y = pd.to_numeric(df["yield_kg_per_hectare"], errors="coerce").astype(np.float32).values
        y = _normalize_yield(y)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=200)
        model.save_model(MODEL_PATH_UBJ)

        self._weights = _compute_group_weights(df)
        print(f"✅ Trained context model and saved to {MODEL_PATH_UBJ}")

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
        region_weight = float(self._weights.get("region", {}).get(region, 1.0))

        # -------------------------
        # Crop-based weighting
        # -------------------------
        crop_weight = float(self._weights.get("crop_type", {}).get(crop_type, 1.0))

        # -------------------------
        # Final calibrated score
        # -------------------------
        final_score = base_score * region_weight * crop_weight

        return float(np.clip(final_score, 0.0, 1.0))


# -------------------------------------------------
# SAFE SINGLETON (no crash on import)
# -------------------------------------------------
context_model = ContextModel()
