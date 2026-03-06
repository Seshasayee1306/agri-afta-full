from backend.stage_engine import calculate_days_after_sowing, identify_growth_stage
from backend.stage_rf_engine import predict_stage
import os
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from prometheus_client import Counter, Histogram, generate_latest

# ✅ Existing imports (UNCHANGED)
from backend.model_loader import ModelWrapper
from backend.explain import shap_contribs, tabnet_masks, llm_explain
from backend.data_logger import append_labeled_row
from backend.utils.sensor_normalizer import normalize
# 🔹 Context model import (UNCHANGED)
from backend.context.context_model import context_model
from backend.context.context_lookup import get_context_lookup
from backend.imputation_defaults import get_median_defaults
from backend.s3_logger import make_key, put_json

# AFTA model feature order used by fed_afta/run_fed.py training.
AFTA_FEATURES_ORDER = [
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
    "potassium",
]

# Stage-personalized (local) AFTA models were trained with this order
# in backend/train_stage_afta.py and backend/evaluate_models.py.
LOCAL_STAGE_AFTA_FEATURES_ORDER = [
    "soil_moisture",
    "temperature",
    "soil_humidity",
    "air_temp",
    "air_humidity",
    "rainfall",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "hour",
    "dayofyear",
]

# AFTA calibration knobs (can be overridden via env).
# Using an uncertainty band helps reduce both false positives and false negatives
# by avoiding hard decisions near 0.5 without extra evidence.
AFTA_GLOBAL_WEIGHT = float(os.getenv("AFTA_GLOBAL_WEIGHT", "0.45"))
AFTA_LOCAL_WEIGHT = float(os.getenv("AFTA_LOCAL_WEIGHT", "0.55"))
AFTA_LOW_THRESHOLD = float(os.getenv("AFTA_LOW_THRESHOLD", "0.42"))
AFTA_HIGH_THRESHOLD = float(os.getenv("AFTA_HIGH_THRESHOLD", "0.58"))
AFTA_TIEBREAK_THRESHOLD = float(os.getenv("AFTA_TIEBREAK_THRESHOLD", "0.48"))


def _build_local_stage_features(feature_dict, hour, dayofyear):
    return [
        float(feature_dict["soil_moisture"]),
        float(feature_dict["temperature"]),
        float(feature_dict["soil_humidity"]),
        float(feature_dict["air_temp"]),
        float(feature_dict["air_humidity"]),
        float(feature_dict["rainfall"]),
        float(feature_dict["ph"]),
        float(feature_dict["nitrogen"]),
        float(feature_dict["phosphorus"]),
        float(feature_dict["potassium"]),
        float(hour),
        float(dayofyear),
    ]

# -----------------------------------------------------
# INTERNAL: FULL-INTELLIGENT COMPUTE (shared by predict + explain)
# -----------------------------------------------------
def _compute_full_intelligent_from_json(json_data):
    def _to_float(name, val):
        try:
            if val is None or (isinstance(val, str) and not val.strip()):
                return None
            return float(val)
        except Exception:
            raise ValueError(f"Invalid numeric value for '{name}'")

    sowing_date = json_data.get("sowing_date")
    current_date = json_data.get("current_date")
    if not sowing_date or not current_date:
        raise ValueError("Missing sowing_date or current_date")

    days = calculate_days_after_sowing(sowing_date, current_date)
    stage = identify_growth_stage(days)

    soil_moisture = _to_float("soil_moisture", json_data.get("soil_moisture"))
    temperature = _to_float("temperature", json_data.get("temperature"))
    humidity = _to_float("humidity", json_data.get("humidity"))
    ph = _to_float("ph", json_data.get("ph"))
    if soil_moisture is None or temperature is None or humidity is None or ph is None:
        raise ValueError("Missing required numeric inputs: soil_moisture, temperature, humidity, ph")

    region = json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
    crop_type = json_data.get("crop_type") or (json_data.get("context", {}) or {}).get("crop_type")
    soil_type = json_data.get("soil_type")
    if not region or not crop_type or soil_type is None or (isinstance(soil_type, str) and not soil_type.strip()):
        raise ValueError("Missing required context inputs: region, crop_type, soil_type")

    med = get_median_defaults()

    derived = get_context_lookup().derive(
        sowing_date=sowing_date,
        region=str(region),
        crop_type=str(crop_type),
        soil_type=str(soil_type) if soil_type is not None else None,
        soil_moisture=float(soil_moisture),
        ph=float(ph),
        temperature=float(temperature),
        humidity=float(humidity),
    )

    rainfall_override = _to_float("rainfall", json_data.get("rainfall"))
    rainfall = float(rainfall_override) if rainfall_override is not None else float(derived.rainfall)

    try:
        ndvi_val = (json_data.get("context", {}) or {}).get("ndvi", derived.ndvi)
        ndvi = float(ndvi_val)
    except Exception:
        ndvi = float(derived.ndvi)
    disease_status = (json_data.get("context", {}) or {}).get("disease_status", derived.disease_status)

    air_humidity_override = _to_float("air_humidity", json_data.get("air_humidity"))
    air_humidity = float(air_humidity_override) if air_humidity_override is not None else float(humidity)

    feature_dict = {
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "soil_humidity": float(json_data.get("soil_humidity") or med["soil_humidity"]),
        "air_temp": float(json_data.get("air_temp") or temperature),
        "air_humidity": air_humidity,
        "rainfall": float(rainfall),
        "ph": float(ph),
        "nitrogen": float(json_data.get("nitrogen") or med["nitrogen"]),
        "phosphorus": float(json_data.get("phosphorus") or med["phosphorus"]),
        "potassium": float(json_data.get("potassium") or med["potassium"]),
    }
    stage_prediction = predict_stage(stage, feature_dict)

    # AFTA model expects the same raw feature order used during training (fed_afta/run_fed.py).
    # We do NOT normalize here because training uses raw values (fillna(0)).
    sensor_features = json_data.get("sensor_features")
    if sensor_features is None:
        # Derive time features from current_date (same payload field already required)
        try:
            from datetime import datetime
            dt = datetime.strptime(current_date, "%Y-%m-%d")
            hour = float(dt.hour)  # will be 0 for date-only inputs
            dayofyear = float(dt.timetuple().tm_yday)
        except Exception:
            hour = 0.0
            dayofyear = 1.0

        sensor_features = [
            float(soil_moisture),              # soil_moisture
            float(temperature),                # temperature
            float(feature_dict["soil_humidity"]),  # soil_humidity
            float(hour),                       # hour
            float(dayofyear),                  # dayofyear
            float(feature_dict["air_temp"]),   # air_temp
            float(feature_dict["air_humidity"]),  # air_humidity
            float(rainfall),                   # rainfall
            float(ph),                         # ph
            float(feature_dict["nitrogen"]),   # nitrogen
            float(feature_dict["phosphorus"]), # phosphorus
            float(feature_dict["potassium"]),  # potassium
        ]

    X = np.asarray(sensor_features, dtype=np.float32).reshape(1, -1)
    global_afta_proba = float(model.predict_proba(X)[0])
    global_afta_prediction = int(global_afta_proba >= 0.5)

    local_model_name = f"{stage}_afta"
    local_model_available = stage in local_stage_afta_models
    local_afta_prediction = global_afta_prediction
    local_afta_proba = global_afta_proba

    if local_model_available:
        local_stage_features = _build_local_stage_features(
            feature_dict=feature_dict,
            hour=float(sensor_features[3]),
            dayofyear=float(sensor_features[4]),
        )
        X_local = np.asarray(local_stage_features, dtype=np.float32).reshape(1, -1)
        try:
            local_afta_proba = float(local_stage_afta_models[stage].predict_proba(X_local)[0])
            local_afta_prediction = int(local_afta_proba >= 0.5)
        except Exception as e:
            print(f"[AFTA] local model fallback to global for stage={stage}: {e}")
            local_afta_proba = global_afta_proba
            local_afta_prediction = global_afta_prediction
            local_model_available = False

    w_sum = max(AFTA_GLOBAL_WEIGHT + AFTA_LOCAL_WEIGHT, 1e-6)
    combined_afta_proba = (
        (AFTA_GLOBAL_WEIGHT * global_afta_proba) +
        (AFTA_LOCAL_WEIGHT * local_afta_proba)
    ) / w_sum

    temperature_ctx = float(temperature)
    rainfall_ctx = float(rainfall)
    humidity_ctx = float(air_humidity)
    try:
        context_score = float(context_model.predict_context_score(
            region=derived.matched_region,
            crop_type=derived.matched_crop_type,
            ndvi=float(ndvi),
            disease_status=disease_status,
            temperature=temperature_ctx,
            rainfall=rainfall_ctx,
            humidity=humidity_ctx
        ))
    except Exception as e:
        print("Context score unavailable:", e)
        context_score = 0.5

    # Calibrated AFTA decision:
    # - high confidence positive above upper threshold
    # - high confidence negative below lower threshold
    # - in uncertain zone, require extra support from stage/context to call positive
    afta_decision_mode = "confident_negative"
    if combined_afta_proba >= AFTA_HIGH_THRESHOLD:
        afta_prediction = 1
        afta_decision_mode = "confident_positive"
    elif combined_afta_proba <= AFTA_LOW_THRESHOLD:
        afta_prediction = 0
        afta_decision_mode = "confident_negative"
    else:
        support_votes = int(stage_prediction) + (1 if context_score >= 0.6 else 0)
        afta_prediction = 1 if (combined_afta_proba >= AFTA_TIEBREAK_THRESHOLD and support_votes >= 1) else 0
        afta_decision_mode = "uncertain_tiebreak"

    stress_index = calculate_stress_index(
        stage=stage,
        soil_moisture=feature_dict["soil_moisture"],
        temperature=feature_dict["temperature"],
        rainfall=feature_dict["rainfall"],
        ndvi=float(ndvi)
    )

    votes = [stage_prediction, afta_prediction]
    votes.append(1 if context_score >= 0.6 else 0)

    if stress_index >= 0.6:
        final_prediction = 1
    elif sum(votes) >= 2:
        final_prediction = 1
    else:
        final_prediction = 0

    if stress_index >= 0.7:
        irrigation_level = "High"
        water_liters = 25
    elif stress_index >= 0.4:
        irrigation_level = "Medium"
        water_liters = 15
    else:
        irrigation_level = "Low"
        water_liters = 5

    if final_prediction == 0:
        irrigation_level = "None"
        water_liters = 0

    response = {
        "growth_stage": stage,
        "stage_model_prediction": stage_prediction,
        "afta_prediction": afta_prediction,
        "afta_global_prediction": global_afta_prediction,
        "afta_local_prediction": local_afta_prediction,
        "afta_combined_prediction": afta_prediction,
        "afta_global_probability": round(global_afta_proba, 4),
        "afta_local_probability": round(local_afta_proba, 4),
        "afta_combined_probability": round(combined_afta_proba, 4),
        "afta_decision_mode": afta_decision_mode,
        "afta_local_model_name": local_model_name,
        "afta_local_model_available": bool(local_model_available),
        "context_score": round(float(context_score), 3),
        "final_prediction": final_prediction,
        "stress_index": stress_index,
        "irrigation_level": irrigation_level,
        "recommended_water_liters": water_liters
    }

    explain_context = {
        "sowing_date": sowing_date,
        "current_date": current_date,
        "days_after_sowing": days,
        "growth_stage": stage,
        "region": str(region),
        "crop_type": str(crop_type),
        "soil_type": str(soil_type),
        "matched_farm_id": derived.matched_farm_id,
        "matched_region": derived.matched_region,
        "matched_crop_type": derived.matched_crop_type,
        "ndvi": float(ndvi),
        "disease_status": disease_status,
        "derived_rainfall": float(rainfall),
        "feature_dict": feature_dict,
        "afta_feature_order": AFTA_FEATURES_ORDER,
        "afta_local_feature_order": LOCAL_STAGE_AFTA_FEATURES_ORDER,
        "afta_sensor_features_12": sensor_features,
        "afta_local_model_name": local_model_name,
        "afta_local_model_available": bool(local_model_available),
        "afta_global_prediction": global_afta_prediction,
        "afta_local_prediction": local_afta_prediction,
        "afta_combined_prediction": afta_prediction,
        "afta_global_probability": float(global_afta_proba),
        "afta_local_probability": float(local_afta_proba),
        "afta_combined_probability": float(combined_afta_proba),
        "afta_decision_mode": afta_decision_mode,
        "afta_thresholds": {
            "low": float(AFTA_LOW_THRESHOLD),
            "high": float(AFTA_HIGH_THRESHOLD),
            "tiebreak": float(AFTA_TIEBREAK_THRESHOLD),
        },
        "stage_model_prediction": stage_prediction,
        "afta_prediction": afta_prediction,
        "context_score": float(context_score),
        "stress_index": float(stress_index),
        "final_prediction": int(final_prediction),
        "recommended_water_liters": int(water_liters),
        "irrigation_level": irrigation_level,
        "votes": votes,
    }

    return response, explain_context


def _compute_hybrid_from_edge_afta(json_data):
    def _to_float(name, val):
        try:
            if val is None or (isinstance(val, str) and not val.strip()):
                return None
            return float(val)
        except Exception:
            raise ValueError(f"Invalid numeric value for '{name}'")

    sowing_date = json_data.get("sowing_date")
    current_date = json_data.get("current_date")
    if not sowing_date or not current_date:
        raise ValueError("Missing sowing_date or current_date")

    try:
        afta_prediction = int(json_data.get("afta_prediction"))
    except Exception:
        raise ValueError("Missing or invalid afta_prediction (expected 0 or 1)")
    if afta_prediction not in (0, 1):
        raise ValueError("Missing or invalid afta_prediction (expected 0 or 1)")

    days = calculate_days_after_sowing(sowing_date, current_date)
    stage = identify_growth_stage(days)

    soil_moisture = _to_float("soil_moisture", json_data.get("soil_moisture"))
    temperature = _to_float("temperature", json_data.get("temperature"))
    humidity = _to_float("humidity", json_data.get("humidity"))
    ph = _to_float("ph", json_data.get("ph"))
    if soil_moisture is None or temperature is None or humidity is None or ph is None:
        raise ValueError("Missing required numeric inputs: soil_moisture, temperature, humidity, ph")

    region = json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
    crop_type = json_data.get("crop_type") or (json_data.get("context", {}) or {}).get("crop_type")
    soil_type = json_data.get("soil_type")
    if not region or not crop_type or soil_type is None or (isinstance(soil_type, str) and not soil_type.strip()):
        raise ValueError("Missing required context inputs: region, crop_type, soil_type")

    med = get_median_defaults()
    derived = get_context_lookup().derive(
        sowing_date=sowing_date,
        region=str(region),
        crop_type=str(crop_type),
        soil_type=str(soil_type) if soil_type is not None else None,
        soil_moisture=float(soil_moisture),
        ph=float(ph),
        temperature=float(temperature),
        humidity=float(humidity),
    )

    rainfall_override = _to_float("rainfall", json_data.get("rainfall"))
    rainfall = float(rainfall_override) if rainfall_override is not None else float(derived.rainfall)

    try:
        ndvi_val = (json_data.get("context", {}) or {}).get("ndvi", derived.ndvi)
        ndvi = float(ndvi_val)
    except Exception:
        ndvi = float(derived.ndvi)
    disease_status = (json_data.get("context", {}) or {}).get("disease_status", derived.disease_status)

    air_humidity_override = _to_float("air_humidity", json_data.get("air_humidity"))
    air_humidity = float(air_humidity_override) if air_humidity_override is not None else float(humidity)

    feature_dict = {
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "soil_humidity": float(json_data.get("soil_humidity") or med["soil_humidity"]),
        "air_temp": float(json_data.get("air_temp") or temperature),
        "air_humidity": air_humidity,
        "rainfall": float(rainfall),
        "ph": float(ph),
        "nitrogen": float(json_data.get("nitrogen") or med["nitrogen"]),
        "phosphorus": float(json_data.get("phosphorus") or med["phosphorus"]),
        "potassium": float(json_data.get("potassium") or med["potassium"]),
    }
    stage_prediction = int(predict_stage(stage, feature_dict))

    temperature_ctx = float(temperature)
    rainfall_ctx = float(rainfall)
    humidity_ctx = float(air_humidity)
    try:
        context_score = float(context_model.predict_context_score(
            region=derived.matched_region,
            crop_type=derived.matched_crop_type,
            ndvi=float(ndvi),
            disease_status=disease_status,
            temperature=temperature_ctx,
            rainfall=rainfall_ctx,
            humidity=humidity_ctx
        ))
    except Exception as e:
        print("Context score unavailable:", e)
        context_score = 0.5

    stress_index = calculate_stress_index(
        stage=stage,
        soil_moisture=feature_dict["soil_moisture"],
        temperature=feature_dict["temperature"],
        rainfall=feature_dict["rainfall"],
        ndvi=float(ndvi)
    )

    votes = [stage_prediction, afta_prediction]
    votes.append(1 if context_score >= 0.6 else 0)

    if stress_index >= 0.6:
        final_prediction = 1
    elif sum(votes) >= 2:
        final_prediction = 1
    else:
        final_prediction = 0

    if stress_index >= 0.7:
        irrigation_level = "High"
        water_liters = 25
    elif stress_index >= 0.4:
        irrigation_level = "Medium"
        water_liters = 15
    else:
        irrigation_level = "Low"
        water_liters = 5

    if final_prediction == 0:
        irrigation_level = "None"
        water_liters = 0

    response = {
        "growth_stage": stage,
        "stage_model_prediction": stage_prediction,
        "afta_prediction": int(afta_prediction),
        "context_score": round(float(context_score), 3),
        "final_prediction": int(final_prediction),
        "stress_index": stress_index,
        "irrigation_level": irrigation_level,
        "recommended_water_liters": int(water_liters),
        "inference_source": "edge_afta_plus_backend_stage"
    }
    return response


def _log_predict_to_s3(payload, response, explain_context):
    try:
        # keep logs compact + deterministic; do not block prediction on S3 failure
        key = make_key("predictions", "json")
        ok = put_json(
            key=key,
            payload={
                "kind": "prediction",
                "request_payload": payload,
                "response": response,
                "explain_context": explain_context,
            },
        )
        if ok and os.getenv("S3_DEBUG") == "1":
            print("✅ S3 prediction uploaded:", key)
    except Exception as e:
        print("⚠️ S3 prediction logging failed:", e)


def _log_label_to_s3(labeled_row):
    try:
        key = make_key("labeled", "json")
        ok = put_json(key=key, payload={"kind": "label", **labeled_row})
        if ok and os.getenv("S3_DEBUG") == "1":
            print("✅ S3 label uploaded:", key)
    except Exception as e:
        print("⚠️ S3 label logging failed:", e)

# =====================================================
# STRESS INDEX CALCULATION
# =====================================================
def calculate_stress_index(stage, soil_moisture, temperature, rainfall, ndvi=0.5):

    soil_factor = 1 - (soil_moisture / 100)
    temp_factor = min(max((temperature - 20) / 20, 0), 1)
    rain_factor = 1 - min(rainfall / 200, 1)
    ndvi_factor = 1 - ndvi

    stage_weights = {
        "germination": 1.2,
        "vegetative": 1.0,
        "flowering": 1.3,
        "harvest": 0.8
    }

    stage_weight = stage_weights.get(stage, 1.0)

    stress = (soil_factor * 0.4 +
              temp_factor * 0.2 +
              rain_factor * 0.2 +
              ndvi_factor * 0.2)

    stress = stress * stage_weight

    return round(min(stress, 1.0), 3)

# -----------------------------------------------------
# FLASK APP
# -----------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer a PVC-synced model if present (Kubernetes), else fall back to the repo artifact.
MODEL_PATH = (
    os.getenv("AFTA_MODEL_PATH")
    or ("/models/final_model.pkl" if os.path.exists("/models/final_model.pkl") else None)
    or os.path.join(BASE_DIR, "final_model.pkl")
)
model = ModelWrapper(MODEL_PATH)

LOCAL_STAGE_AFTA_DIR = os.path.join(BASE_DIR, "stage_afta_models")


def _load_local_stage_afta_models():
    models = {}
    for stage in ("germination", "vegetative", "flowering", "harvest"):
        path = os.path.join(LOCAL_STAGE_AFTA_DIR, f"{stage}_afta.pkl")
        if not os.path.exists(path):
            continue
        try:
            models[stage] = ModelWrapper(path)
        except Exception as e:
            print(f"[AFTA] failed to load local model {path}: {e}")
    return models


local_stage_afta_models = _load_local_stage_afta_models()
print(
    "[AFTA] loaded local stage models:",
    sorted(local_stage_afta_models.keys()) if local_stage_afta_models else "none",
)

# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend is running"})

# -----------------------------------------------------
# PREDICT
# -----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.get_json(silent=True)
    data = json_data.get("features") if json_data else None

    if data is None:
        return jsonify({"error": "No input features provided"}), 400

    feature_mins = np.array([0,0,0,0,1,0,0,0,0,0,0,0], dtype=np.float32)
    feature_maxs = np.array([100,50,100,23,365,50,100,50,14,100,50,100], dtype=np.float32)

    X_norm = np.array(data, dtype=np.float32).reshape(1, -1)
    X_scaled = X_norm * (feature_maxs - feature_mins) + feature_mins

    pred = model.predict(X_scaled)

    return jsonify({"prediction": int(pred)})

# -----------------------------------------------------
# EXPLAIN
# -----------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    def _to_float(name, val):
        try:
            if val is None or (isinstance(val, str) and not val.strip()):
                return None
            return float(val)
        except Exception:
            raise ValueError(f"Invalid numeric value for '{name}'")

    feature_names = json_data.get("feature_names")

    # Backwards compatible:
    # - old clients send {"features": [...]} (either already-normalized or raw)
    # - new clients send minimal fields and we deterministically build features
    features = json_data.get("features")
    raw_row = None
    if features is not None:
        try:
            features = [float(x) for x in features]
        except Exception:
            return jsonify({"error": "Invalid 'features' array"}), 400

        if feature_names and len(feature_names) == len(features):
            raw_row = {feature_names[i]: features[i] for i in range(len(features))}
        else:
            raw_row = {f"f{i}": features[i] for i in range(len(features))}

        # Treat provided features as raw AFTA features by default (matches fed_afta/run_fed.py training).
        X = np.asarray(features, dtype=np.float32).reshape(1, -1)
        used_feature_names = feature_names or AFTA_FEATURES_ORDER
    else:
        # If the client provides the same payload as /predict_full_intelligent, explain the
        # full intelligent decision (so explanation matches the prediction page).
        if json_data.get("current_date") and (
            json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
        ):
            try:
                full_response, explain_context = _compute_full_intelligent_from_json(json_data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            # Use the same 12 AFTA features for embeddings/SHAP placeholders, but explain the final decision.
            features = explain_context.get("afta_sensor_features_12")
            used_feature_names = AFTA_FEATURES_ORDER
            raw_row = dict(explain_context)
            raw_row["system_response"] = full_response
            X = np.asarray(features, dtype=np.float32).reshape(1, -1)

            pred = int(full_response["final_prediction"])
            prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

            emb, proba, _ = model.get_embeddings_and_pred(X)
            shap_vals = np.zeros((1, len(features)), dtype=np.float32)
            masks = np.zeros((len(features),), dtype=np.float32)

            explanation = llm_explain(
                raw_row=raw_row,
                shap_vals=shap_vals[0],
                masks=masks,
                pred=pred
            )

            return jsonify({
                "prediction": int(pred),
                "prediction_text": prediction_text,
                "probability": float(proba),
                "feature_names": used_feature_names,
                "features": [float(x) for x in features],
                "shap_values": shap_vals[0].tolist(),
                "tabnet_masks": masks.tolist(),
                "llm_explanation": explanation,
                "system_response": full_response,
            })

        # New minimal-input explain flow: build the same AFTA 12-feature vector as predict_full_intelligent,
        # and enrich raw_row with context (region/crop/soil), season, and optional stage info.
        sowing_date = json_data.get("sowing_date")
        current_date = json_data.get("current_date")
        region = json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
        crop_type = json_data.get("crop_type") or (json_data.get("context", {}) or {}).get("crop_type")
        soil_type = json_data.get("soil_type")

        if not sowing_date or not region or not crop_type or soil_type is None or (isinstance(soil_type, str) and not soil_type.strip()):
            return jsonify({
                "error": "Missing required fields for explanation",
                "required": ["sowing_date", "region", "crop_type", "soil_type"]
            }), 400

        try:
            soil_moisture = _to_float("soil_moisture", json_data.get("soil_moisture"))
            temperature = _to_float("temperature", json_data.get("temperature"))
            humidity = _to_float("humidity", json_data.get("humidity"))
            ph = _to_float("ph", json_data.get("ph"))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if soil_moisture is None or temperature is None or humidity is None or ph is None:
            return jsonify({
                "error": "Missing required fields for explanation",
                "required": ["soil_moisture", "temperature", "humidity", "ph"]
            }), 400

        med = get_median_defaults()

        # Optional overrides (if provided, we use them instead of medians/derived context).
        try:
            rainfall_override = _to_float("rainfall", json_data.get("rainfall"))
            soil_humidity_override = _to_float("soil_humidity", json_data.get("soil_humidity"))
            air_temp_override = _to_float("air_temp", json_data.get("air_temp"))
            air_humidity_override = _to_float("air_humidity", json_data.get("air_humidity"))
            wind_speed_override = _to_float("wind_speed", json_data.get("wind_speed"))
            wind_gust_override = _to_float("wind_gust", json_data.get("wind_gust"))
            pressure_override = _to_float("pressure_kpa", json_data.get("pressure_kpa"))
            nitrogen_override = _to_float("nitrogen", json_data.get("nitrogen"))
            phosphorus_override = _to_float("phosphorus", json_data.get("phosphorus"))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        derived = get_context_lookup().derive(
            sowing_date=sowing_date,
            region=str(region),
            crop_type=str(crop_type),
            soil_type=str(soil_type) if soil_type is not None else None,
            soil_moisture=float(soil_moisture),
            ph=float(ph),
            temperature=float(temperature),
            humidity=float(humidity),
        )

        rainfall = float(rainfall_override) if rainfall_override is not None else float(derived.rainfall)
        try:
            ndvi_val = (json_data.get("context", {}) or {}).get("ndvi", derived.ndvi)
            ndvi = float(ndvi_val)
        except Exception:
            ndvi = float(derived.ndvi)
        disease_status = (json_data.get("context", {}) or {}).get("disease_status", derived.disease_status)

        soil_humidity = float(soil_humidity_override) if soil_humidity_override is not None else float(med["soil_humidity"])
        air_temp = float(air_temp_override) if air_temp_override is not None else float(temperature)
        air_humidity = float(air_humidity_override) if air_humidity_override is not None else float(humidity)
        wind_speed = float(wind_speed_override) if wind_speed_override is not None else float(med["wind_speed"])
        wind_gust = float(wind_gust_override) if wind_gust_override is not None else float(med["wind_gust"])
        pressure_kpa = float(pressure_override) if pressure_override is not None else float(med["pressure_kpa"])
        nitrogen = float(nitrogen_override) if nitrogen_override is not None else float(med["nitrogen"])
        phosphorus = float(phosphorus_override) if phosphorus_override is not None else float(med["phosphorus"])

        # Keep AFTA features consistent with the trained order.
        # Derive hour/dayofyear from current_date if available; else use safe defaults.
        try:
            from datetime import datetime
            dt = datetime.strptime(current_date, "%Y-%m-%d") if current_date else None
            hour = float(dt.hour) if dt else 0.0
            dayofyear = float(dt.timetuple().tm_yday) if dt else 1.0
        except Exception:
            hour = 0.0
            dayofyear = 1.0

        features = [
            float(soil_moisture),
            float(temperature),
            float(soil_humidity),
            float(hour),
            float(dayofyear),
            float(air_temp),
            float(air_humidity),
            float(rainfall),
            float(ph),
            float(nitrogen),
            float(phosphorus),
            float(med["potassium"]),
        ]
        used_feature_names = AFTA_FEATURES_ORDER
        raw_row = {used_feature_names[i]: features[i] for i in range(len(features))}

        growth_stage = None
        days_after_sowing = None
        if current_date:
            try:
                days_after_sowing = calculate_days_after_sowing(sowing_date, current_date)
                growth_stage = identify_growth_stage(days_after_sowing)
            except Exception:
                days_after_sowing = None
                growth_stage = None

        # Compute context score so LLM can explain how region/crop/ndvi/disease/rainfall affect risk.
        try:
            context_score = float(context_model.predict_context_score(
                region=derived.matched_region,
                crop_type=derived.matched_crop_type,
                ndvi=float(ndvi),
                disease_status=disease_status,
                temperature=float(temperature),
                rainfall=float(rainfall),
                humidity=float(air_humidity),
            ))
        except Exception as e:
            print("Context score unavailable in explain:", e)
            context_score = None

        raw_row.update({
            "sowing_date": sowing_date,
            "current_date": current_date,
            "days_after_sowing": days_after_sowing,
            "growth_stage": growth_stage,
            "region": str(region),
            "crop_type": str(crop_type),
            "soil_type": str(soil_type) if soil_type is not None else None,
            "derived_rainfall_source": "smart_farming_best_row",
            "matched_farm_id": derived.matched_farm_id,
            "matched_region": derived.matched_region,
            "matched_crop_type": derived.matched_crop_type,
            "ndvi": float(ndvi),
            "disease_status": disease_status,
            "context_score": context_score,
        })
        X = np.asarray(features, dtype=np.float32).reshape(1, -1)

    pred = int(model.predict(X))
    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    emb, proba, _ = model.get_embeddings_and_pred(X)
    # Production-safe fallback explanations: keep lengths aligned to input features.
    shap_vals = np.zeros((1, len(features)), dtype=np.float32)
    masks = np.zeros((len(features),), dtype=np.float32)

    explanation = llm_explain(
        raw_row=raw_row,
        shap_vals=shap_vals[0],
        masks=masks,
        pred=pred
    )

    return jsonify({
        "prediction": int(pred),
        "prediction_text": prediction_text,
        "probability": float(proba),
        "feature_names": used_feature_names,
        "features": [float(x) for x in features],
        "shap_values": shap_vals[0].tolist(),
        "tabnet_masks": masks.tolist(),
        "llm_explanation": explanation
    })

# -----------------------------------------------------
# LABEL DATA
# -----------------------------------------------------
@app.route("/label", methods=["POST"])
def label_data():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    label = data.get("label")
    if label is None:
        return jsonify({"error": "Missing label"}), 400

    try:
        label_int = int(label)
    except Exception:
        return jsonify({"error": "Label must be 0 or 1"}), 400

    if label_int not in (0, 1):
        return jsonify({"error": "Label must be 0 or 1"}), 400

    # Accept either:
    # - legacy: {"features":[..12..], "label":0/1}
    # - new: same payload as /predict_full_intelligent + "label"
    features = data.get("features")
    if features is not None:
        if not isinstance(features, list) or len(features) != 12:
            return jsonify({"error": "Expected exactly 12 features"}), 400
        try:
            features = [float(x) for x in features]
        except Exception:
            return jsonify({"error": "Invalid features array"}), 400
        append_labeled_row(features, label_int)
        _log_label_to_s3({
            "features_order": AFTA_FEATURES_ORDER,
            "features": features,
            "label": label_int,
            "source": "client_features",
        })
        return jsonify({"status": "Labeled data appended successfully"})

    # New flow: build the exact AFTA features used in prediction and attach label.
    try:
        _, explain_context = _compute_full_intelligent_from_json(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    afta_features = explain_context.get("afta_sensor_features_12")
    if not afta_features or len(afta_features) != 12:
        return jsonify({"error": "Unable to build AFTA features for labeling"}), 400

    append_labeled_row(afta_features, label_int)
    _log_label_to_s3({
        "features_order": AFTA_FEATURES_ORDER,
        "features": afta_features,
        "label": label_int,
        "source": "derived_from_payload",
        "context": {
            "region": explain_context.get("region"),
            "crop_type": explain_context.get("crop_type"),
            "soil_type": explain_context.get("soil_type"),
            "growth_stage": explain_context.get("growth_stage"),
        },
    })
    return jsonify({"status": "Labeled data appended successfully"})

# -----------------------------------------------------
# PROMETHEUS METRICS
# -----------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

@app.after_request
def after_request(response):
    REQUEST_COUNT.labels(
        request.method,
        request.path,
        response.status_code
    ).inc()
    return response

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# -----------------------------------------------------
# HEALTH
# -----------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "backend running"})

# =====================================================
# NEW: STAGE-AWARE PREDICTION
# =====================================================
@app.route("/predict_stage_aware", methods=["POST"])
def predict_stage_aware():

    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    sowing_date = json_data.get("sowing_date")
    current_date = json_data.get("current_date")

    if not sowing_date or not current_date:
        return jsonify({"error": "Missing sowing_date or current_date"}), 400

    days = calculate_days_after_sowing(sowing_date, current_date)
    stage = identify_growth_stage(days)

    feature_dict = {
        "soil_moisture": json_data.get("soil_moisture"),
        "temperature": json_data.get("temperature"),
        "soil_humidity": json_data.get("soil_humidity"),
        "air_temp": json_data.get("air_temp"),
        "air_humidity": json_data.get("air_humidity"),
        "rainfall": json_data.get("rainfall"),
        "ph": json_data.get("ph"),
        "nitrogen": json_data.get("nitrogen"),
        "phosphorus": json_data.get("phosphorus"),
        "potassium": json_data.get("potassium")
    }

    stage_prediction = predict_stage(stage, feature_dict)

    return jsonify({
        "growth_stage": stage,
        "needs_water_prediction": stage_prediction
    })

# =====================================================
# CONTEXT-AWARE PREDICTION
# =====================================================
@app.route("/predict_with_context", methods=["POST"])
def predict_with_context():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    sensor_features = json_data.get("features")
    if sensor_features is None:
        return jsonify({"error": "Missing sensor features"}), 400

    if len(sensor_features) != 12:
        return jsonify({
            "error": "Expected exactly 12 sensor features",
            "received": len(sensor_features)
        }), 400

    X = normalize(sensor_features).reshape(1, -1)
    sensor_prediction = int(model.predict(X))

    context = json_data.get("context", {})

    region = context.get("region", "Unknown")
    crop_type = context.get("crop_type", "Unknown")
    ndvi = float(context.get("ndvi", 0.5))
    disease_status = context.get("disease_status", "None")
    temperature = float(context.get("temperature", 25))
    rainfall = float(context.get("rainfall", 100))
    humidity = float(context.get("humidity", 60))

    context_score = context_model.predict_context_score(
        region=region,
        crop_type=crop_type,
        ndvi=ndvi,
        disease_status=disease_status,
        temperature=temperature,
        rainfall=rainfall,
        humidity=humidity
    )

    if context_score < 0.3:
        final_prediction = 0
    elif sensor_prediction == 1 or context_score >= 0.6:
        final_prediction = 1
    else:
        final_prediction = 0

    return jsonify({
        "sensor_prediction": sensor_prediction,
        "context_score": round(float(context_score), 3),
        "final_prediction": final_prediction,
        "decision_reason": (
            "Sensor-based irrigation need"
            if sensor_prediction == 1
            else "Context-driven irrigation risk"
            if context_score >= 0.6
            else "No irrigation required"
        )
    })

# =====================================================
# FULL INTELLIGENT STAGE + AFTA + CONTEXT
# =====================================================
@app.route("/predict_full_intelligent", methods=["POST"])
def predict_full_intelligent():

    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        response, explain_context = _compute_full_intelligent_from_json(json_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Terminal logs: show all filled values + sources (not returned to frontend)
    print("\n[Full Intelligent Prediction] Filled inputs")
    # explain_context is already a fully expanded record; keep logs stable and readable.
    print(explain_context)
    print(" fixed_medians_source: irrigation_dataset/irrigation_stage_dataset\n")

    _log_predict_to_s3(json_data, response, explain_context)

    return jsonify(response)


@app.route("/predict_edge_afta", methods=["POST"])
def predict_edge_afta():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        response = _compute_hybrid_from_edge_afta(json_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(response)

from flask import send_from_directory, render_template

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
