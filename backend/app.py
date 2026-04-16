from backend.stage_engine import calculate_days_after_sowing, identify_growth_stage
from backend.stage_rf_engine import predict_stage
import os
import numpy as np
import json
import joblib
import pandas as pd
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
from datetime import datetime, timezone

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
CHALLENGER_VALIDATION_AUC = float(os.getenv("CHALLENGER_VALIDATION_AUC", "0.84"))
CHALLENGER_MODEL_KIND_ENV = os.getenv("AFTA_CHALLENGER_KIND")
CHALLENGER_MODEL_KIND = (CHALLENGER_MODEL_KIND_ENV or "").strip().lower()
CHALLENGER_MODEL_FILES = {
    "random_forest": "random_forest_model.pkl",
    "xgboost": "xgboost_model.pkl",
    "catboost": "catboost_model.pkl",
}
CHALLENGER_XGB_RF_ENSEMBLE_KIND = "xgb_rf_ensemble"
CHALLENGER_MODEL_LABELS = {
    "random_forest": "RandomForest Challenger Model",
    "xgboost": "XGBoost Challenger Model",
    "catboost": "CatBoost Challenger Model",
    CHALLENGER_XGB_RF_ENSEMBLE_KIND: "XGBoost + RandomForest Ensemble",
}
CHALLENGER_VIRTUAL_MODEL_KINDS = {CHALLENGER_XGB_RF_ENSEMBLE_KIND}
CHALLENGER_SUPPORTED_MODEL_KINDS = set(CHALLENGER_MODEL_FILES.keys()) | CHALLENGER_VIRTUAL_MODEL_KINDS
CHALLENGER_KIND_TIEBREAK_PRIORITY = {
    # Keep XGBoost preference during ties for stability with existing deployments.
    "xgboost": 3,
    "catboost": 2,
    "random_forest": 1,
}

# Validation guardrails for real-world input quality.
STRICT_CONTEXT_MATCH = os.getenv("AFTA_STRICT_CONTEXT_MATCH", "1").strip().lower() not in ("0", "false", "no")
MAX_DAYS_AFTER_SOWING = int(os.getenv("AFTA_MAX_DAYS_AFTER_SOWING", "3650"))

CORE_INPUT_RANGES = {
    "soil_moisture": (0.0, 100.0),
    "temperature": (-20.0, 70.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "soil_humidity": (0.0, 100.0),
    "air_temp": (-20.0, 70.0),
    "air_humidity": (0.0, 100.0),
    "rainfall": (0.0, 500.0),
    "nitrogen": (0.0, 300.0),
    "phosphorus": (0.0, 300.0),
    "potassium": (0.0, 300.0),
    "ndvi": (-1.0, 1.0),
}

SENSOR_FEATURE_RANGES = [
    ("soil_moisture", 0.0, 100.0),
    ("temperature", -20.0, 70.0),
    ("soil_humidity", 0.0, 100.0),
    ("hour", 0.0, 23.0),
    ("dayofyear", 1.0, 366.0),
    ("air_temp", -20.0, 70.0),
    ("air_humidity", 0.0, 100.0),
    ("rainfall", 0.0, 500.0),
    ("ph", 0.0, 14.0),
    ("nitrogen", 0.0, 300.0),
    ("phosphorus", 0.0, 300.0),
    ("potassium", 0.0, 300.0),
]


def _ensure_in_range(name, value, lo, hi):
    if value is None:
        return
    if not np.isfinite(value):
        raise ValueError(f"Invalid numeric value for '{name}'")
    if value < lo or value > hi:
        raise ValueError(f"'{name}' out of range [{lo}, {hi}]")


def _validate_required_text(name, value):
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required context input: {name}")
    if text.lower() in ("unknown", "none", "na", "n/a"):
        raise ValueError(f"'{name}' cannot be placeholder text")
    return text


def _validate_dates_or_raise(sowing_date, current_date):
    try:
        sowing_dt = datetime.strptime(str(sowing_date), "%Y-%m-%d")
        current_dt = datetime.strptime(str(current_date), "%Y-%m-%d")
    except Exception:
        raise ValueError("Invalid date format. Use YYYY-MM-DD for sowing_date and current_date")

    if current_dt < sowing_dt:
        raise ValueError("current_date must be on or after sowing_date")
    days_after_sowing = (current_dt - sowing_dt).days
    if days_after_sowing > MAX_DAYS_AFTER_SOWING:
        raise ValueError(f"days after sowing is too large (>{MAX_DAYS_AFTER_SOWING})")
    return days_after_sowing


def _validate_sensor_features(sensor_features):
    if len(sensor_features) != 12:
        raise ValueError(f"Expected exactly 12 sensor features, received {len(sensor_features)}")
    for idx, (name, lo, hi) in enumerate(SENSOR_FEATURE_RANGES):
        _ensure_in_range(name, float(sensor_features[idx]), lo, hi)


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
    _validate_dates_or_raise(sowing_date, current_date)

    days = calculate_days_after_sowing(sowing_date, current_date)
    stage = identify_growth_stage(days)

    soil_moisture = _to_float("soil_moisture", json_data.get("soil_moisture"))
    temperature = _to_float("temperature", json_data.get("temperature"))
    humidity = _to_float("humidity", json_data.get("humidity"))
    ph = _to_float("ph", json_data.get("ph"))
    if soil_moisture is None or temperature is None or humidity is None or ph is None:
        raise ValueError("Missing required numeric inputs: soil_moisture, temperature, humidity, ph")
    _ensure_in_range("soil_moisture", float(soil_moisture), *CORE_INPUT_RANGES["soil_moisture"])
    _ensure_in_range("temperature", float(temperature), *CORE_INPUT_RANGES["temperature"])
    _ensure_in_range("humidity", float(humidity), *CORE_INPUT_RANGES["humidity"])
    _ensure_in_range("ph", float(ph), *CORE_INPUT_RANGES["ph"])

    region = json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
    crop_type = json_data.get("crop_type") or (json_data.get("context", {}) or {}).get("crop_type")
    soil_type = json_data.get("soil_type")
    if not region or not crop_type or soil_type is None or (isinstance(soil_type, str) and not soil_type.strip()):
        raise ValueError("Missing required context inputs: region, crop_type, soil_type")
    region = _validate_required_text("region", region)
    crop_type = _validate_required_text("crop_type", crop_type)
    soil_type = _validate_required_text("soil_type", soil_type)

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
    _ensure_in_range("rainfall", rainfall_override, *CORE_INPUT_RANGES["rainfall"])
    rainfall = float(rainfall_override) if rainfall_override is not None else float(derived.rainfall)
    _ensure_in_range("rainfall", rainfall, *CORE_INPUT_RANGES["rainfall"])

    try:
        ndvi_val = (json_data.get("context", {}) or {}).get("ndvi", derived.ndvi)
        ndvi = float(ndvi_val)
    except Exception:
        ndvi = float(derived.ndvi)
    _ensure_in_range("ndvi", ndvi, *CORE_INPUT_RANGES["ndvi"])
    disease_status = (json_data.get("context", {}) or {}).get("disease_status", derived.disease_status)

    air_humidity_override = _to_float("air_humidity", json_data.get("air_humidity"))
    _ensure_in_range("air_humidity", air_humidity_override, *CORE_INPUT_RANGES["air_humidity"])
    air_humidity = float(air_humidity_override) if air_humidity_override is not None else float(humidity)
    _ensure_in_range("air_humidity", air_humidity, *CORE_INPUT_RANGES["air_humidity"])

    soil_humidity_override = _to_float("soil_humidity", json_data.get("soil_humidity"))
    air_temp_override = _to_float("air_temp", json_data.get("air_temp"))
    nitrogen_override = _to_float("nitrogen", json_data.get("nitrogen"))
    phosphorus_override = _to_float("phosphorus", json_data.get("phosphorus"))
    potassium_override = _to_float("potassium", json_data.get("potassium"))
    _ensure_in_range("soil_humidity", soil_humidity_override, *CORE_INPUT_RANGES["soil_humidity"])
    _ensure_in_range("air_temp", air_temp_override, *CORE_INPUT_RANGES["air_temp"])
    _ensure_in_range("nitrogen", nitrogen_override, *CORE_INPUT_RANGES["nitrogen"])
    _ensure_in_range("phosphorus", phosphorus_override, *CORE_INPUT_RANGES["phosphorus"])
    _ensure_in_range("potassium", potassium_override, *CORE_INPUT_RANGES["potassium"])

    if STRICT_CONTEXT_MATCH:
        if derived.match_notes.get("region_match") == "unknown":
            raise ValueError("Unknown region. Use a valid region name from training context")
        if derived.match_notes.get("crop_match") == "unknown":
            raise ValueError("Unknown crop_type. Use a valid crop type from training context")

    feature_dict = {
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "soil_humidity": float(soil_humidity_override) if soil_humidity_override is not None else float(med["soil_humidity"]),
        "air_temp": float(air_temp_override) if air_temp_override is not None else float(temperature),
        "air_humidity": air_humidity,
        "rainfall": float(rainfall),
        "ph": float(ph),
        "nitrogen": float(nitrogen_override) if nitrogen_override is not None else float(med["nitrogen"]),
        "phosphorus": float(phosphorus_override) if phosphorus_override is not None else float(med["phosphorus"]),
        "potassium": float(potassium_override) if potassium_override is not None else float(med["potassium"]),
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

    _validate_sensor_features([float(x) for x in sensor_features])

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
    _validate_dates_or_raise(sowing_date, current_date)

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
    _ensure_in_range("soil_moisture", float(soil_moisture), *CORE_INPUT_RANGES["soil_moisture"])
    _ensure_in_range("temperature", float(temperature), *CORE_INPUT_RANGES["temperature"])
    _ensure_in_range("humidity", float(humidity), *CORE_INPUT_RANGES["humidity"])
    _ensure_in_range("ph", float(ph), *CORE_INPUT_RANGES["ph"])

    region = json_data.get("region") or (json_data.get("context", {}) or {}).get("region")
    crop_type = json_data.get("crop_type") or (json_data.get("context", {}) or {}).get("crop_type")
    soil_type = json_data.get("soil_type")
    if not region or not crop_type or soil_type is None or (isinstance(soil_type, str) and not soil_type.strip()):
        raise ValueError("Missing required context inputs: region, crop_type, soil_type")
    region = _validate_required_text("region", region)
    crop_type = _validate_required_text("crop_type", crop_type)
    soil_type = _validate_required_text("soil_type", soil_type)

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
    _ensure_in_range("rainfall", rainfall_override, *CORE_INPUT_RANGES["rainfall"])
    rainfall = float(rainfall_override) if rainfall_override is not None else float(derived.rainfall)
    _ensure_in_range("rainfall", rainfall, *CORE_INPUT_RANGES["rainfall"])

    try:
        ndvi_val = (json_data.get("context", {}) or {}).get("ndvi", derived.ndvi)
        ndvi = float(ndvi_val)
    except Exception:
        ndvi = float(derived.ndvi)
    _ensure_in_range("ndvi", ndvi, *CORE_INPUT_RANGES["ndvi"])
    disease_status = (json_data.get("context", {}) or {}).get("disease_status", derived.disease_status)

    air_humidity_override = _to_float("air_humidity", json_data.get("air_humidity"))
    _ensure_in_range("air_humidity", air_humidity_override, *CORE_INPUT_RANGES["air_humidity"])
    air_humidity = float(air_humidity_override) if air_humidity_override is not None else float(humidity)
    _ensure_in_range("air_humidity", air_humidity, *CORE_INPUT_RANGES["air_humidity"])

    soil_humidity_override = _to_float("soil_humidity", json_data.get("soil_humidity"))
    air_temp_override = _to_float("air_temp", json_data.get("air_temp"))
    nitrogen_override = _to_float("nitrogen", json_data.get("nitrogen"))
    phosphorus_override = _to_float("phosphorus", json_data.get("phosphorus"))
    potassium_override = _to_float("potassium", json_data.get("potassium"))
    _ensure_in_range("soil_humidity", soil_humidity_override, *CORE_INPUT_RANGES["soil_humidity"])
    _ensure_in_range("air_temp", air_temp_override, *CORE_INPUT_RANGES["air_temp"])
    _ensure_in_range("nitrogen", nitrogen_override, *CORE_INPUT_RANGES["nitrogen"])
    _ensure_in_range("phosphorus", phosphorus_override, *CORE_INPUT_RANGES["phosphorus"])
    _ensure_in_range("potassium", potassium_override, *CORE_INPUT_RANGES["potassium"])

    if STRICT_CONTEXT_MATCH:
        if derived.match_notes.get("region_match") == "unknown":
            raise ValueError("Unknown region. Use a valid region name from training context")
        if derived.match_notes.get("crop_match") == "unknown":
            raise ValueError("Unknown crop_type. Use a valid crop type from training context")

    feature_dict = {
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "soil_humidity": float(soil_humidity_override) if soil_humidity_override is not None else float(med["soil_humidity"]),
        "air_temp": float(air_temp_override) if air_temp_override is not None else float(temperature),
        "air_humidity": air_humidity,
        "rainfall": float(rainfall),
        "ph": float(ph),
        "nitrogen": float(nitrogen_override) if nitrogen_override is not None else float(med["nitrogen"]),
        "phosphorus": float(phosphorus_override) if phosphorus_override is not None else float(med["phosphorus"]),
        "potassium": float(potassium_override) if potassium_override is not None else float(med["potassium"]),
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

latest_sensor_readings = None

ESP32_ADC_MAX = 4095.0


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _adc_to_ph(raw_value):
    # ESP32 ADC range (0..4095) mapped to pH scale (0..14)
    ph_value = (float(raw_value) * 14.0) / ESP32_ADC_MAX
    return _clamp(ph_value, 0.0, 14.0)


def _adc_to_soil_moisture_percent(raw_value):
    # Typical capacitive probes: higher ADC means drier soil.
    # Map 0..4095 to 100..0 moisture%.
    pct = (1.0 - (float(raw_value) / ESP32_ADC_MAX)) * 100.0
    return _clamp(pct, 0.0, 100.0)

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

# Preferred challenger: a different model family artifact from challenger_models/.
CHALLENGER_MODELS_DIR = (
    os.getenv("AFTA_CHALLENGER_MODELS_DIR")
    or ("/models/challenger_models" if os.path.exists("/models/challenger_models") else None)
    or os.path.join(BASE_DIR, "challenger_models")
)

CHALLENGER_METRICS_PATH = os.path.join(CHALLENGER_MODELS_DIR, "metrics.json")
challenger_metrics = {}
if os.path.exists(CHALLENGER_METRICS_PATH):
    try:
        with open(CHALLENGER_METRICS_PATH, "r", encoding="utf-8") as f:
            loaded_metrics = json.load(f)
            if isinstance(loaded_metrics, dict):
                challenger_metrics = loaded_metrics
    except Exception as e:
        print("[AFTA] challenger metrics load failed:", e)

if CHALLENGER_MODEL_KIND and CHALLENGER_MODEL_KIND not in CHALLENGER_SUPPORTED_MODEL_KINDS:
    print(f"[AFTA] invalid challenger kind '{CHALLENGER_MODEL_KIND}', ignoring override and auto-selecting")
    CHALLENGER_MODEL_KIND = ""


def _load_challenger_artifacts():
    artifacts = {}
    for kind, fname in CHALLENGER_MODEL_FILES.items():
        path = os.path.join(CHALLENGER_MODELS_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            artifacts[kind] = joblib.load(path)
            print("[AFTA] loaded challenger artifact:", kind, path)
        except Exception as e:
            print(f"[AFTA] challenger artifact load failed for {kind}:", e)
    return artifacts


challenger_artifacts = _load_challenger_artifacts()


def _safe_metric(metric_entry, key):
    try:
        return float(metric_entry.get(key))
    except Exception:
        return float("-inf")


def _select_best_challenger_kind(available_kinds):
    if not available_kinds:
        return None

    ranked = []
    for kind in available_kinds:
        metric_entry = challenger_metrics.get(kind, {})
        auc = _safe_metric(metric_entry, "validation_auc")
        acc = _safe_metric(metric_entry, "validation_accuracy")
        tiebreak = CHALLENGER_KIND_TIEBREAK_PRIORITY.get(kind, 0)
        ranked.append((auc, acc, tiebreak, kind))

    ranked.sort(reverse=True)
    return ranked[0][3]


def _is_challenger_kind_available(kind):
    if kind in challenger_artifacts:
        return True
    if kind == CHALLENGER_XGB_RF_ENSEMBLE_KIND:
        return "xgboost" in challenger_artifacts and "random_forest" in challenger_artifacts
    return False


def _available_challenger_kinds():
    available = sorted(challenger_artifacts.keys())
    if _is_challenger_kind_available(CHALLENGER_XGB_RF_ENSEMBLE_KIND):
        available.append(CHALLENGER_XGB_RF_ENSEMBLE_KIND)
    return sorted(available)


if CHALLENGER_MODEL_KIND:
    if not _is_challenger_kind_available(CHALLENGER_MODEL_KIND):
        best_kind = _select_best_challenger_kind(list(challenger_artifacts.keys()))
        if best_kind is not None:
            print(
                f"[AFTA] requested challenger '{CHALLENGER_MODEL_KIND}' not available; "
                f"using best available '{best_kind}'"
            )
            CHALLENGER_MODEL_KIND = best_kind
        else:
            CHALLENGER_MODEL_KIND = "xgboost"
else:
    best_kind = _select_best_challenger_kind(list(challenger_artifacts.keys()))
    CHALLENGER_MODEL_KIND = best_kind or "xgboost"
    print(f"[AFTA] auto-selected challenger model: {CHALLENGER_MODEL_KIND}")

CHALLENGER_MODEL_PATH = (
    os.getenv("AFTA_CHALLENGER_MODEL_PATH")
    or (
        "/models/final_model_tuned.pkl"
        if os.path.exists("/models/final_model_tuned.pkl")
        else None
    )
    or os.path.join(BASE_DIR, "final_model_tuned.pkl")
)
challenger_model = None
if CHALLENGER_MODEL_PATH and os.path.exists(CHALLENGER_MODEL_PATH):
    try:
        challenger_model = ModelWrapper(CHALLENGER_MODEL_PATH)
        print("[AFTA] loaded legacy challenger model:", CHALLENGER_MODEL_PATH)
    except Exception as e:
        print("[AFTA] legacy challenger model load failed:", e)
else:
    print("[AFTA] legacy challenger model unavailable:", CHALLENGER_MODEL_PATH)

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


@app.route("/sensor_readings", methods=["POST"])
def ingest_sensor_readings():
    global latest_sensor_readings

    def _to_float(name, value):
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            raise ValueError(f"Invalid numeric value for '{name}'")
        if not np.isfinite(parsed):
            raise ValueError(f"Non-finite numeric value for '{name}'")
        return parsed

    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        temperature = _to_float("temperature", json_data.get("temperature"))
        humidity = _to_float("humidity", json_data.get("humidity"))
        ph_input = _to_float("ph", json_data.get("ph"))
        soil_input = _to_float("soil_moisture", json_data.get("soil_moisture"))
        ph_raw = _to_float("ph_raw", json_data.get("ph_raw"))
        soil_raw = _to_float("soil_raw", json_data.get("soil_raw"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if temperature is None or humidity is None:
        return jsonify({"error": "Missing required fields: temperature, humidity"}), 400

    if ph_input is None and ph_raw is None:
        return jsonify({"error": "Missing required field: ph (or ph_raw)"}), 400
    if soil_input is None and soil_raw is None:
        return jsonify({"error": "Missing required field: soil_moisture (or soil_raw)"}), 400

    # Prefer explicit non-raw fields; auto-convert if raw-like values are sent there.
    if ph_input is not None:
        ph_value = _adc_to_ph(ph_input) if ph_input > 14.0 else _clamp(ph_input, 0.0, 14.0)
    else:
        ph_value = _adc_to_ph(ph_raw)

    if soil_input is not None:
        soil_moisture_value = (
            _adc_to_soil_moisture_percent(soil_input)
            if soil_input > 100.0
            else _clamp(soil_input, 0.0, 100.0)
        )
    else:
        soil_moisture_value = _adc_to_soil_moisture_percent(soil_raw)

    payload = {
        "soil_moisture": round(float(soil_moisture_value), 2),
        "temperature": round(float(temperature), 2),
        "humidity": round(float(_clamp(humidity, 0.0, 100.0)), 2),
        "ph": round(float(ph_value), 2),
        "ph_raw": ph_raw if ph_raw is not None else json_data.get("ph_raw"),
        "soil_raw": soil_raw if soil_raw is not None else json_data.get("soil_raw"),
        "device_id": json_data.get("device_id", "esp32"),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }

    latest_sensor_readings = payload
    return jsonify({"status": "ok", "latest_sensor_readings": payload})


@app.route("/sensor_readings/latest", methods=["GET"])
def get_latest_sensor_readings():
    if latest_sensor_readings is None:
        return jsonify({"error": "No sensor readings received yet"}), 404
    return jsonify(latest_sensor_readings)

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

    try:
        _validate_sensor_features([float(x) for x in sensor_features])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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


def _extract_probability(predicted):
    arr = np.asarray(predicted, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            probability = float(arr[0, 1])
        else:
            probability = float(arr[0, 0])
    else:
        flat = arr.ravel()
        if flat.size == 0:
            raise ValueError("Empty probability output")
        probability = float(flat[0])
    return float(np.clip(probability, 0.0, 1.0))


def _resolve_challenger_kind(requested_kind):
    if isinstance(requested_kind, str):
        candidate = requested_kind.strip().lower()
        if _is_challenger_kind_available(candidate):
            return candidate
    if _is_challenger_kind_available(CHALLENGER_MODEL_KIND):
        return CHALLENGER_MODEL_KIND
    best_kind = _select_best_challenger_kind(list(challenger_artifacts.keys()))
    if best_kind is not None:
        return best_kind
    return "xgboost"


def _predict_single_challenger_probability(kind, sensor_features):
    artifact = challenger_artifacts.get(kind)
    if artifact is None:
        return None

    model_obj = artifact
    imputer = None
    feature_order = AFTA_FEATURES_ORDER

    if isinstance(artifact, dict):
        model_obj = artifact.get("model")
        imputer = artifact.get("imputer")
        feature_order = artifact.get("feature_order") or AFTA_FEATURES_ORDER

    if model_obj is None or not hasattr(model_obj, "predict_proba"):
        return None

    if not isinstance(feature_order, list) or len(feature_order) != len(sensor_features):
        feature_order = AFTA_FEATURES_ORDER

    X_df = pd.DataFrame([sensor_features], columns=feature_order)
    X_input = imputer.transform(X_df) if imputer is not None else X_df
    return _extract_probability(model_obj.predict_proba(X_input))


def _get_validation_auc_for_kind(kind):
    metric_entry = challenger_metrics.get(kind, {})
    auc = metric_entry.get("validation_auc")
    try:
        return float(auc)
    except Exception:
        return CHALLENGER_VALIDATION_AUC


def _predict_non_afta_challenger(sensor_features, requested_kind=None):
    resolved_kind = _resolve_challenger_kind(requested_kind)
    component_probabilities = None
    if resolved_kind == CHALLENGER_XGB_RF_ENSEMBLE_KIND:
        xgb_probability = _predict_single_challenger_probability("xgboost", sensor_features)
        rf_probability = _predict_single_challenger_probability("random_forest", sensor_features)
        if xgb_probability is None or rf_probability is None:
            return None
        probability = float(np.mean([xgb_probability, rf_probability]))
        validation_auc = float(
            np.mean(
                [
                    _get_validation_auc_for_kind("xgboost"),
                    _get_validation_auc_for_kind("random_forest"),
                ]
            )
        )
        component_probabilities = {
            "xgboost": round(float(xgb_probability), 4),
            "random_forest": round(float(rf_probability), 4),
        }
    else:
        probability = _predict_single_challenger_probability(resolved_kind, sensor_features)
        if probability is None:
            return None
        validation_auc = _get_validation_auc_for_kind(resolved_kind)

    result = {
        "probability": probability,
        "validation_auc": float(validation_auc),
        "model_name": CHALLENGER_MODEL_LABELS.get(resolved_kind, "Challenger Model"),
        "model_family": resolved_kind,
    }
    if component_probabilities is not None:
        result["component_probabilities"] = component_probabilities
    return result


def _apply_challenger_safety_override(sensor_features, probability):
    """
    Safety guard: zero soil humidity/moisture is treated as extreme dryness.
    This ensures challenger output aligns with agronomic intuition for this edge case.
    """
    try:
        soil_moisture = float(sensor_features[0])
        soil_humidity = float(sensor_features[2])
    except Exception:
        return int(probability >= 0.5), float(probability), None

    if soil_humidity <= 0.0 or soil_moisture <= 0.0:
        adjusted_probability = max(float(probability), 0.5001)
        override_reason = (
            "Safety override applied: zero soil humidity or soil moisture "
            "is treated as an irrigation-needed condition."
        )
        return 1, adjusted_probability, override_reason

    return int(probability >= 0.5), float(probability), None


@app.route("/predict_challenger_compare", methods=["POST"])
@app.route("/predict_catboost_compare", methods=["POST"])
def predict_challenger_compare():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    sensor_features = json_data.get("sensor_features")
    requested_kind = json_data.get("challenger_kind")
    if not requested_kind and request.path.endswith("/predict_catboost_compare"):
        # Backward-compatible behavior for the legacy catboost route alias.
        requested_kind = "catboost"
    explain_context = None

    if sensor_features is None:
        try:
            _, explain_context = _compute_full_intelligent_from_json(json_data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        sensor_features = explain_context.get("afta_sensor_features_12")

    if sensor_features is None:
        return jsonify({"error": "Missing sensor_features"}), 400

    try:
        sensor_features = [float(x) for x in sensor_features]
    except Exception:
        return jsonify({"error": "Invalid sensor_features array"}), 400

    try:
        _validate_sensor_features(sensor_features)
    except ValueError as e:
        return jsonify({"error": str(e), "received": len(sensor_features)}), 400

    challenger_result = None
    try:
        challenger_result = _predict_non_afta_challenger(sensor_features, requested_kind=requested_kind)
    except Exception as e:
        print("[AFTA] non-AFTA challenger prediction failed:", e)

    if challenger_result is not None:
        probability = float(challenger_result["probability"])
        model_name = challenger_result["model_name"]
        model_family = challenger_result["model_family"]
        validation_auc = float(challenger_result["validation_auc"])
        fallback_used = False
        fallback_source = "none"
    else:
        compare_model = challenger_model if challenger_model is not None else model
        X = np.asarray(sensor_features, dtype=np.float32).reshape(1, -1)
        probability = float(compare_model.predict_proba(X)[0])
        model_name = "Challenger Model (Legacy AFTA)"
        model_family = "legacy_afta"
        validation_auc = CHALLENGER_VALIDATION_AUC
        fallback_used = True
        fallback_source = "legacy_tuned_afta" if challenger_model is not None else "main_model"

    final_prediction, probability, safety_override_reason = _apply_challenger_safety_override(
        sensor_features=sensor_features,
        probability=probability,
    )

    confidence_distance = abs(probability - 0.5) * 2.0
    if confidence_distance >= 0.7:
        confidence_band = "High"
    elif confidence_distance >= 0.4:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"

    if final_prediction == 1:
        decision_reason = (
            "Strong irrigation risk signal from challenger model."
            if probability >= 0.7
            else "Moderate irrigation risk; field check is recommended."
        )
    else:
        decision_reason = (
            "Low irrigation risk signal from challenger model."
            if probability <= 0.3
            else "Borderline no-irrigation decision; monitor moisture trend."
        )
    if safety_override_reason:
        decision_reason = safety_override_reason

    response = {
        "model_name": model_name,
        "model_family": model_family,
        "requested_model_family": _resolve_challenger_kind(requested_kind),
        "available_model_families": _available_challenger_kinds(),
        "final_prediction": int(final_prediction),
        "prediction_text": "Irrigation Needed" if final_prediction == 1 else "No Irrigation Needed",
        "probability": round(probability, 4),
        "confidence_band": confidence_band,
        "validation_auc": validation_auc,
        "decision_reason": decision_reason,
        "fallback_used": fallback_used,
        "fallback_source": fallback_source,
    }
    if safety_override_reason:
        response["safety_override"] = {
            "applied": True,
            "reason": safety_override_reason,
        }

    if explain_context and explain_context.get("context_score") is not None:
        response["context_score"] = round(float(explain_context["context_score"]), 3)
    if challenger_result and challenger_result.get("component_probabilities"):
        response["component_probabilities"] = challenger_result["component_probabilities"]

    return jsonify(response)

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
