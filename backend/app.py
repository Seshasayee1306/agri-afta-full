# backend/app.py
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
# =====================================================
# STRESS INDEX CALCULATION
# =====================================================
def calculate_stress_index(stage, soil_moisture, temperature, rainfall, ndvi=0.5):

    # Normalize inputs
    soil_factor = 1 - (soil_moisture / 100)
    temp_factor = min(max((temperature - 20) / 20, 0), 1)
    rain_factor = 1 - min(rainfall / 200, 1)
    ndvi_factor = 1 - ndvi

    # Stage sensitivity weight
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
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")
model = ModelWrapper(MODEL_PATH)

# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend is running"})

# -----------------------------------------------------
# PREDICT (UNCHANGED)
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
# EXPLAIN (UNCHANGED)
# -----------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    features = json_data.get("features")
    feature_names = json_data.get("feature_names")

    if features is None:
        return jsonify({"error": "Missing 'features' field"}), 400

    X = np.array(features, dtype=np.float32).reshape(1, -1)

    pred = model.predict(X)
    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    emb, proba, _ = model.get_embeddings_and_pred(X)
    shap_vals = shap_contribs(model.head, emb)
    masks = tabnet_masks(model.encoder, X)

    if feature_names and len(feature_names) == len(features):
        raw_row = {feature_names[i]: features[i] for i in range(len(features))}
    else:
        raw_row = {f"f{i}": features[i] for i in range(len(features))}

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
        "shap_values": shap_vals[0].tolist(),
        "tabnet_masks": masks.tolist(),
        "llm_explanation": explanation
    })

# -----------------------------------------------------
# LABEL DATA (UNCHANGED)
# -----------------------------------------------------
@app.route("/label", methods=["POST"])
def label_data():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    features = data.get("features")
    label = data.get("label")

    if features is None or label is None:
        return jsonify({"error": "Missing features or label"}), 400

    append_labeled_row(features, label)
    return jsonify({"status": "Data appended successfully"})

# -----------------------------------------------------
# PROMETHEUS METRICS (UNCHANGED)
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
# ✅ CONTEXT-AWARE PREDICTION (FIXED ONLY)
# =====================================================
@app.route("/predict_with_context", methods=["POST"])
def predict_with_context():
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Sensor features (UNCHANGED)
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

    # Context features
    context = json_data.get("context", {})

    region = context.get("region", "Unknown")   # ✅ FIX
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

    # -----------------------------
    # 1️⃣ Stage Detection
    # -----------------------------
    sowing_date = json_data.get("sowing_date")
    current_date = json_data.get("current_date")

    if not sowing_date or not current_date:
        return jsonify({"error": "Missing sowing_date or current_date"}), 400

    days = calculate_days_after_sowing(sowing_date, current_date)
    stage = identify_growth_stage(days)

    # -----------------------------
    # 2️⃣ Stage Model Prediction
    # -----------------------------
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

    # -----------------------------
    # 3️⃣ AFTA Sensor Prediction
    # -----------------------------
    sensor_features = json_data.get("sensor_features")
    if sensor_features is None:
        return jsonify({"error": "Missing sensor_features for AFTA model"}), 400

    X = normalize(sensor_features).reshape(1, -1)
    afta_prediction = int(model.predict(X))

    # -----------------------------
    # 4️⃣ Context Prediction
    # -----------------------------
    context = json_data.get("context", {})

    region = context.get("region", "Unknown")
    crop_type = context.get("crop_type", "Unknown")
    ndvi = float(context.get("ndvi", 0.5))
    disease_status = context.get("disease_status", "None")
    temperature_ctx = float(context.get("temperature", 25))
    rainfall_ctx = float(context.get("rainfall", 100))
    humidity_ctx = float(context.get("humidity", 60))

    try:
        context_score = context_model.predict_context_score(
            region=region,
            crop_type=crop_type,
            ndvi=ndvi,
            disease_status=disease_status,
            temperature=temperature_ctx,
            rainfall=rainfall_ctx,
            humidity=humidity_ctx
        )
    except Exception as e:
        print("Context model not available:", e)
        context_score = 0.5

    # -----------------------------
    # 5️⃣ Final Decision Logic
    # -----------------------------
    votes = [stage_prediction, afta_prediction]

    if context_score >= 0.6:
        votes.append(1)
    else:
        votes.append(0)

    if stress_index >= 0.6:
        final_prediction = 1
    elif sum(votes) >= 2:
        final_prediction = 1
    else:
        final_prediction = 0
    # -----------------------------------------------------
    # 6️⃣ STRESS + WATER RECOMMENDATION
    # -----------------------------------------------------
    stress_index = calculate_stress_index(
        stage=stage,
        soil_moisture=feature_dict["soil_moisture"],
        temperature=feature_dict["temperature"],
        rainfall=feature_dict["rainfall"],
        ndvi=ndvi
    )

    if stress_index >= 0.7:
        irrigation_level = "High"
        water_liters = 25
    elif stress_index >= 0.4:
        irrigation_level = "Medium"
        water_liters = 15
    else:
        irrigation_level = "Low"
        water_liters = 5

    return jsonify({
        "growth_stage": stage,
        "stage_model_prediction": stage_prediction,
        "afta_prediction": afta_prediction,
        "context_score": round(float(context_score), 3),
        "final_prediction": final_prediction,
        "stress_index": stress_index,
        "irrigation_level": irrigation_level,
        "recommended_water_liters": water_liters
    })
from flask import send_from_directory, render_template

# -----------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
# -----------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
