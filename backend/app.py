# backend/app.py

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
from backend.utils.validator import validate_sensor_data

# ... (imports)

@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.get_json(silent=True)
    data = json_data.get("features") if json_data else None

    if data is None:
        return jsonify({"error": "No input features provided"}), 400

    # 🚨 PHYSICS VALIDATION
    # If features is a list, we do strict validation (legacy)
    if isinstance(data, list):
        is_valid, reason = validate_sensor_data(data)
        if not is_valid:
            print(f"⚠️ Invalid Data Detected: {reason}")
            return jsonify({
                "prediction": 0,  
                "confidence_score": 0.0, 
                "error": reason   
            })
        # For list, we pass as is (wrapper expects list or dict)
        X_input = data
    elif isinstance(data, dict):
        # For dict (partial), we skip strict "all 12 present" validation
        # and let the wrapper impute.
        # We could add range checks for present keys later.
        X_input = data
    else:
        return jsonify({"error": "Features must be a list or a dictionary"}), 400

    # 🔹 FINE-TUNED ML MODEL + RULE-BASED SAFETY NET
    # Extract key features for rule-based logic (safety net only)
    soil_moisture = None
    temperature = None
    rainfall = None
    
    if isinstance(X_input, list) and len(X_input) >= 8:
        soil_moisture = X_input[0]
        temperature = X_input[1]
        rainfall = X_input[7]
    elif isinstance(X_input, dict):
        soil_moisture = X_input.get('soil_moisture')
        temperature = X_input.get('temperature') 
        rainfall = X_input.get('rainfall')
    
    # Rule 1: Extreme Drought (very dry soil + no/low rain)
    if soil_moisture is not None and rainfall is not None:
        if soil_moisture < 25 and rainfall < 20:
            return jsonify({
                "prediction": 1,
                "accuracy": 0.95,
                "confidence_score": 0.95,
                "source": "rule-based-drought"
            })
    
    # Rule 2: Very dry soil regardless of rain
    if soil_moisture is not None and soil_moisture < 15:
        return jsonify({
            "prediction": 1,
            "accuracy": 0.90,
            "confidence_score": 0.90,
            "source": "rule-based-extreme-dry"
        })
    
    # Rule 3: Hot + Moderately dry
    if soil_moisture is not None and temperature is not None:
        if soil_moisture < 40 and temperature > 35:
            return jsonify({
                "prediction": 1,
                "accuracy": 0.85,
                "confidence_score": 0.85,
                "source": "rule-based-hot-dry"
            })
    
    # Rule 4: Very wet soil (no irrigation needed)
    if soil_moisture is not None and soil_moisture > 75:
        return jsonify({
            "prediction": 0,
            "accuracy": 0.05,
            "confidence_score": 0.95,
            "source": "rule-based-wet"
        })
    
    # Fine-tuned ML model (should handle all cases correctly now)
    pred = model.predict(X_input)
    prob = model.predict_proba(X_input)
    
    # Calculate explicit confidence (0.5 to 1.0)
    confidence = prob if pred == 1 else 1 - prob

    return jsonify({
        "prediction": int(pred), 
        "accuracy": float(prob), 
        "confidence_score": float(confidence),
        "source": "ml-model"
    })

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
        
    # 🚨 PHYSICS VALIDATION (Added to Explain too)
    is_valid, reason = validate_sensor_data(features)
    if not is_valid:
        return jsonify({
            "prediction": 0,
            "prediction_text": "Invalid Data",
            "probability": 0.0,
            "confidence_score": 0.0,
            "llm_explanation": f"⚠️ Cannot explain invalid data: {reason}",
            "shap_values": [0]*12,
            "tabnet_masks": [0]*12
        })

    # 🔹 RULE-BASED LOGIC (SAME AS /predict)
    # Extract key features
    soil_moisture = features[0] if len(features) > 0 else None
    temperature = features[1] if len(features) > 1 else None
    rainfall = features[7] if len(features) > 7 else None
    
    # Rule 1: Extreme Drought
    if soil_moisture is not None and rainfall is not None:
        if soil_moisture < 25 and rainfall < 20:
            pred = 1
            proba = 0.95
            prediction_text = "Needs water"
            source = "rule-based-drought"
            
            # Still generate SHAP/masks for explanation
            X = np.array(features, dtype=np.float32).reshape(1, -1)
            emb, _, _ = model.get_embeddings_and_pred(X)
            shap_vals = shap_contribs(model.head, emb)
            masks = tabnet_masks(model.encoder, X)
            confidence = proba
            
            if feature_names and len(feature_names) == len(features):
                raw_row = {feature_names[i]: features[i] for i in range(len(features))}
            else:
                raw_row = {f"f{i}": features[i] for i in range(len(features))}
            
            explanation = llm_explain(raw_row=raw_row, shap_vals=shap_vals[0], masks=masks, pred=pred)
            
            return jsonify({
                "prediction": int(pred),
                "prediction_text": prediction_text,
                "probability": float(proba),
                "confidence_score": float(confidence),
                "source": source,
                "shap_values": shap_vals[0].tolist(),
                "tabnet_masks": masks.tolist(),
                "llm_explanation": explanation
            })
    
    # Rule 2: Very dry soil
    if soil_moisture is not None and soil_moisture < 15:
        pred = 1
        proba = 0.90
        prediction_text = "Needs water"
        source = "rule-based-extreme-dry"
        
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        emb, _, _ = model.get_embeddings_and_pred(X)
        shap_vals = shap_contribs(model.head, emb)
        masks = tabnet_masks(model.encoder, X)
        confidence = proba
        
        if feature_names and len(feature_names) == len(features):
            raw_row = {feature_names[i]: features[i] for i in range(len(features))}
        else:
            raw_row = {f"f{i}": features[i] for i in range(len(features))}
        
        explanation = llm_explain(raw_row=raw_row, shap_vals=shap_vals[0], masks=masks, pred=pred)
        
        return jsonify({
            "prediction": int(pred),
            "prediction_text": prediction_text,
            "probability": float(proba),
            "confidence_score": float(confidence),
            "source": source,
            "shap_values": shap_vals[0].tolist(),
            "tabnet_masks": masks.tolist(),
            "llm_explanation": explanation
        })
    
    # Rule 3: Hot + Moderately dry
    if soil_moisture is not None and temperature is not None:
        if soil_moisture < 40 and temperature > 35:
            pred = 1
            proba = 0.85
            prediction_text = "Needs water"
            source = "rule-based-hot-dry"
            
            X = np.array(features, dtype=np.float32).reshape(1, -1)
            emb, _, _ = model.get_embeddings_and_pred(X)
            shap_vals = shap_contribs(model.head, emb)
            masks = tabnet_masks(model.encoder, X)
            confidence = proba
            
            if feature_names and len(feature_names) == len(features):
                raw_row = {feature_names[i]: features[i] for i in range(len(features))}
            else:
                raw_row = {f"f{i}": features[i] for i in range(len(features))}
            
            explanation = llm_explain(raw_row=raw_row, shap_vals=shap_vals[0], masks=masks, pred=pred)
            
            return jsonify({
                "prediction": int(pred),
                "prediction_text": prediction_text,
                "probability": float(proba),
                "confidence_score": float(confidence),
                "source": source,
                "shap_values": shap_vals[0].tolist(),
                "tabnet_masks": masks.tolist(),
                "llm_explanation": explanation
            })
    
    # Rule 4: Very wet soil
    if soil_moisture is not None and soil_moisture > 75:
        pred = 0
        proba = 0.05
        prediction_text = "No irrigation needed"
        source = "rule-based-wet"
        
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        emb, _, _ = model.get_embeddings_and_pred(X)
        shap_vals = shap_contribs(model.head, emb)
        masks = tabnet_masks(model.encoder, X)
        confidence = 0.95
        
        if feature_names and len(feature_names) == len(features):
            raw_row = {feature_names[i]: features[i] for i in range(len(features))}
        else:
            raw_row = {f"f{i}": features[i] for i in range(len(features))}
        
        explanation = llm_explain(raw_row=raw_row, shap_vals=shap_vals[0], masks=masks, pred=pred)
        
        return jsonify({
            "prediction": int(pred),
            "prediction_text": prediction_text,
            "probability": float(proba),
            "confidence_score": float(confidence),
            "source": source,
            "shap_values": shap_vals[0].tolist(),
            "tabnet_masks": masks.tolist(),
            "llm_explanation": explanation
        })

    # ML Model (if no rules triggered)
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    pred = model.predict(X)
    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    # Restore AFTA Hybrid logic
    emb, proba, _ = model.get_embeddings_and_pred(X)
    shap_vals = shap_contribs(model.head, emb) # XGBoost head is back
    masks = tabnet_masks(model.encoder, X)
    
    # Calculate explicit confidence
    confidence = proba if pred == 1 else 1 - proba

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
        "confidence_score": float(confidence),
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
# ✅ ADAPTIVE AFTA SYSTEM (NEW)
# =====================================================
from backend.afta_system import process_afta_request, format_afta_output

@app.route("/afta", methods=["POST"])
def afta():
    """
    Adaptive Fuzzy Threshold Adjustment endpoint
    Accepts any number of sensor values and returns intelligent irrigation decision
    """
    json_data = request.get_json(silent=True)
    
    # Handle auto-generation mode
    if not json_data or json_data.get("values") == "auto":
        result = process_afta_request("auto")
    else:
        values = json_data.get("values", [])
        result = process_afta_request(values)
    
    # Return both JSON and formatted text
    formatted_text = format_afta_output(result)
    
    return jsonify({
        **result,
        "formatted_output": formatted_text
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

# -----------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=False, use_reloader=False)
