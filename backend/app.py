# backend/app.py

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ✅ Relative imports (work in local + Docker)
from backend.model_loader import ModelWrapper
from backend.explain import shap_contribs, tabnet_masks, llm_explain
from backend.data_logger import append_labeled_row

# -----------------------------------------------------
# FLASK APP
# -----------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------
# LOAD MODEL (safe path resolution)
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")

model = ModelWrapper(MODEL_PATH)

# -----------------------------------------------------
# ROOT ENDPOINT (prevents 404 confusion)
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend is running"})

# -----------------------------------------------------
# SIMPLE PREDICT ENDPOINT (with scaling)
# -----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.get_json(silent=True)
    data = json_data.get("features") if json_data else None

    if data is None:
        return jsonify({"error": "No input features provided"}), 400

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    feature_mins = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.float32
    )
    feature_maxs = np.array(
        [100, 50, 100, 23, 365, 50, 100, 50, 14, 100, 50, 100],
        dtype=np.float32
    )

    X_norm = np.array(data, dtype=np.float32).reshape(1, -1)
    X_scaled = X_norm * (feature_maxs - feature_mins) + feature_mins

    # -----------------------------
    # PREDICTION
    # -----------------------------
    pred = model.predict(X_scaled)

    return jsonify({
        "prediction": int(pred)
    })

# -----------------------------------------------------
# FULL EXPLAIN ENDPOINT
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

    # 1. Prediction
    pred = model.predict(X)
    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    # 2. Embeddings + probability
    emb, proba, pred_class = model.get_embeddings_and_pred(X)

    # 3. SHAP contributions
    shap_vals = shap_contribs(model.head, emb)

    # 4. TabNet masks (fallback safe)
    masks = tabnet_masks(model.encoder, X)

    # 5. Raw row for LLM explanation
    if feature_names and len(feature_names) == len(features):
        raw_row = {feature_names[i]: features[i] for i in range(len(features))}
    else:
        raw_row = {f"f{i}": features[i] for i in range(len(features))}

    # 6. LLM explanation
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


@app.route("/label", methods=["POST"])
def label_data():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    features = data.get("features")
    label = data.get("label")

    if features is None or label is None:
        return jsonify({"error": "Missing features or label"}), 400

    try:
        append_labeled_row(features, label)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "Data appended successfully"})

# -----------------------------------------------------
# HEALTH CHECK (K8s / Docker friendly)
# -----------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "backend running"})

# -----------------------------------------------------
# LOCAL DEV ENTRYPOINT ONLY
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)