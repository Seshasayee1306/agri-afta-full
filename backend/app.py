# backend/app.py

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_loader import ModelWrapper
from explain import shap_contribs, tabnet_masks, llm_explain

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "final_model.pkl")
model = ModelWrapper(model_path)

# -----------------------------------------------------
# SIMPLE PREDICT ENDPOINT
# -----------------------------------------------------
# -----------------------------------------------------
# SIMPLE PREDICT ENDPOINT (with automatic scaling)
# -----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("features", None)

    if data is None:
        return jsonify({"error": "No input features provided"}), 400

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    # Define min/max for each feature based on training CSV
    feature_mins = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    feature_maxs = np.array([100, 50, 100, 23, 365, 50, 100, 50, 14, 100, 50, 100], dtype=np.float32)
    
    X_norm = np.array(data, dtype=np.float32).reshape(1, -1)
    
    # Scale normalized values [0,1] → original ranges
    X_scaled = X_norm * (feature_maxs - feature_mins) + feature_mins

    # -----------------------------
    # PREDICTION
    # -----------------------------
    pred = model.predict(X_scaled)

    return jsonify({"prediction": int(pred)})



# -----------------------------------------------------
# 🔥 FULL EXPLAIN ENDPOINT
# -----------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    json_data = request.json
    features = json_data.get("features", None)
    feature_names = json_data.get("feature_names", None)

    if features is None:
        return jsonify({"error": "Missing 'features' field"}), 400

    X = np.array(features, dtype=np.float32).reshape(1, -1)

    # 1. Use predict() endpoint logic
    pred = model.predict(X)                # 0 or 1
    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    # 2. Get embeddings and probability
    emb, proba, pred_class = model.get_embeddings_and_pred(X)

    # 3. SHAP values
    shap_vals = shap_contribs(model.head, emb)

    # 4. TabNet masks (optional/fallback)
    masks = tabnet_masks(model.encoder, X)

    # 5. Build raw_row for LLM explanation
    if feature_names and len(feature_names) == len(features):
        raw_row = {feature_names[i]: features[i] for i in range(len(features))}
    else:
        raw_row = {f"f{i}": features[i] for i in range(len(features))}

    # 6. LLM Explanation using consistent prediction
    explanation = llm_explain(
        raw_row=raw_row,
        shap_vals=shap_vals[0],
        masks=masks,
        pred=pred,
    )

    # 7. Response
    return jsonify({
        "prediction": int(pred),
        "prediction_text": prediction_text,      # human-readable label
        "probability": float(proba),
        "shap_values": shap_vals[0].tolist(),
        "tabnet_masks": masks.tolist(),
        "llm_explanation": explanation
    })


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "backend running"}


# -----------------------------------------------------
# RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
