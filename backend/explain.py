# backend/explain.py

import os
import numpy as np
from typing import Dict
import shap
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# LOAD GROQ KEY
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("⚠️ Groq client error:", e)
        client = None

# ------------------------------------------------------
# 1. SHAP CONTRIBUTIONS
# ------------------------------------------------------
def shap_contribs(head, embeddings):
    """
    Compute SHAP values for model predictions.
    head: XGBoost Booster object
    embeddings: numpy array of embeddings
    """

    # 🔧 FIX 1: sanitize base_score BEFORE SHAP reads it
    try:
        raw_base = head.attr("base_score")
        if raw_base:
            # "[3.7E-1]" → 0.37
            clean_base = float(raw_base.strip("[]"))
            head.set_attr(base_score=str(clean_base))
    except Exception as e:
        print("⚠️ base_score sanitization skipped:", e)

    try:
        explainer = shap.TreeExplainer(head)
        vals = explainer.shap_values(embeddings)

        # Binary classification → class 1
        if isinstance(vals, list):
            return vals[1]

        return vals

    except Exception as e:
        print("⚠️ SHAP failed:", e)
        return None

# ------------------------------------------------------
# 2. TABNET MASKS (optional / fallback)
# ------------------------------------------------------
def tabnet_masks(encoder, X):
    """
    Returns TabNet masks. Currently returns zeros as placeholder.
    """
    try:
        return np.zeros(X.shape[1])
    except Exception:
        return np.zeros(len(X))

# ------------------------------------------------------
# 3. LLM EXPLANATION (GROQ WORKING MODEL + fallback)
# ------------------------------------------------------
def llm_explain(raw_row: Dict, shap_vals, masks, pred):
    """
    Generate natural language explanation for prediction.
    raw_row: dictionary of feature_name: value
    shap_vals: numpy array of SHAP contributions
    masks: TabNet masks
    pred: integer prediction (0/1)
    """

    # 🔧 FIX 2: guard if SHAP failed
    if shap_vals is None:
        shap_vals = np.zeros(len(raw_row))

    prediction_text = "Needs water" if pred == 1 else "No irrigation needed"

    prompt = f"""
You are an agricultural assistant.

Sensor readings: {raw_row}
Prediction: {prediction_text}

SHAP values: {shap_vals.tolist()}
TabNet masks: {masks.tolist()}

Explain simply:
1) Why the model predicted this.
2) Top 3 most influential features.
3) A clear recommendation for the farmer.
Use simple non-technical language.
"""

    # ---------- GROQ LLM ----------
    if client:
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            print("⚠️ Groq error — using fallback:", e)

    # ---------- FALLBACK ----------
    top_idx = np.argsort(-np.abs(shap_vals))[0:3]
    feats = list(raw_row.keys())
    important = [f"{feats[i]} is strongly influencing the decision." for i in top_idx]

    return (
        f"Prediction: {prediction_text}. "
        "Key reasons: "
        + " | ".join(important)
        + ". Recommendation: follow the model's suggestion."
    )
