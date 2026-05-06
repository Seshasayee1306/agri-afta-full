import base64
import os
from io import BytesIO

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq
from PIL import Image

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as exc:
        print("⚠️ Groq client init failed:", exc)

app = Flask(__name__)
CORS(app)


def _decode_image_from_request(req):
    if "image" in req.files:
        f = req.files["image"]
        data = f.read()
        if not data:
            raise ValueError("Uploaded file is empty")
        return data

    payload = req.get_json(silent=True) or {}
    b64 = payload.get("image_base64")
    if not b64:
        raise ValueError("Missing image. Send multipart field 'image' or JSON field 'image_base64'.")

    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
    except Exception:
        raise ValueError("Invalid base64 image")


def _bytes_to_rgb(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception:
        raise ValueError("Unable to decode image")


def _segment_hsv_otsu(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    disease_pct = float(np.count_nonzero(mask) / mask.size) * 100.0

    overlay = image_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.45 + np.array([180, 20, 20])).astype(np.uint8)

    return mask, overlay.astype(np.uint8), disease_pct


def _classify_from_coverage(disease_percentage):
    if disease_percentage < 5:
        disease_class = "Healthy"
        severity = "None"
        confidence = 0.95
    elif disease_percentage < 15:
        disease_class = "Early Blight"
        severity = "Mild"
        confidence = 0.86
    elif disease_percentage < 30:
        disease_class = "Leaf Spot"
        severity = "Moderate"
        confidence = 0.82
    elif disease_percentage < 50:
        disease_class = "Late Blight"
        severity = "Severe"
        confidence = 0.79
    else:
        disease_class = "Critical Leaf Disease"
        severity = "Critical"
        confidence = 0.84

    return {
        "disease_class": disease_class,
        "severity": severity,
        "confidence": confidence,
    }


def _png_base64(image_arr):
    pil = Image.fromarray(image_arr)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _llm_explain(result):
    if not client:
        return (
            "The highlighted regions show likely stressed/diseased tissue. "
            "Larger affected coverage increases disease severity."
        )

    prompt = f"""
You are an agriculture disease analyst.
A plant image was segmented for disease-like regions.

Result:
- disease_class: {result['disease_class']}
- severity: {result['severity']}
- confidence: {result['confidence']}
- disease_coverage_percent: {result['disease_coverage_percent']}

Explain in simple terms:
1) Why this disease class was predicted from the visible pattern.
2) What visual signs likely contributed.
3) What immediate farmer action to take.
Keep it concise and practical.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print("⚠️ Groq explain failed:", exc)
        return (
            "The model found discolored leaf regions that indicate stress. "
            "Please inspect leaf underside, isolate affected plants, and begin targeted treatment."
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "disease service running"})


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        image_bytes = _decode_image_from_request(request)
        image_rgb = _bytes_to_rgb(image_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    mask, overlay, coverage = _segment_hsv_otsu(image_rgb)
    disease_info = _classify_from_coverage(coverage)

    result = {
        **disease_info,
        "disease_coverage_percent": round(float(coverage), 2),
        "mask_base64": _png_base64(mask),
        "overlay_base64": _png_base64(overlay),
        "recommended_context_disease_status": disease_info["disease_class"],
        "inference_source": "hsv_otsu_segmentation",
    }

    result["llm_explanation"] = _llm_explain(result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, debug=True)
