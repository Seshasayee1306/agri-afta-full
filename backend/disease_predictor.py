import base64
import json
import os
from io import BytesIO
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from groq import Groq
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = os.getenv(
    "DISEASE_MODEL_PATH",
    os.path.join(BASE_DIR, "disease_model.pt"),
)
DISEASE_CLASSES_PATH = os.getenv(
    "DISEASE_CLASSES_PATH",
    os.path.join(BASE_DIR, "disease_class_names.json"),
)

_CLASSIFIER = None
_CLASS_NAMES = None
_CLASSIFIER_ERROR = None


def _build_classifier(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _load_classifier_once():
    global _CLASSIFIER, _CLASS_NAMES, _CLASSIFIER_ERROR

    if _CLASSIFIER is not None or _CLASSIFIER_ERROR is not None:
        return

    if not (os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_CLASSES_PATH)):
        _CLASSIFIER_ERROR = "trained_model_missing"
        return

    try:
        with open(DISEASE_CLASSES_PATH, "r", encoding="utf-8") as f:
            _CLASS_NAMES = json.load(f)
        if not isinstance(_CLASS_NAMES, list) or not _CLASS_NAMES:
            raise ValueError("invalid class names")

        model = _build_classifier(len(_CLASS_NAMES))
        state = torch.load(DISEASE_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _CLASSIFIER = model
        print(f"[Disease] loaded trained classifier from {DISEASE_MODEL_PATH}")
    except Exception as exc:
        _CLASSIFIER_ERROR = str(exc)
        _CLASSIFIER = None
        _CLASS_NAMES = None
        print(f"[Disease] classifier load failed: {exc}")


_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _predict_with_classifier(image_rgb: np.ndarray) -> Dict[str, object] | None:
    _load_classifier_once()
    if _CLASSIFIER is None or not _CLASS_NAMES:
        return None

    pil = Image.fromarray(image_rgb)
    x = _INFER_TRANSFORM(pil).unsqueeze(0)

    with torch.no_grad():
        logits = _CLASSIFIER(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    label = _CLASS_NAMES[int(idx.item())]
    confidence = float(conf.item())
    severity = "None" if "healthy" in label.lower() else "Moderate"
    if confidence >= 0.85 and severity != "None":
        severity = "High"

    return {
        "disease_class": label,
        "severity": severity,
        "confidence": round(confidence, 4),
        "inference_source": "trained_classifier",
    }


def _groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        print("⚠️ Groq client init failed:", exc)
        return None


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception:
        raise ValueError("Unable to decode image")


def decode_base64_image(image_base64: str) -> np.ndarray:
    if not image_base64:
        raise ValueError("Missing image_base64")
    try:
        b64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        image_bytes = base64.b64decode(b64)
        return decode_image_bytes(image_bytes)
    except Exception:
        raise ValueError("Invalid base64 image")


def segment_hsv_otsu(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    coverage = float(np.count_nonzero(mask) / mask.size) * 100.0

    overlay = image_rgb.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.45 + np.array([180, 20, 20])
    ).astype(np.uint8)

    return mask, overlay.astype(np.uint8), coverage


def classify_from_coverage(coverage: float) -> Dict[str, object]:
    if coverage < 5:
        return {"disease_class": "Healthy", "severity": "None", "confidence": 0.95}
    if coverage < 15:
        return {"disease_class": "Early Blight", "severity": "Mild", "confidence": 0.86}
    if coverage < 30:
        return {"disease_class": "Leaf Spot", "severity": "Moderate", "confidence": 0.82}
    if coverage < 50:
        return {"disease_class": "Late Blight", "severity": "Severe", "confidence": 0.79}
    return {
        "disease_class": "Critical Leaf Disease",
        "severity": "Critical",
        "confidence": 0.84,
    }


def classify_disease(image_rgb: np.ndarray, coverage: float) -> Dict[str, object]:
    trained = _predict_with_classifier(image_rgb)
    if trained is not None:
        return trained
    return classify_from_coverage(coverage)


def to_png_base64(image_arr: np.ndarray) -> str:
    pil = Image.fromarray(image_arr)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def llm_explain_disease(result: Dict[str, object]) -> str:
    client = _groq_client()
    if not client:
        return (
            "The highlighted regions show likely stressed or diseased tissue. "
            "Higher affected area suggests stronger disease severity."
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
            "The model found discolored leaf regions indicating plant stress. "
            "Inspect nearby leaves, isolate affected plants, and start targeted treatment."
        )
