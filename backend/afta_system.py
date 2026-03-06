"""
Adaptive Fuzzy Threshold Adjustment (AFTA) System
Handles variable sensor inputs with intelligent cleaning and dynamic accuracy
"""

import numpy as np
from flask import jsonify
import random


def clean_sensor_values(values):
    """
    Clean sensor values by removing invalid data
    - Remove negatives
    - Remove values > 100
    - Remove null/non-numeric
    """
    if not values:
        return None
    
    cleaned = []
    for v in values:
        if v is not None and isinstance(v, (int, float)):
            if 0 <= v <= 100:
                cleaned.append(float(v))
    
    return cleaned if cleaned else None


def generate_realistic_sensor_data():
    """Generate realistic sensor data for self-generation mode"""
    # Generate 8-12 sensor values
    n = random.randint(8, 12)
    
    # Mix of scenarios
    scenario = random.choice(['wet', 'dry', 'moderate'])
    
    if scenario == 'wet':
        values = [random.uniform(60, 90) for _ in range(n)]
    elif scenario == 'dry':
        values = [random.uniform(10, 35) for _ in range(n)]
    else:
        values = [random.uniform(35, 65) for _ in range(n)]
    
    return values


def calculate_statistics(values):
    """Calculate mean, variance, and stability index"""
    mean = np.mean(values)
    variance = np.var(values)
    
    # Stability index: higher is more stable (0-1 scale)
    # Low variance → high stability
    stability_index = 1 / (1 + variance / 100)
    
    return mean, variance, stability_index


def adaptive_threshold_decision(mean, variance, stability_index):
    """
    Apply adaptive fuzzy threshold logic
    Thresholds adjust based on data stability
    """
    # Base thresholds
    dry_threshold = 30
    moderate_threshold = 45
    
    # Adjust thresholds slightly based on variance
    # High variance → more conservative (increase dry threshold)
    if variance > 20:
        dry_threshold += 2
        moderate_threshold += 2
    
    # Decision logic
    if mean < dry_threshold:
        decision = "NEED WATER"
        base_confidence = 95
    elif dry_threshold <= mean <= moderate_threshold:
        decision = "WATER REQUIRED"
        base_confidence = 80
    else:
        decision = "NO IRRIGATION REQUIRED"
        base_confidence = 90
    
    # Adjust confidence based on stability
    # More stable data → higher confidence
    confidence = min(99, base_confidence * stability_index)
    
    return decision, confidence


def calculate_dynamic_accuracy(original_count, cleaned_count, stability_index):
    """
    Calculate dynamic accuracy based on data quality
    Not a fixed value!
    """
    # Valid value ratio
    valid_ratio = cleaned_count / original_count if original_count > 0 else 0
    
    # Noise removal factor
    noise_factor = 1 - ((original_count - cleaned_count) / original_count) if original_count > 0 else 1
    
    # Combine factors
    accuracy = valid_ratio * stability_index * noise_factor * 100
    
    return min(99.9, accuracy)


def process_afta_request(values):
    """
    Main AFTA processing function
    """
    # Handle auto-generation
    if values == "auto" or not values:
        values = generate_realistic_sensor_data()
        original_values = values.copy()
    else:
        original_values = values.copy()
    
    # Clean values
    cleaned = clean_sensor_values(values)
    
    # Check for sensor failure
    if not cleaned:
        return {
            "status": "error",
            "message": "Sensor Failure – Recalibration Required",
            "cleaned_values": [],
            "mean_moisture": 0,
            "variance": 0,
            "stability_index": 0,
            "decision": "SENSOR FAILURE",
            "confidence_score": 0,
            "dynamic_accuracy": 0
        }
    
    # Calculate statistics
    mean, variance, stability_index = calculate_statistics(cleaned)
    
    # Make decision
    decision, confidence = adaptive_threshold_decision(mean, variance, stability_index)
    
    # Calculate dynamic accuracy
    accuracy = calculate_dynamic_accuracy(len(original_values), len(cleaned), stability_index)
    
    return {
        "status": "success",
        "cleaned_values": [round(v, 2) for v in cleaned],
        "mean_moisture": round(mean, 2),
        "variance": round(variance, 2),
        "stability_index": round(stability_index, 2),
        "decision": decision,
        "confidence_score": round(confidence, 1),
        "dynamic_accuracy": round(accuracy, 1)
    }


def format_afta_output(result):
    """Format output in the required structure"""
    if result["status"] == "error":
        return f"""
---------------------------------
{result["message"]}
Cleaned Sensor Values: []
Mean Moisture: 0.00
Variance: 0.00
Stability Index: 0.00
Decision: SENSOR FAILURE
Confidence Score: 0%
Dynamic Accuracy: 0%
---------------------------------
"""
    
    return f"""
---------------------------------
Cleaned Sensor Values: {result["cleaned_values"]}
Mean Moisture: {result["mean_moisture"]}
Variance: {result["variance"]}
Stability Index: {result["stability_index"]}
Decision: {result["decision"]}
Confidence Score: {result["confidence_score"]}%
Dynamic Accuracy: {result["dynamic_accuracy"]}%
---------------------------------
"""
