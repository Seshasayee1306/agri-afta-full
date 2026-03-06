# backend/utils/sensor_normalizer.py

import numpy as np

# EXACTLY 12 FEATURES — MUST MATCH ENCODER
RANGES = [
    (5, 60),     # soil_moisture (%)
    (10, 40),    # temperature (°C)
    (20, 90),    # soil_humidity (%)
    (10, 45),    # air_temp (°C)
    (0, 12),     # wind_speed (m/s)
    (20, 100),   # air_humidity (%)
    (0, 20),     # wind_gust (m/s)
    (98, 103),   # pressure_kpa
    (5.5, 8.5),  # pH
    (0, 50),     # rainfall (mm/day)
    (5, 150),    # nitrogen (ppm)
    (5, 100)     # phosphorus (ppm)
]

def normalize(features):
    # Debug logging
    print(f"DEBUG: Received features type: {type(features)}")
    print(f"DEBUG: Received features: {features}")
    print(f"DEBUG: Features length: {len(features) if hasattr(features, '__len__') else 'N/A'}")
    
    features = np.array(features, dtype=np.float32)
    
    print(f"DEBUG: Array shape: {features.shape}")
    print(f"DEBUG: Expected RANGES length: {len(RANGES)}")

    if len(features) != len(RANGES):
        raise ValueError(
            f"Invalid sensor feature count. Expected {len(RANGES)}, got {len(features)}"
        )

    norm = []
    for val, (lo, hi) in zip(features, RANGES):
        val = np.clip(val, lo, hi)
        norm.append((val - lo) / (hi - lo))

    return np.array(norm, dtype=np.float32)