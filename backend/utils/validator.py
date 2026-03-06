
# backend/utils/validator.py

def validate_sensor_data(features):
    """
    Checks if sensor data is within physically meaningful ranges.
    Returns (is_valid, reason)
    """
    # Features order:
    # 0: soil_moisture (0-100)
    # 1: temperature (-10 to 60)
    # 2: soil_humidity (0-100)
    # 3: hour (0-24)
    # 4: dayofyear (1-366)
    # 5: air_temp (-10 to 60)
    # 6: air_humidity (0-100)
    # 7: rainfall (0 to 5000 approx, but definitely not negative)
    # 8: ph (0-14)
    # 9: nitrogen (0-500)
    # 10: phosphorus (0-500)
    # 11: potassium (0-500)

    ranges = [
        (0, 100, "Soil Moisture"),
        (-20, 60, "Temperature"),
        (0, 100, "Soil Humidity"),
        (0, 24, "Time of Day"),
        (0, 366, "Day of Year"),
        (-20, 60, "Air Temperature"),
        (0, 100, "Air Humidity"),
        (0, 2000, "Rainfall"),
        (0, 14, "pH Level"),
        (0, 500, "Nitrogen"),
        (0, 500, "Phosphorus"),
        (0, 500, "Potassium")
    ]

    for i, (min_val, max_val, name) in enumerate(ranges):
        # Optimization: Allow partial lists (validate what is present)
        if i >= len(features):
            break
            
        val = features[i]
        if val < min_val or val > max_val:
            return False, f"{name} ({val}) is out of physical range [{min_val}, {max_val}]"

    return True, "Valid"
