"""
Deterministic, training-derived defaults used to fill missing model inputs.

These values are intentionally NOT "guessed" at runtime. They are fixed
statistics (medians) computed from the existing training datasets in this repo:
- dataset/irrigation_dataset.csv
- dataset/irrigation_stage_dataset.csv
"""

# Medians from dataset/irrigation_dataset.csv (computed locally)
IRRIGATION_DATASET_MEDIANS = {
    "soil_humidity": 45.0,
    "wind_speed": 9.53,
    "wind_gust": 37.24,
    "pressure_kpa": 101.12,
    "rainfall": 94.782,
    "nitrogen": 37.0,
    "phosphorus": 51.0,
}

# Medians from dataset/irrigation_stage_dataset.csv (computed locally)
IRRIGATION_STAGE_DATASET_MEDIANS = {
    "potassium": 32.0,
}


def get_median_defaults():
    return {
        **IRRIGATION_DATASET_MEDIANS,
        **IRRIGATION_STAGE_DATASET_MEDIANS,
    }

