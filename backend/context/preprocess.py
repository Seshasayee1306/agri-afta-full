# backend/context/preprocess.py

import pandas as pd
import numpy as np


def preprocess_context_data(csv_path):
    """
    Preprocess context (region + crop + agronomy) dataset.

    Input  : Raw CSV with region, crop, soil, NDVI, yield, etc.
    Output : X (DataFrame), y (Series)
    """

    # -------------------------------------------------
    # LOAD DATA (USE ARGUMENT, NOT HARDCODED PATH)
    # -------------------------------------------------
    df = pd.read_csv(csv_path)

    # -------------------------------------------------
    # BASIC CLEANUP
    # -------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[
        "region",
        "crop_type",
        "soil_pH",
        "rainfall_mm",
        "sunlight_hours",
        "NDVI_index",
        "yield_kg_per_hectare"
    ])

    # -------------------------------------------------
    # DATE FEATURES → SEASON
    # -------------------------------------------------
    df["sowing_date"] = pd.to_datetime(df["sowing_date"], errors="coerce")
    df["sowing_month"] = df["sowing_date"].dt.month

    def get_season(month):
        if month in [6, 7, 8, 9]:
            return "kharif"
        elif month in [10, 11, 12, 1]:
            return "rabi"
        else:
            return "summer"

    df["season"] = df["sowing_month"].apply(get_season)

    # -------------------------------------------------
    # SELECT CONTEXT FEATURES (NO SENSOR LEAKAGE)
    # -------------------------------------------------
    feature_cols = [
        "region",
        "latitude",
        "longitude",
        "crop_type",
        "soil_pH",
        "rainfall_mm",
        "sunlight_hours",
        "NDVI_index",
        "irrigation_type",
        "fertilizer_type",
        "total_days",
        "season"
    ]

    df = df[feature_cols + ["yield_kg_per_hectare"]]

    # -------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # -------------------------------------------------
    categorical_cols = [
        "region",
        "crop_type",
        "irrigation_type",
        "fertilizer_type",
        "season"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # -------------------------------------------------
    # SPLIT FEATURES / TARGET
    # -------------------------------------------------
    X = df.drop(columns=["yield_kg_per_hectare"])
    y = df["yield_kg_per_hectare"]

    # -------------------------------------------------
    # FINAL SAFETY CHECK
    # -------------------------------------------------
    if X.shape[0] == 0:
        raise ValueError("Context dataset empty after preprocessing")

    print(f"[Context Preprocess] Final shape: X={X.shape}, y={y.shape}")

    return X, y
