import pandas as pd
import numpy as np
from backend.stage_engine import calculate_days_after_sowing, identify_growth_stage

# If dataset has no sowing_date, simulate it
def generate_sowing_dates(df):
    df["sowing_date"] = pd.to_datetime("2024-01-01")
    df["current_date"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="D"
    )
    return df


def add_growth_stage(df):

    if "sowing_date" not in df.columns:
        df = generate_sowing_dates(df)

    df["days_after_sowing"] = (
        pd.to_datetime(df["current_date"]) -
        pd.to_datetime(df["sowing_date"])
    ).dt.days

    df["growth_stage"] = df["days_after_sowing"].apply(
        lambda x: identify_growth_stage(x)
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv("dataset/irrigation_dataset.csv")
    df = add_growth_stage(df)

    df.to_csv("dataset/irrigation_stage_dataset.csv", index=False)
    print("✅ Stage-aware dataset created.")