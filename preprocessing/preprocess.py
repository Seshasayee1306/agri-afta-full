# preprocessing/preprocess.py
import pandas as pd
import numpy as np
import os

IN = "../dataset/tarp.csv"  # when running from repo root: python preprocessing/preprocess.py
OUT = "../dataset/irrigation_dataset.csv"

FEATURES = [
    "Soil Moisture","Temperature"," Soil Humidity","Time","Air temperature (C)",
    "Wind speed (Km/h)","Air humidity (%)","Wind gust (Km/h)","Pressure (KPa)",
    "ph","rainfall","N","P","K"
]

def safe_series(df, col, fallback=None):
    # returns a pd.Series guaranteed
    if col in df.columns:
        return df[col]
    if fallback and fallback in df.columns:
        return df[fallback]
    return pd.Series([np.nan]*len(df), index=df.index)

def parse_time(df):
    col = "Time"
    if col in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
            df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
            df['dayofyear'] = df['timestamp'].dt.dayofyear.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
            df['dayofyear'] = 0
    else:
        df['hour'] = 0
        df['dayofyear'] = 0
    return df

def create_target(df):
    # Standardize column names to simple ones for pipeline
    df = df.rename(columns={
        "Air temperature (C)": "air_temp",
        "Air humidity (%)": "air_humidity",
        "Soil Moisture": "soil_moisture",
        " Temperature": "temperature",
        "Temperature": "temperature",
        " Soil Humidity": "soil_humidity",
        "Wind speed (Km/h)": "wind_speed",
        "Wind gust (Km/h)": "wind_gust",
        "Pressure (KPa)": "pressure_kpa",
        "ph": "ph",
        "rainfall": "rainfall",
        "N": "nitrogen",
        "P": "phosphorus",
        "K": "potassium"
    })

    # Ensure required numeric columns exist and are float
    numeric_cols = ["soil_moisture","temperature","soil_humidity","air_temp","air_humidity","rainfall","ph","nitrogen","phosphorus","potassium"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # simple target rule: needs_water if soil_moisture < 35 and rainfall small OR temp high and low humidity
    df['needs_water'] = (
        ((df['soil_moisture'] < 35) & (df['rainfall'].fillna(0) < 1)) |
        ((df['temperature'] > 33) & (df['air_humidity'].fillna(100) < 45))
    ).astype(int)

    # if dataset has Status column, create multi-label mapping too
    if 'Status' in df.columns:
        df['status_label'] = df['Status'].astype(str)

    # synthesize client_id if not present
    if 'client_id' not in df.columns:
        # group by location if exists, else random partition
        np.random.seed(42)
        df['client_id'] = np.random.randint(0, 5, size=len(df))

    # add time features
    df = parse_time(df)

    return df

def main():
    # locate dataset relative to this file
    filepath = os.path.join(os.path.dirname(__file__), IN)
    df = pd.read_csv(filepath)
    df = create_target(df)
    outpath = os.path.join(os.path.dirname(__file__), OUT)
    df.to_csv(outpath, index=False)
    print("Wrote", outpath)
    print("Columns:", df.columns.tolist())
    print("Example rows:\n", df.head().to_string())

if __name__ == "__main__":
    main()
