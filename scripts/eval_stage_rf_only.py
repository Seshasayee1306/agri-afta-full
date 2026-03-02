import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "dataset", "irrigation_stage_dataset.csv")
MODEL_DIR = os.path.join(BASE, "backend", "stage_models")

FEATURES = [
    "soil_moisture","temperature","soil_humidity","air_temp","air_humidity",
    "rainfall","ph","nitrogen","phosphorus","potassium"
]
TARGET = "needs_water"

def load_stage_artifact(stage):
    p = os.path.join(MODEL_DIR, f"{stage}_model.pkl")
    if os.path.exists(p):
        return joblib.load(p)
    gp = os.path.join(MODEL_DIR, "global_model.pkl")
    if os.path.exists(gp):
        return joblib.load(gp)
    raise FileNotFoundError(f"Missing model for stage={stage} and global fallback")

def predict_stage_rows(df_stage):
    preds = np.zeros(len(df_stage), dtype=int)
    for stage in df_stage["growth_stage"].dropna().unique():
        idx = df_stage["growth_stage"] == stage
        Xdf = df_stage.loc[idx, FEATURES].fillna(0)

        art = load_stage_artifact(stage)
        if isinstance(art, dict) and "model" in art:
            m = art["model"]
            t = float(art.get("threshold", 0.5))
            p = m.predict_proba(Xdf)[:, 1]
            preds[idx.values] = (p >= t).astype(int)
        else:
            preds[idx.values] = art.predict(Xdf).astype(int)
    return preds

def sanitize(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET, "growth_stage"]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0,1])]
    return df

def eval_split(df, split, seed=42):
    if split == "random":
        _, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[TARGET])
    elif split == "time":
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
        d2 = df.copy()
        d2["timestamp"] = dt
        d2 = d2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        cut = int(len(d2)*0.8)
        test = d2.iloc[cut:]
    elif split == "client":
        if "client_id" not in df.columns:
            raise ValueError("client_id column missing")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        _, test_idx = next(gss.split(df, groups=df["client_id"].fillna(0).values))
        test = df.iloc[test_idx]
    else:
        raise ValueError(split)

    y = test[TARGET].values
    p = predict_stage_rows(test)
    print(f"\n[Stage RF] split={split} rows={len(test)}")
    print("accuracy:", round(accuracy_score(y,p),4))
    print("balanced_accuracy:", round(balanced_accuracy_score(y,p),4))
    print("f1:", round(f1_score(y,p, zero_division=0),4))

if __name__ == "__main__":
    df = sanitize(pd.read_csv(DATASET))
    for s in ["random","time","client"]:
        try:
            eval_split(df, s)
        except Exception as e:
            print(f"[Stage RF] split={s} skipped: {e}")
