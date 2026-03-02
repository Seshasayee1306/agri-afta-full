import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE, "dataset", "irrigation_stage_dataset.csv")
OUT_DIR = os.path.join(BASE, "backend", "stage_models")

FEATURES = [
    "soil_moisture", "temperature", "soil_humidity", "air_temp", "air_humidity",
    "rainfall", "ph", "nitrogen", "phosphorus", "potassium"
]
TARGET = "needs_water"
STAGE = "growth_stage"
MIN_STAGE_ROWS = 500

def sanitize(df):
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET, STAGE]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])]
    return df

def balance_binary(df):
    c0 = df[df[TARGET] == 0]
    c1 = df[df[TARGET] == 1]
    if len(c0) == 0 or len(c1) == 0:
        return df
    n = min(len(c0), len(c1))
    return pd.concat([c0.sample(n, random_state=42), c1.sample(n, random_state=42)]).sample(frac=1, random_state=42)

def fit_model(train_df):
    X, y = train_df[FEATURES].values, train_df[TARGET].values
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(rf, grid, n_iter=20, scoring="f1", cv=cv, random_state=42, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_

def best_threshold(model, Xv, yv):
    probs = model.predict_proba(Xv)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.2, 0.8, 61):
        pred = (probs >= t).astype(int)
        f1 = f1_score(yv, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def save_artifact(model, threshold, stage, y_true, y_pred, out_path):
    artifact = {
        "model": model,
        "threshold": threshold,
        "features": FEATURES,
        "stage": stage,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        },
    }
    joblib.dump(artifact, out_path)
    print(f"saved: {out_path} -> {artifact['metrics']} threshold={threshold:.2f}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = sanitize(pd.read_csv(DATASET))
    print("stage counts:", df[STAGE].value_counts().to_dict())

    for stage in sorted(df[STAGE].unique()):
        d = df[df[STAGE] == stage].copy()
        if len(d) < MIN_STAGE_ROWS:
            print(f"skip {stage}: only {len(d)} rows")
            continue
        d = balance_binary(d)
        tr, va = train_test_split(d, test_size=0.2, random_state=42, stratify=d[TARGET])
        model = fit_model(tr)
        Xv, yv = va[FEATURES].values, va[TARGET].values
        t = best_threshold(model, Xv, yv)
        pred = (model.predict_proba(Xv)[:, 1] >= t).astype(int)
        save_artifact(model, t, stage, yv, pred, os.path.join(OUT_DIR, f"{stage}_model.pkl"))

    g = balance_binary(df)
    tr, va = train_test_split(g, test_size=0.2, random_state=42, stratify=g[TARGET])
    model = fit_model(tr)
    Xv, yv = va[FEATURES].values, va[TARGET].values
    t = best_threshold(model, Xv, yv)
    pred = (model.predict_proba(Xv)[:, 1] >= t).astype(int)
    save_artifact(model, t, "global", yv, pred, os.path.join(OUT_DIR, "global_model.pkl"))

if __name__ == "__main__":
    main()
