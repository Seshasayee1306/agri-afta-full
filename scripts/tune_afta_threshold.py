import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AFTA_FEATURES = [
    "soil_moisture",
    "temperature",
    "soil_humidity",
    "hour",
    "dayofyear",
    "air_temp",
    "air_humidity",
    "rainfall",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Tune AFTA decision threshold on validation split.")
    ap.add_argument("--dataset", default=str(REPO_ROOT / "dataset" / "irrigation_stage_dataset.csv"))
    ap.add_argument("--model", default=str(REPO_ROOT / "backend" / "final_model.pkl"))
    ap.add_argument("--output-model", default=str(REPO_ROOT / "backend" / "final_model_tuned.pkl"))
    ap.add_argument("--split", choices=["random", "time", "client"], default="client")
    ap.add_argument("--objective", choices=["balanced_accuracy", "f1", "accuracy"], default="balanced_accuracy")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _metric(y_true, y_pred, name: str) -> float:
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    return float(accuracy_score(y_true, y_pred))


def split_test(df: pd.DataFrame, split: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "random":
        train_val, test = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["needs_water"].astype(int).values,
        )
        return train_val.reset_index(drop=True), test.reset_index(drop=True)
    if split == "time":
        d2 = df.copy()
        d2["timestamp"] = pd.to_datetime(d2["timestamp"], errors="coerce")
        d2 = d2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        cut = int(len(d2) * 0.8)
        return d2.iloc[:cut].reset_index(drop=True), d2.iloc[cut:].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    groups = df["client_id"].astype(str).values
    tr_idx, te_idx = next(splitter.split(df, groups=groups))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def run() -> None:
    args = parse_args()
    from backend.model_loader import ModelWrapper

    model_path = Path(args.model)
    output_path = Path(args.output_model)

    df = pd.read_csv(args.dataset)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in AFTA_FEATURES + ["needs_water"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    required_cols = set(AFTA_FEATURES + ["needs_water"])
    if args.split == "time":
        required_cols.add("timestamp")
    if args.split == "client":
        required_cols.add("client_id")
    df = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    df["needs_water"] = df["needs_water"].astype(int).clip(0, 1)

    train_val, test = split_test(df, split=args.split, seed=args.seed)
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=args.seed,
        stratify=train_val["needs_water"].astype(int).values,
    )

    model = ModelWrapper(str(model_path.resolve()))
    X_val = val[AFTA_FEATURES].fillna(0).values.astype(np.float32)
    y_val = val["needs_water"].astype(int).values
    p_val = np.asarray(model.predict_proba(X_val), dtype=np.float32).ravel()

    best_thr = 0.5
    best_score = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (p_val >= t).astype(int)
        score = _metric(y_val, pred, args.objective)
        if score > best_score:
            best_score = score
            best_thr = float(t)

    X_test = test[AFTA_FEATURES].fillna(0).values.astype(np.float32)
    y_test = test["needs_water"].astype(int).values
    p_test = np.asarray(model.predict_proba(X_test), dtype=np.float32).ravel()
    pred_05 = (p_test >= 0.5).astype(int)
    pred_tuned = (p_test >= best_thr).astype(int)

    def summarize(y, p, name):
        return {
            "name": name,
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "pred_pos_rate": float(np.mean(p)),
        }

    b = summarize(y_test, pred_05, "threshold_0.5")
    t = summarize(y_test, pred_tuned, f"threshold_{best_thr:.2f}")

    artifact = joblib.load(str(model_path.resolve()))
    artifact["threshold"] = float(best_thr)
    md = artifact.get("metadata", {})
    md.update(
        {
            "threshold_tuned_on": str(args.dataset),
            "threshold_tuning_split": args.split,
            "threshold_objective": args.objective,
            "threshold_seed": int(args.seed),
            "threshold_value": float(best_thr),
        }
    )
    artifact["metadata"] = md
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, str(output_path.resolve()))

    print(f"Tuned threshold: {best_thr:.4f} (objective={args.objective}, split={args.split})")
    print("Before:", b)
    print("After :", t)
    print(f"Saved tuned model: {output_path.resolve()}")


if __name__ == "__main__":
    run()
