import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


AFTA_FEATURES_ORDER = [
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

STAGE_RF_FEATURES = [
    "soil_moisture",
    "temperature",
    "soil_humidity",
    "air_temp",
    "air_humidity",
    "rainfall",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
]


@dataclass(frozen=True)
class Metrics:
    acc: float
    bal_acc: float
    precision: float
    recall: float
    f1: float
    cm: np.ndarray
    pos_rate: float
    pred_pos_rate: float


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return Metrics(
        acc=float(accuracy_score(y_true, y_pred)),
        bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        cm=confusion_matrix(y_true, y_pred),
        pos_rate=float(np.mean(y_true)),
        pred_pos_rate=float(np.mean(y_pred)),
    )


def _print_metrics(title: str, m: Metrics):
    maj = max(m.pos_rate, 1.0 - m.pos_rate)
    print(f"\n== {title} ==")
    print(f"rows: (see split output)")
    print(f"positive_rate: {m.pos_rate:.4f} | majority_acc: {maj:.4f}")
    print(
        "acc: {acc:.4f} | bal_acc: {bal_acc:.4f} | prec: {precision:.4f} | rec: {recall:.4f} | f1: {f1:.4f}".format(
            acc=m.acc,
            bal_acc=m.bal_acc,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
        )
    )
    print(f"pred_positive_rate: {m.pred_pos_rate:.4f}")
    print("confusion_matrix [[tn fp],[fn tp]]:")
    print(m.cm)


def _load_df(path: str, max_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _coerce_label(df: pd.DataFrame, col: str = "needs_water") -> pd.Series:
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    y = y.where(y.isin([0, 1]), 0)
    return y


def _afta_predict(df: pd.DataFrame, model_path: str) -> np.ndarray:
    from backend.model_loader import ModelWrapper

    X = df[AFTA_FEATURES_ORDER].fillna(0).values.astype(np.float32)
    resolved = model_path
    if not os.path.isabs(resolved):
        resolved = os.path.join(REPO_ROOT, resolved)
    model = ModelWrapper(resolved)
    probs = np.asarray(model.predict_proba(X), dtype=np.float32).ravel()
    return (probs >= 0.5).astype(int)


def _stage_rf_predict(df_stage: pd.DataFrame, stage_models_dir: str) -> np.ndarray:
    import joblib

    preds = np.zeros(len(df_stage), dtype=int)
    for stage in df_stage["growth_stage"].dropna().unique():
        path = f"{stage_models_dir}/{stage}_model.pkl"
        mdl = joblib.load(path)
        idx = df_stage["growth_stage"] == stage
        X = df_stage.loc[idx, STAGE_RF_FEATURES].fillna(0).values
        preds[idx.values] = mdl.predict(X).astype(int)
    return preds


def _split_random(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=_coerce_label(df).values if "needs_water" in df.columns else None,
    )
    return train, test


def _split_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing; cannot do time split")
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
    df2 = df.copy()
    df2["timestamp"] = dt
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df2) * 0.8)
    return df2.iloc[:cut], df2.iloc[cut:]


def _split_client(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "client_id" not in df.columns:
        raise ValueError("client_id column missing; cannot do group split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    groups = df["client_id"].fillna(0).values
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


def eval_irrigation_dataset(
    dataset_path: str,
    model_path: str,
    split: str,
    seed: int,
    max_rows: Optional[int],
):
    df = _load_df(dataset_path, max_rows)
    missing = [c for c in (AFTA_FEATURES_ORDER + ["needs_water"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_path}: {missing}")

    if split == "random":
        _, test = _split_random(df, seed)
    elif split == "time":
        _, test = _split_time(df)
    elif split == "client":
        _, test = _split_client(df, seed)
    else:
        raise ValueError(f"Unknown split: {split}")

    y = _coerce_label(test).values
    p = _afta_predict(test, model_path)
    print(f"\n[AFTA] split={split} test_rows={len(test)}")
    _print_metrics("AFTA (final_model.pkl)", _compute_metrics(y, p))


def eval_stage_dataset(
    dataset_path: str,
    model_path: str,
    stage_models_dir: str,
    split: str,
    seed: int,
    max_rows: Optional[int],
):
    df = _load_df(dataset_path, max_rows)
    required = ["growth_stage", "needs_water"] + STAGE_RF_FEATURES + AFTA_FEATURES_ORDER
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_path}: {missing}")

    if split == "random":
        _, test = _split_random(df, seed)
    elif split == "time":
        # stage dataset has timestamps too in this repo; use if present
        _, test = _split_time(df)
    elif split == "client":
        _, test = _split_client(df, seed)
    else:
        raise ValueError(f"Unknown split: {split}")

    y = _coerce_label(test).values

    rf_p = _stage_rf_predict(test, stage_models_dir)
    print(f"\n[Stage RF] split={split} test_rows={len(test)}")
    _print_metrics("Stage RF (stage_models/*.pkl)", _compute_metrics(y, rf_p))

    afta_p = _afta_predict(test, model_path)
    ens_or = ((rf_p + afta_p) >= 1).astype(int)
    _print_metrics("Ensemble OR(stage_rf, afta)", _compute_metrics(y, ens_or))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "time", "client", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--irrigation-ds", default="dataset/irrigation_dataset.csv")
    ap.add_argument("--stage-ds", default="dataset/irrigation_stage_dataset.csv")
    ap.add_argument("--afta-model", default="backend/final_model.pkl")
    ap.add_argument("--stage-models-dir", default="backend/stage_models")
    args = ap.parse_args()

    splits: List[str] = ["random", "time", "client"] if args.split == "all" else [args.split]

    for s in splits:
        try:
            eval_irrigation_dataset(
                dataset_path=args.irrigation_ds,
                model_path=args.afta_model,
                split=s,
                seed=args.seed,
                max_rows=args.max_rows,
            )
        except Exception as e:
            print(f"\n[AFTA] split={s} skipped: {e}")

        try:
            eval_stage_dataset(
                dataset_path=args.stage_ds,
                model_path=args.afta_model,
                stage_models_dir=args.stage_models_dir,
                split=s,
                seed=args.seed,
                max_rows=args.max_rows,
            )
        except Exception as e:
            print(f"\n[Stage] split={s} skipped: {e}")


if __name__ == "__main__":
    main()
