import argparse
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.model_loader import ModelWrapper

MAIN_FEATURES = [
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

STAGE_FEATURES = [
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
    "hour",
    "dayofyear",
]

PLOT_COLORS = {
    "main": "#b45309",
    "ensemble": "#1d4ed8",
}

MODEL_LABELS = {
    "main": "Main AFTA",
    "ensemble": "Ensemble",
}

ACTIVE_MODELS = ["main", "ensemble"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality evaluation graphs for irrigation models."
    )
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "dataset" / "irrigation_stage_dataset.csv"),
    )
    parser.add_argument(
        "--main-model",
        default=str(REPO_ROOT / "backend" / "final_model.pkl"),
    )
    parser.add_argument(
        "--stage-model-dir",
        default=str(REPO_ROOT / "backend" / "stage_afta_models"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "backend" / "static" / "paper_graphs"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["all", "random", "time", "client"], default="all")
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob),
        "avg_precision": safe_ap(y_true, y_prob),
    }


def load_data(dataset_path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, nrows=max_rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    if "needs_water" in df.columns:
        df["needs_water"] = pd.to_numeric(df["needs_water"], errors="coerce")
        df = df.dropna(subset=["needs_water"])
        df["needs_water"] = df["needs_water"].astype(int).clip(0, 1)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.dropna(subset=["growth_stage"])
    df = df.reset_index(drop=True)
    return df


def holdout_subset(df: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    if split == "all":
        return df.reset_index(drop=True)
    if split == "random":
        _, test = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
            stratify=df["needs_water"].astype(int).values,
        )
        return test.reset_index(drop=True)
    if split == "time":
        if "timestamp" not in df.columns:
            raise ValueError("timestamp column missing for time split")
        d2 = df.copy()
        d2["timestamp"] = pd.to_datetime(d2["timestamp"], errors="coerce")
        d2 = d2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        cut = int(len(d2) * 0.8)
        return d2.iloc[cut:].reset_index(drop=True)
    if split == "client":
        if "client_id" not in df.columns:
            raise ValueError("client_id column missing for client split")
        d2 = df.dropna(subset=["client_id"]).copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        groups = d2["client_id"].astype(str).values
        _, test_idx = next(splitter.split(d2, groups=groups))
        return d2.iloc[test_idx].reset_index(drop=True)
    raise ValueError(f"Unknown split: {split}")


def predict_all(
    df: pd.DataFrame,
    main_model_path: Path,
    stage_model_dir: Path,
) -> Dict[str, np.ndarray]:
    main_model = ModelWrapper(str(main_model_path))

    main_input = df[MAIN_FEATURES].fillna(0).values.astype(np.float32)
    main_probs = np.asarray(main_model.predict_proba(main_input), dtype=float).ravel()
    main_preds = (main_probs >= float(main_model.threshold)).astype(int)

    stage_probs = np.zeros(len(df), dtype=float)
    stage_thresholds = np.full(len(df), 0.5, dtype=float)
    missing_stages: List[str] = []

    for stage in sorted(df["growth_stage"].dropna().unique()):
        stage_path = stage_model_dir / f"{stage}_afta.pkl"
        idx = df["growth_stage"] == stage
        stage_input = df.loc[idx, STAGE_FEATURES].fillna(0).values.astype(np.float32)
        if not stage_path.exists():
            stage_probs[idx.values] = main_probs[idx.values]
            missing_stages.append(stage)
            continue
        stage_model = ModelWrapper(str(stage_path))
        probs = np.asarray(stage_model.predict_proba(stage_input), dtype=float).ravel()
        stage_probs[idx.values] = probs
        stage_thresholds[idx.values] = float(stage_model.threshold)

    stage_preds = (stage_probs >= stage_thresholds).astype(int)
    ensemble_probs = np.maximum(stage_probs, main_probs)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    if missing_stages:
        print("Missing stage model(s), fallback to main model for:", ", ".join(missing_stages))

    return {
        "main_probs": main_probs,
        "main_preds": main_preds,
        "stage_probs": stage_probs,
        "stage_preds": stage_preds,
        "ensemble_probs": ensemble_probs,
        "ensemble_preds": ensemble_preds,
    }


def plot_overall_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    metrics_to_plot = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, model_key in enumerate(ACTIVE_MODELS):
        color = PLOT_COLORS[model_key]
        vals = metrics_df.loc[model_key, metrics_to_plot].values
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=MODEL_LABELS[model_key], color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Comparison Across Core Metrics")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_overall_metrics.png")
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, preds: Dict[str, np.ndarray], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_key in ACTIVE_MODELS:
        color = PLOT_COLORS[model_key]
        probs = preds[f"{model_key}_probs"]

        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            axes[0].plot(fpr, tpr, linewidth=2.2, color=color, label=f"{MODEL_LABELS[model_key]} (AUC={auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        axes[1].plot(recall, precision, linewidth=2.2, color=color, label=f"{MODEL_LABELS[model_key]} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_roc_pr_curves.png")
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, preds: Dict[str, np.ndarray], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(ACTIVE_MODELS), figsize=(10.5, 4.6), constrained_layout=True)
    if len(ACTIVE_MODELS) == 1:
        axes = [axes]

    for i, model_key in enumerate(ACTIVE_MODELS):
        cm = confusion_matrix(y_true, preds[f"{model_key}_preds"], labels=[0, 1])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax = axes[i]
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]}\n({cm_norm[r, c]:.2f})",
                    ha="center",
                    va="center",
                    color="#111827",
                    fontsize=10,
                )
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Water", "Need Water"])
        ax.set_yticklabels(["No Water", "Need Water"])
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.04, pad=0.02)
    cbar.set_label("Row-normalized ratio")
    fig.suptitle("Confusion Matrices (Counts + Row-normalized Values)")
    fig.savefig(output_dir / "figure_3_confusion_matrices.png")
    plt.close(fig)


def stage_order(df: pd.DataFrame) -> List[str]:
    if "days_after_sowing" in df.columns:
        grp = df.groupby("growth_stage")["days_after_sowing"].median().sort_values()
        return grp.index.tolist()
    return sorted(df["growth_stage"].unique())


def plot_stagewise_f1(df: pd.DataFrame, y_true: np.ndarray, preds: Dict[str, np.ndarray], output_dir: Path) -> pd.DataFrame:
    rows = []
    ordered_stages = stage_order(df)

    for stage in ordered_stages:
        idx = (df["growth_stage"] == stage).values
        y_s = y_true[idx]
        for model_key in ACTIVE_MODELS:
            p_s = preds[f"{model_key}_preds"][idx]
            if len(np.unique(y_s)) < 2:
                f1 = float(f1_score(y_s, p_s, zero_division=0))
                bal_acc = float("nan")
            else:
                f1 = float(f1_score(y_s, p_s, zero_division=0))
                bal_acc = float(balanced_accuracy_score(y_s, p_s))
            rows.append(
                {
                    "growth_stage": stage,
                    "model": model_key,
                    "f1": f1,
                    "balanced_accuracy": bal_acc,
                    "rows": int(idx.sum()),
                }
            )

    stage_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(ordered_stages))

    for model_key in ACTIVE_MODELS:
        color = PLOT_COLORS[model_key]
        sub = stage_df[stage_df["model"] == model_key].set_index("growth_stage").reindex(ordered_stages)
        axes[0].plot(x, sub["f1"], marker="o", linewidth=2.2, color=color, label=MODEL_LABELS[model_key])
        axes[1].plot(
            x,
            sub["balanced_accuracy"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=MODEL_LABELS[model_key],
        )

    axes[0].set_title("Stage-wise F1 Score")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="best")

    axes[1].set_title("Stage-wise Balanced Accuracy")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered_stages, rotation=30, ha="right")
    axes[1].set_xlabel("Growth Stage")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_stagewise_scores.png")
    plt.close(fig)
    return stage_df


def plot_calibration(y_true: np.ndarray, preds: Dict[str, np.ndarray], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key in ACTIVE_MODELS:
        color = PLOT_COLORS[model_key]
        probs = preds[f"{model_key}_probs"]
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.1, color=color, label=MODEL_LABELS[model_key])

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.2, label="Perfect calibration")
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_5_calibration.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    dataset_path = Path(args.dataset)
    main_model_path = Path(args.main_model)
    stage_model_dir = Path(args.stage_model_dir)
    output_dir = Path(args.output_dir)
    if args.output_tag:
        output_dir = output_dir / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(dataset_path, args.max_rows, args.seed)
    df = holdout_subset(df, split=args.split, seed=args.seed)
    y_true = df["needs_water"].astype(int).values

    print("Running predictions...")
    preds = predict_all(df, main_model_path, stage_model_dir)

    print("Computing metrics...")
    metrics_rows = []
    for model_key in ACTIVE_MODELS:
        y_pred = preds[f"{model_key}_preds"]
        y_prob = preds[f"{model_key}_probs"]
        row = compute_metrics(y_true, y_pred, y_prob)
        row["model"] = model_key
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    metrics_df.to_csv(output_dir / "overall_metrics.csv")

    print("Generating figures...")
    plot_overall_metrics(metrics_df, output_dir)
    plot_roc_pr_curves(y_true, preds, output_dir)
    plot_confusion_matrices(y_true, preds, output_dir)
    stage_df = plot_stagewise_f1(df, y_true, preds, output_dir)
    stage_df.to_csv(output_dir / "stagewise_metrics.csv", index=False)
    plot_calibration(y_true, preds, output_dir)

    with (output_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Generated paper figures:\n")
        f.write(f"split={args.split}\n")
        f.write("1) figure_1_overall_metrics.png\n")
        f.write("2) figure_2_roc_pr_curves.png\n")
        f.write("3) figure_3_confusion_matrices.png\n")
        f.write("4) figure_4_stagewise_scores.png\n")
        f.write("5) figure_5_calibration.png\n")
        f.write("Also saved: overall_metrics.csv, stagewise_metrics.csv\n")

    print("\nDone. Outputs saved to:", output_dir)
    print(metrics_df.round(4))


if __name__ == "__main__":
    main()
