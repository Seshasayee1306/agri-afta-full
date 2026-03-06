#!/usr/bin/env python3
"""
Train a compact edge AFTA model and export assets for ESP32 (TFLite Micro).

Outputs (default to ./esp32):
- afta_edge_model.tflite
- model_data.h
- afta_feature_stats.h
- afta_edge_metadata.json

Usage:
  python3 scripts/train_edge_afta_tflite.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
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
TARGET = "needs_water"


def _load_dataset(dataset_path: Path, new_labels_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if new_labels_path.exists():
        df_new = pd.read_csv(new_labels_path)
        needed = FEATURES + [TARGET]
        missing_new = [c for c in needed if c not in df_new.columns]
        if not missing_new:
            df = pd.concat([df, df_new[needed]], ignore_index=True)

    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])].copy()

    # Keep behavior aligned with existing pipeline: fill missing feature values with 0.
    df[FEATURES] = df[FEATURES].fillna(0.0)

    return df


def _build_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _representative_dataset(X: np.ndarray) -> Iterable[Tuple[np.ndarray]]:
    # Small representative sample for stable int8 quantization.
    n = min(200, len(X))
    idx = np.linspace(0, len(X) - 1, n, dtype=int) if len(X) > 0 else np.array([], dtype=int)
    for i in idx:
        yield [X[i : i + 1].astype(np.float32)]


def _write_model_header(model_bytes: bytes, out_path: Path) -> None:
    hex_bytes = [f"0x{b:02x}" for b in model_bytes]
    lines = []
    chunk = 12
    for i in range(0, len(hex_bytes), chunk):
        lines.append("  " + ", ".join(hex_bytes[i : i + chunk]))

    content = [
        "#ifndef MODEL_DATA_H",
        "#define MODEL_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        "const unsigned char afta_model_tflite[] = {",
        ",\n".join(lines),
        "};",
        f"const unsigned int afta_model_tflite_len = {len(model_bytes)};",
        "",
        "#endif  // MODEL_DATA_H",
        "",
    ]
    out_path.write_text("\n".join(content), encoding="utf-8")


def _fmt_float_list(arr: np.ndarray) -> str:
    return ", ".join(f"{float(x):.8f}f" for x in arr)


def _write_feature_header(
    means: np.ndarray,
    stds: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    out_path: Path,
) -> None:
    content = [
        "#ifndef AFTA_FEATURE_STATS_H",
        "#define AFTA_FEATURE_STATS_H",
        "",
        "constexpr int AFTA_NUM_FEATURES = 12;",
        f"constexpr float AFTA_FEATURE_MEANS[AFTA_NUM_FEATURES] = {{{_fmt_float_list(means)}}};",
        f"constexpr float AFTA_FEATURE_STDS[AFTA_NUM_FEATURES] = {{{_fmt_float_list(stds)}}};",
        f"constexpr float AFTA_INPUT_SCALE = {input_scale:.10f}f;",
        f"constexpr int AFTA_INPUT_ZERO_POINT = {int(input_zero_point)};",
        f"constexpr float AFTA_OUTPUT_SCALE = {output_scale:.10f}f;",
        f"constexpr int AFTA_OUTPUT_ZERO_POINT = {int(output_zero_point)};",
        "",
        "#endif  // AFTA_FEATURE_STATS_H",
        "",
    ]
    out_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/export edge AFTA TinyML model")
    parser.add_argument("--dataset", default="dataset/irrigation_dataset.csv")
    parser.add_argument("--new-labels", default="dataset/irrigation_new_labels.csv")
    parser.add_argument("--out-dir", default="esp32")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "TensorFlow is required for export. Install with: pip install tensorflow\n"
            f"Import error: {e}"
        )

    root = Path(__file__).resolve().parents[1]
    dataset_path = root / args.dataset
    new_labels_path = root / args.new_labels
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(dataset_path, new_labels_path)

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    means = X_train.mean(axis=0).astype(np.float32)
    stds = X_train.std(axis=0).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)

    X_train_n = (X_train - means) / stds
    X_val_n = (X_val - means) / stds

    model = _build_model(input_dim=len(FEATURES))

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_n,
        y_train,
        validation_data=(X_val_n, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    val_loss, val_acc, val_auc = model.evaluate(X_val_n, y_val, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(X_train_n)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = out_dir / "afta_edge_model.tflite"
    tflite_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    _write_model_header(tflite_model, out_dir / "model_data.h")
    _write_feature_header(
        means=means,
        stds=stds,
        input_scale=float(in_scale),
        input_zero_point=int(in_zp),
        output_scale=float(out_scale),
        output_zero_point=int(out_zp),
        out_path=out_dir / "afta_feature_stats.h",
    )

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_auc": float(val_auc),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zp)},
        "outputs": {
            "tflite": str(tflite_path),
            "model_data_h": str(out_dir / "model_data.h"),
            "feature_stats_h": str(out_dir / "afta_feature_stats.h"),
        },
    }

    metadata_path = out_dir / "afta_edge_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Edge AFTA export complete")
    print(f"- TFLite model: {tflite_path}")
    print(f"- C model header: {out_dir / 'model_data.h'}")
    print(f"- Feature stats header: {out_dir / 'afta_feature_stats.h'}")
    print(f"- Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
