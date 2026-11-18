"""Arquitectura del Perceptrón Multicapa (MLP) para estimar la probabilidad de fuga."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers, regularizers  # noqa: E402

from src import config  # noqa: E402


def set_seeds(seed: int = config.SEED) -> None:
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - depende de la versión de TF
        pass


def build_mlp(n_inputs: int, params: dict | None = None) -> keras.Model:
    """MLP de dos capas ocultas con regularización.

    Entrada -> Dense(128, ReLU) + BatchNorm + Dropout(0.3)
            -> Dense(64, ReLU) + Dropout(0.2)
            -> Dense(1, Sigmoide)
    """
    p = {**config.TRAIN_PARAMS, **(params or {})}
    reg = regularizers.l2(p["l2"])
    model = keras.Sequential(
        [
            layers.Input(shape=(n_inputs,), name="entrada"),
            layers.Dense(p["units_1"], kernel_regularizer=reg, name="densa_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Activation("relu", name="relu_1"),
            layers.Dropout(p["dropout_1"], name="dropout_1"),
            layers.Dense(p["units_2"], activation="relu", kernel_regularizer=reg, name="densa_2"),
            layers.Dropout(p["dropout_2"], name="dropout_2"),
            layers.Dense(1, activation="sigmoid", name="salida"),
        ],
        name="deep_retention_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model
