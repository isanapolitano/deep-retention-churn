"""Carga, validación y preprocesamiento de los datos de clientes."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Lee el CSV original y verifica que tenga las columnas esperadas."""
    df = pd.read_csv(path)
    required = set(config.FEATURES + [config.TARGET, config.ID_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el archivo de datos: {sorted(missing)}")
    df = df.drop_duplicates(subset=config.ID_COL).reset_index(drop=True)
    return df


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza registros nuevos antes de la inferencia.

    Lanza ValueError con un mensaje claro si falta una columna o si un valor
    categórico no fue visto durante el entrenamiento.
    """
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    out = df.copy()
    for col in config.NUMERIC_FEATURES + config.BINARY_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[config.NUMERIC_FEATURES + config.BINARY_FEATURES].isna().any().any():
        bad = out[config.NUMERIC_FEATURES + config.BINARY_FEATURES].columns[
            out[config.NUMERIC_FEATURES + config.BINARY_FEATURES].isna().any()
        ].tolist()
        raise ValueError(f"Valores numéricos vacíos o inválidos en: {bad}")
    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).str.strip()
    return out


def split_data(df: pd.DataFrame):
    """Divide en entrenamiento / validación / prueba de forma estratificada."""
    p = config.TRAIN_PARAMS
    X = df[config.FEATURES]
    y = df[config.TARGET].astype(int)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=p["test_size"], stratify=y, random_state=config.SEED
    )
    val_share = p["val_size"] / (1 - p["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_share, stratify=y_tmp, random_state=config.SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """Escalado de variables numéricas y codificación one-hot de categóricas.

    Se ajusta solo con el conjunto de entrenamiento para evitar fuga de datos.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("bin", "passthrough", config.BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def feature_groups(preprocessor: ColumnTransformer) -> dict[str, list[int]]:
    """Mapea cada variable original a sus columnas transformadas (para SHAP)."""
    names = list(preprocessor.get_feature_names_out())
    groups: dict[str, list[int]] = {}
    for feat in config.FEATURES:
        if feat in config.CATEGORICAL_FEATURES:
            idx = [i for i, n in enumerate(names) if n.startswith(f"{feat}_")]
        else:
            idx = [names.index(feat)]
        groups[feat] = idx
    return groups
