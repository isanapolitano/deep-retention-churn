"""Canal de inferencia: carga los artefactos entrenados y puntúa clientes.

Está separado del entrenamiento: la aplicación solo importa este módulo y
nunca vuelve a entrenar la red.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorflow import keras  # noqa: E402

from src import config  # noqa: E402
from src.data_prep import validate_input  # noqa: E402

BACKGROUND_PATH = config.MODELS_DIR / "shap_background.csv"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Calibración de Platt: corrige el sesgo que introduce la ponderación de clases.

    Al entrenar con class_weight la red sobreestima la probabilidad de fuga.
    Como la fórmula financiera multiplica directamente por esa probabilidad,
    se recalibra con una regresión logística sobre el logit de la salida.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(p_raw).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        z = self.a * _logit(np.asarray(p_raw, dtype=float)) + self.b
        return 1 / (1 + np.exp(-z))

    def save(self, path=config.CALIBRATOR_PATH) -> None:
        path.write_text(json.dumps({"method": "platt", "a": self.a, "b": self.b}, indent=2))

    @classmethod
    def load(cls, path=config.CALIBRATOR_PATH) -> "PlattCalibrator":
        d = json.loads(path.read_text())
        return cls(d["a"], d["b"])


class ChurnPredictor:
    """Envuelve preprocesador + red neuronal + calibrador."""

    def __init__(self, model, preprocessor, calibrator: PlattCalibrator):
        self.model = model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self._explainer = None

    @classmethod
    def load(cls) -> "ChurnPredictor":
        missing = [p for p in (config.MODEL_PATH, config.PREPROCESSOR_PATH, config.CALIBRATOR_PATH)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "No se encontraron los artefactos del modelo. Ejecute primero "
                "'python -m src.train'. Faltan: " + ", ".join(p.name for p in missing)
            )
        model = keras.models.load_model(config.MODEL_PATH)
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        return cls(model, preprocessor, PlattCalibrator.load())

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(validate_input(df)[config.FEATURES]).astype("float32")
        # Llamada directa al modelo (más rápida que model.predict para lotes pequeños,
        # algo clave para que SHAP responda en segundos dentro de la aplicación).
        out = [self.model(X[i:i + 8192], training=False).numpy().ravel()
               for i in range(0, len(X), 8192)]
        return np.concatenate(out) if out else np.array([])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de fuga calibrada, entre 0.00 y 1.00."""
        return self.calibrator.transform(self.predict_raw(df))

    # ------------------------------------------------------------------ SHAP
    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        """Representación numérica de las variables originales (categorías -> código)."""
        cats = {c: list(v) for c, v in zip(config.CATEGORICAL_FEATURES,
                                           self.preprocessor.named_transformers_["cat"].categories_)}
        X = df[config.FEATURES].copy()
        for c in config.CATEGORICAL_FEATURES:
            X[c] = X[c].map({v: i for i, v in enumerate(cats[c])}).fillna(0)
        return X.to_numpy(dtype=float)

    def _decode(self, arr: np.ndarray) -> pd.DataFrame:
        cats = {c: list(v) for c, v in zip(config.CATEGORICAL_FEATURES,
                                           self.preprocessor.named_transformers_["cat"].categories_)}
        X = pd.DataFrame(arr, columns=config.FEATURES)
        for c in config.CATEGORICAL_FEATURES:
            X[c] = [cats[c][int(round(v))] for v in X[c]]
        return X

    def _f(self, arr: np.ndarray) -> np.ndarray:
        return self.predict_proba(self._decode(arr))

    def explainer(self, background: pd.DataFrame | None = None):
        import shap

        if self._explainer is None:
            if background is None:
                background = pd.read_csv(BACKGROUND_PATH)
            masker = shap.maskers.Independent(self._encode(background), max_samples=100)
            self._explainer = shap.PermutationExplainer(self._f, masker, seed=config.SEED)
        return self._explainer

    def explain(self, df: pd.DataFrame, background: pd.DataFrame | None = None,
                n_permutations: int = 5):
        """Valores SHAP (en escala de probabilidad) para uno o varios clientes."""
        exp = self.explainer(background)
        max_evals = n_permutations * (2 * len(config.FEATURES) + 1)
        return exp(self._encode(validate_input(df)), max_evals=max_evals, silent=True)
