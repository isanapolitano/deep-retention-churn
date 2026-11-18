"""Configuración central del proyecto Deep Retention.

Todas las rutas son relativas a la raíz del repositorio, de modo que el
proyecto se puede ejecutar en cualquier equipo sin editar rutas locales.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "churn_mlp.keras"
WEIGHTS_PATH = MODELS_DIR / "churn_mlp.weights.h5"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
SHAP_PATH = MODELS_DIR / "shap_importance.csv"
SCORED_PORTFOLIO_PATH = MODELS_DIR / "portfolio_scored.csv"

SEED = 42

TARGET = "Exited"
ID_COL = "CustomerId"

# Columnas que no aportan información predictiva (identificadores).
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# 'Complain' coincide con 'Exited' en el 99.86 % de los registros: es un
# registro posterior (o simultáneo) a la salida del cliente y provocaría fuga
# de datos (data leakage). Se excluye del modelo de forma deliberada.
LEAKAGE_COLS = ["Complain"]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "Satisfaction Score",
    "Point Earned",
]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "Card Type"]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Nombres en español para la interfaz y el informe.
FEATURE_LABELS = {
    "CreditScore": "Puntaje crediticio",
    "Age": "Edad",
    "Tenure": "Antigüedad (años)",
    "Balance": "Saldo",
    "NumOfProducts": "Número de productos",
    "EstimatedSalary": "Salario estimado",
    "Satisfaction Score": "Satisfacción",
    "Point Earned": "Puntos acumulados",
    "HasCrCard": "Tiene tarjeta de crédito",
    "IsActiveMember": "Cliente activo",
    "Geography": "País",
    "Gender": "Género",
    "Card Type": "Tipo de tarjeta",
}

# Hiperparámetros de entrenamiento.
TRAIN_PARAMS = {
    "test_size": 0.15,
    "val_size": 0.15,
    "batch_size": 64,
    "epochs": 300,
    "learning_rate": 1e-3,
    "patience": 15,
    "units_1": 128,
    "units_2": 64,
    "dropout_1": 0.30,
    "dropout_2": 0.20,
    "l2": 1e-4,
}

# Bandas de riesgo usadas para reportar (no para decidir).
CRITICAL_RISK = 0.65
LOW_RISK = 0.30
