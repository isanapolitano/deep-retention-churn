"""Canal de entrenamiento del sistema Deep Retention.

Uso (desde la raíz del repositorio):

    python -m src.train

Genera en /models: la red entrenada (.keras y pesos .h5), el preprocesador,
el calibrador, la importancia SHAP, la cartera puntuada y un archivo de
metadatos con todas las métricas. Las figuras quedan en /reports/figures.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from tensorflow import keras  # noqa: E402

from src import config  # noqa: E402
from src.data_prep import build_preprocessor, load_raw, split_data  # noqa: E402
from src.financial import FinancialParams, evaluate_portfolio, portfolio_kpis  # noqa: E402
from src.inference import BACKGROUND_PATH, ChurnPredictor, PlattCalibrator  # noqa: E402
from src.model import build_mlp, set_seeds  # noqa: E402
from src.plots import SERIES, TEXT_2, apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE: diferencia media ponderada entre probabilidad predicha y tasa real."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def leakage_check(df: pd.DataFrame) -> dict:
    """Demuestra la fuga de datos de 'Complain' con un modelo lineal sencillo."""
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_tr)
    Xte = pre.transform(X_te)
    comp_tr = df.loc[X_tr.index, "Complain"].to_numpy().reshape(-1, 1)
    comp_te = df.loc[X_te.index, "Complain"].to_numpy().reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(np.hstack([Xtr, comp_tr]), y_tr)
    auc_with = roc_auc_score(y_te, lr.predict_proba(np.hstack([Xte, comp_te]))[:, 1])
    return {
        "coincidencia_complain_exited": float((df["Complain"] == df["Exited"]).mean()),
        "roc_auc_logistica_con_complain": float(auc_with),
    }


def main() -> None:
    apply_style()
    set_seeds()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = config.TRAIN_PARAMS

    # 1. Datos -------------------------------------------------------------
    df = load_raw()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"Entrenamiento: {len(X_train)} | Validación: {len(X_val)} | Prueba: {len(X_test)}")
    print(f"Tasa de fuga: {y_train.mean():.3f} (train) / {y_test.mean():.3f} (test)")

    # 2. Preprocesamiento (ajustado solo con entrenamiento) ----------------
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train).astype("float32")
    Xva = pre.transform(X_val).astype("float32")
    Xte = pre.transform(X_test).astype("float32")
    joblib.dump(pre, config.PREPROCESSOR_PATH)

    # 3. Desbalance: ponderación de clases ---------------------------------
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Pesos de clase: {class_weight}")

    # 4. Entrenamiento con Early Stopping ----------------------------------
    model = build_mlp(Xtr.shape[1])
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                          min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        Xtr, y_train.to_numpy(),
        validation_data=(Xva, y_val.to_numpy()),
        epochs=p["epochs"], batch_size=p["batch_size"],
        class_weight=class_weight, callbacks=callbacks, verbose=2,
    )
    model.save(config.MODEL_PATH)
    model.save_weights(config.WEIGHTS_PATH)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    # 5. Calibración (con validación, nunca con prueba) --------------------
    p_val_raw = model.predict(Xva, verbose=0).ravel()
    calibrator = PlattCalibrator().fit(p_val_raw, y_val.to_numpy())
    calibrator.save()

    # 6. Evaluación en prueba ----------------------------------------------
    p_test_raw = model.predict(Xte, verbose=0).ravel()
    p_test = calibrator.transform(p_test_raw)
    yt = y_test.to_numpy()

    base = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_train)
    p_base = base.predict_proba(Xte)[:, 1]

    tn, fp, fn, tp = confusion_matrix(yt, (p_test_raw >= 0.5).astype(int)).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(yt, p_test)),
        "pr_auc": float(average_precision_score(yt, p_test)),
        "brier_sin_calibrar": float(brier_score_loss(yt, p_test_raw)),
        "brier_calibrado": float(brier_score_loss(yt, p_test)),
        "ece_sin_calibrar": expected_calibration_error(yt, p_test_raw),
        "ece_calibrado": expected_calibration_error(yt, p_test),
        "prob_media_predicha": float(p_test.mean()),
        "tasa_fuga_real": float(yt.mean()),
        "recall_umbral_05": float(tp / (tp + fn)),
        "precision_umbral_05": float(tp / (tp + fp)),
        "baseline_logistica_roc_auc": float(roc_auc_score(yt, p_base)),
        "baseline_logistica_pr_auc": float(average_precision_score(yt, p_base)),
        "train_roc_auc": float(roc_auc_score(y_train, calibrator.transform(
            model.predict(Xtr, verbose=0).ravel()))),
    }
    leak = leakage_check(df)
    print(json.dumps(metrics, indent=2))

    # 7. Figuras -------------------------------------------------------------
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ep = np.arange(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["loss"], color=SERIES[0], label="Entrenamiento")
    ax[0].plot(ep, h["val_loss"], color=SERIES[1], label="Validación")
    ax[0].axvline(best_epoch, color=TEXT_2, lw=1, ls="--")
    ax[0].text(best_epoch, ax[0].get_ylim()[1], f" mejor época: {best_epoch}", va="top",
               fontsize=8, color=TEXT_2)
    ax[0].set(title="Pérdida (binary cross-entropy)", xlabel="Época")
    ax[0].legend()
    ax[1].plot(ep, h["roc_auc"], color=SERIES[0], label="Entrenamiento")
    ax[1].plot(ep, h["val_roc_auc"], color=SERIES[1], label="Validación")
    ax[1].axvline(best_epoch, color=TEXT_2, lw=1, ls="--")
    ax[1].set(title="ROC-AUC por época", xlabel="Época")
    ax[1].legend()
    fig.savefig(config.FIGURES_DIR / "curvas_entrenamiento.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    fpr, tpr, _ = roc_curve(yt, p_test)
    fpr_b, tpr_b, _ = roc_curve(yt, p_base)
    ax[0].plot(fpr, tpr, color=SERIES[0], label=f"Red neuronal (AUC {metrics['roc_auc']:.3f})")
    ax[0].plot(fpr_b, tpr_b, color=SERIES[1],
               label=f"Regresión logística (AUC {metrics['baseline_logistica_roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls=":")
    ax[0].set(title="Curva ROC (conjunto de prueba)", xlabel="Tasa de falsos positivos",
              ylabel="Tasa de verdaderos positivos")
    ax[0].legend(loc="lower right")
    pr, rc, _ = precision_recall_curve(yt, p_test)
    pr_b, rc_b, _ = precision_recall_curve(yt, p_base)
    ax[1].plot(rc, pr, color=SERIES[0], label=f"Red neuronal (AP {metrics['pr_auc']:.3f})")
    ax[1].plot(rc_b, pr_b, color=SERIES[1],
               label=f"Regresión logística (AP {metrics['baseline_logistica_pr_auc']:.3f})")
    ax[1].axhline(yt.mean(), color=TEXT_2, lw=1, ls=":")
    ax[1].set(title="Curva Precision-Recall (prueba)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(loc="upper right")
    fig.savefig(config.FIGURES_DIR / "roc_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for probs, color, lab in [(p_test_raw, SERIES[1], "Sin calibrar"), (p_test, SERIES[0], "Calibrada (Platt)")]:
        frac, mean_pred = calibration_curve(yt, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=5, color=color, label=lab)
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls=":", label="Calibración perfecta")
    ax.set(title="Curva de calibración (prueba)", xlabel="Probabilidad predicha",
           ylabel="Tasa real de fuga")
    ax.legend(loc="upper left")
    fig.savefig(config.FIGURES_DIR / "calibracion.png")
    plt.close(fig)

    # 8. Explicabilidad con SHAP -------------------------------------------
    predictor = ChurnPredictor(model, pre, calibrator)
    background = X_train.sample(100, random_state=config.SEED)
    background.to_csv(BACKGROUND_PATH, index=False)
    sample = X_test.sample(600, random_state=config.SEED)
    print("Calculando valores SHAP (puede tardar un par de minutos)...")
    sv = predictor.explain(sample, background=background, n_permutations=10)
    mean_abs = np.abs(sv.values).mean(axis=0)
    imp = (pd.DataFrame({"variable": config.FEATURES, "mean_abs_shap": mean_abs})
           .assign(etiqueta=lambda d: d["variable"].map(config.FEATURE_LABELS))
           .sort_values("mean_abs_shap", ascending=False))
    # Dirección del efecto: correlación entre valor de la variable y su SHAP.
    enc = predictor._encode(sample)
    direction = []
    for i, feat in enumerate(config.FEATURES):
        if feat in config.CATEGORICAL_FEATURES or sample[feat].nunique() <= 5:
            s = pd.Series(sv.values[:, i]).groupby(sample[feat].to_numpy()).mean()
            direction.append(f"mayor riesgo: {s.idxmax()}")
        else:
            c = np.corrcoef(enc[:, i], sv.values[:, i])[0, 1]
            direction.append("↑ valor → ↑ riesgo" if c > 0 else "↑ valor → ↓ riesgo")
    imp["direccion"] = imp["variable"].map(dict(zip(config.FEATURES, direction)))
    imp.to_csv(config.SHAP_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    top = imp.iloc[::-1]
    ax.barh(top["etiqueta"], top["mean_abs_shap"], color=SERIES[0], height=0.6)
    ax.set(title="Importancia de variables (|SHAP| medio)",
           xlabel="Impacto medio en la probabilidad de fuga")
    ax.grid(axis="y", visible=False)
    for y_, v in enumerate(top["mean_abs_shap"]):
        ax.text(v, y_, f" {v:.3f}", va="center", fontsize=7, color=TEXT_2)
    fig.savefig(config.FIGURES_DIR / "shap_importancia.png")
    plt.close(fig)

    sv_named = shap.Explanation(values=sv.values, base_values=sv.base_values, data=enc,
                                feature_names=[config.FEATURE_LABELS[f] for f in config.FEATURES])
    plt.figure()
    shap.plots.beeswarm(sv_named, max_display=13, show=False, plot_size=(7, 4.6))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Valor SHAP (impacto en la probabilidad de fuga)", fontsize=9)
    fig.axes[0].tick_params(axis="y", labelsize=9)
    if len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.set_ylabel("Valor de la variable", fontsize=8)
        cbar.set_yticks(cbar.get_ylim(), ["Bajo", "Alto"], fontsize=8)
    plt.title("Efecto de cada variable por cliente (SHAP)", fontsize=11, fontweight="bold")
    plt.savefig(config.FIGURES_DIR / "shap_beeswarm.png")
    plt.close("all")

    # 9. Cartera puntuada con los supuestos por defecto ---------------------
    proba_all = predictor.predict_proba(df)
    portfolio = df.assign(prob_fuga=proba_all)
    portfolio.to_csv(config.SCORED_PORTFOLIO_PATH, index=False)
    kpis = portfolio_kpis(evaluate_portfolio(df, proba_all, FinancialParams()))

    # 10. Metadatos -----------------------------------------------------------
    meta = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "versiones": {"python": platform.python_version(), "tensorflow": tf.__version__,
                      "scikit-learn": sklearn.__version__, "shap": shap.__version__},
        "datos": {"registros": len(df), "train": len(X_train), "val": len(X_val),
                  "test": len(X_test), "tasa_fuga": float(df[config.TARGET].mean())},
        "variables": config.FEATURES,
        "variables_excluidas": {"identificadores": config.DROP_COLS,
                                "fuga_de_datos": config.LEAKAGE_COLS},
        "fuga_de_datos": leak,
        "hiperparametros": p,
        "class_weight": class_weight,
        "epocas_entrenadas": len(h["loss"]),
        "mejor_epoca": best_epoch,
        "calibrador": {"a": calibrator.a, "b": calibrator.b},
        "metricas_prueba": metrics,
        "kpis_cartera_supuestos_base": kpis,
        "parametros_capas": int(model.count_params()),
    }
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Entrenamiento completo. Artefactos guardados en /models.")
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
