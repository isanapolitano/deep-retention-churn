# 🏦 Deep Retention: Sistema de Retención Selectiva de Clientes

Deep Retention es un sistema que estima, con una **red neuronal (Perceptrón Multicapa)**, la probabilidad de que un cliente abandone el banco. Con ese dato, un **motor financiero** decide si conviene invertir en retenerlo. El incentivo solo se autoriza cuando el cliente representa una ganancia neta positiva después de descontar el costo del beneficio.

📄 **[Informe Ejecutivo (PDF)](reports/Informe_Ejecutivo_Deep_Retention.pdf)**

🌐 **Aplicación en línea:** _[pendiente: enlace de Streamlit Community Cloud]_

---

## Resultados principales

| Indicador | Valor |
|---|---|
| Clientes analizados | 10.000 |
| Capital en riesgo crítico (P ≥ 0,65) | $69,5 M (832 clientes) |
| Capital rescatable estimado (clientes rentables a intervenir) | $201,0 M (1.668 clientes) |
| Inversión en fidelización | $471.110 |
| Beneficio neto esperado | $1.161.729 |
| ROI | 246,6 % |
| ROC-AUC de la red (prueba) | 0,854 (regresión logística: 0,764) |
| Precision-Recall AUC (prueba) | 0,680 (azar: 0,204) |
| Error de calibración (ECE) | 0,018 (sin calibrar: 0,158) |

Frente a una estrategia indiscriminada (incentivar a todo cliente con P ≥ 0,50), el sistema genera **$320.415 más de beneficio neto** y elimina las **352 intervenciones que dan pérdida**.

## La regla de negocio

```
Ganancia Neta Esperada = (P(fuga) × Efectividad del Incentivo × Margen del Cliente) − Costo del Incentivo
```

Para cada cliente se evalúan los incentivos del catálogo a los que es elegible y se elige el de mayor ganancia. **La acción solo se ejecuta si la ganancia es mayor a cero.** No se usa un umbral fijo de 0,5.

| Segmento | Decisión | Incentivo |
|---|---|---|
| A: Alto Valor / Alto Riesgo | Retención Prioritaria | Gestor VIP y Tarifa Plana |
| B: Medio / Riesgo Moderado | Retención Automatizada | Exoneración de Comisiones o Bonificación Transaccional |
| C: No Rentable | Salida Permitida | Sin intervención (gasto bloqueado) |
| D: Estable (P < 0,30) | Sin Acción Requerida | Mantenimiento ordinario |

## Decisiones técnicas relevantes

- **Control de fuga de datos.** La columna `Complain` coincide con `Exited` en el 99,86 % de los casos. Con ella, una regresión logística llega a un ROC-AUC de 0,999. Como el reclamo suele registrarse durante la salida del cliente, se **excluyó del modelo**.
- **Preprocesamiento sin fuga.** `StandardScaler` y `OneHotEncoder` se ajustan solo con el conjunto de entrenamiento. La división es estratificada: 70 % entrenamiento, 15 % validación y 15 % prueba.
- **Desbalance.** Se usa `class_weight` balanceado en Keras.
- **Regularización.** Batch Normalization, Dropout (0,3 / 0,2), L2 y Early Stopping sobre `val_loss` con paciencia de 15 épocas.
- **Calibración.** La ponderación de clases infla las probabilidades y la fórmula financiera las usa directamente, así que se aplica un **escalado de Platt** ajustado con el conjunto de validación.
- **Explicabilidad.** Se calculan valores **SHAP** sobre las variables originales. Se usan en el informe y también dentro de la aplicación, para explicar cada cliente.

Arquitectura: `Entrada (19) → Dense 128 + BatchNorm + ReLU + Dropout 0,3 → Dense 64 + ReLU + Dropout 0,2 → Dense 1 (Sigmoide)`, optimizador Adam.

## Estructura del repositorio

```
deep-retention-churn/
├── app/
│   └── streamlit_app.py        # Aplicación interactiva
├── data/raw/
│   └── Customer-Churn-Records.csv
├── models/                     # Artefactos entrenados (listos para usar)
│   ├── churn_mlp.keras         # Red neuronal completa
│   ├── churn_mlp.weights.h5    # Pesos del modelo
│   ├── preprocessor.joblib     # Escalado + codificación
│   ├── calibrator.json         # Parámetros de calibración de Platt
│   ├── metadata.json           # Métricas, hiperparámetros y versiones
│   ├── shap_importance.csv     # Importancia de variables
│   ├── shap_background.csv     # Muestra de referencia para SHAP
│   └── portfolio_scored.csv    # Cartera con su probabilidad de fuga
├── reports/
│   ├── Informe_Ejecutivo_Deep_Retention.pdf
│   └── figures/                # Gráficos del informe
├── src/
│   ├── config.py               # Rutas, variables e hiperparámetros
│   ├── data_prep.py            # Carga, validación y preprocesamiento
│   ├── model.py                # Arquitectura del MLP
│   ├── train.py                # Canal de entrenamiento
│   ├── inference.py            # Canal de inferencia (separado del entrenamiento)
│   ├── financial.py            # Margen, incentivos y regla de decisión
│   ├── plots.py                # Estilo de gráficos
│   └── report.py               # Generación del informe PDF
└── requirements.txt
```

## Cómo ejecutarlo

Requisitos: Python 3.10 – 3.13.

```bash
git clone https://github.com/isanapolitano/deep-retention-churn.git
cd deep-retention-churn

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
```

**1. Abrir la aplicación** (los modelos ya vienen entrenados en `/models`):

```bash
streamlit run app/streamlit_app.py
```

**2. Reentrenar la red (opcional, tarda unos 2 minutos en CPU):**

```bash
python -m src.train
```

Este paso regenera los artefactos en `/models` y las figuras en `/reports/figures`. Las semillas están fijadas, por lo que los resultados son reproducibles.

**3. Regenerar el informe PDF (opcional):**

```bash
python -m src.report
```

Todos los comandos se ejecutan desde la raíz del repositorio. Las rutas son relativas, así que no hace falta editar nada.

## Módulos de la aplicación

1. **Buscador por Cliente:** consulta por ID o apellido, o con un formulario para clientes nuevos. Muestra la probabilidad de la red, el margen proyectado, el cálculo de la fórmula para cada incentivo, la decisión sugerida y la explicación SHAP del caso.
2. **Panel de Cartera:** capital en riesgo, capital rescatable, inversión, ROI, efectividad proyectada, matriz de decisión y lista de trabajo descargable.
3. **Simulador de Parámetros:** permite ajustar costos y efectividad de las campañas, margen de interés, horizonte, etc. Compara el escenario actual con el base y con una estrategia indiscriminada, e incluye un análisis de sensibilidad.
4. **Evaluación por Lote:** carga de un CSV con clientes nuevos, validación de datos y descarga de resultados.
5. **Modelo y Explicabilidad:** métricas, arquitectura y gráficos de diagnóstico.

## Supuestos financieros

El dataset no trae margen por cliente ni costos de incentivos, así que se definieron supuestos de banca minorista. Todos se pueden modificar en el simulador:

| Concepto | Valor |
|---|---|
| Margen neto de interés sobre el saldo | 1,5 % anual |
| Comisión por producto | $45 anuales |
| Ingreso por tarjeta (Silver / Gold / Platinum / Diamond) | $25 / $50 / $80 / $120 |
| Costo por punto acumulado | $0,10 |
| Costo operativo por cliente | $180 anuales |
| Horizonte de permanencia | 3 años |
| Gestor VIP y Tarifa Plana (saldo ≥ $120.000) | $450 · efectividad 40 % |
| Exoneración de Comisiones (clientes con saldo) | $150 · efectividad 25 % |
| Bonificación Transaccional (clientes con tarjeta) | $80 · efectividad 15 % |

## Datos

Bank Customer Churn (Kaggle), archivo `Customer-Churn-Records.csv`: 10.000 clientes de un banco europeo (Francia, Alemania y España).

## Autor(es)

- Isabella Napolitano
- Manuela Matos

Materia: Inteligencia Artificial y Aprendizaje Automático
