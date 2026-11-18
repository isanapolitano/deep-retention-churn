"""Genera el Informe Ejecutivo en PDF a partir de los artefactos del entrenamiento.

Uso (después de 'python -m src.train'):

    python -m src.report

Todas las cifras se leen de /models, de modo que el informe siempre coincide
con el modelo entrenado.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config
from src.financial import (
    FinancialParams,
    evaluate_portfolio,
    naive_strategy,
    portfolio_kpis,
    segment_matrix,
)
from src.plots import GRID, SERIES, TEXT_2, apply_style

# ------------------------------------------------------------ datos de portada
AUTORES = ["Isabella Napolitano", "Manuela Matos"]
CURSO = "Inteligencia Artificial y Aprendizaje Automático"
REPO_URL = "github.com/isanapolitano/deep-retention-churn"

OUTPUT = config.REPORTS_DIR / "Informe_Ejecutivo_Deep_Retention.pdf"

# --------------------------------------------------------------------- estilo
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                              boldItalic="Body-Bold")

NAVY = colors.HexColor("#14325c")
BLUE = colors.HexColor(SERIES[0])
LIGHT = colors.HexColor("#eef3fa")
MUTED = colors.HexColor(TEXT_2)
RULE = colors.HexColor("#c9d3e0")

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.2,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=15, leading=19, textColor=NAVY,
                         spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=11, leading=15, textColor=NAVY,
                         spaceBefore=8, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=10.6),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.2, leading=10.6),
    "head": ParagraphStyle("head", fontName="Body-Bold", fontSize=8.2, leading=10.6,
                           textColor=colors.white),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=8, leading=10.5,
                              textColor=MUTED, alignment=TA_CENTER, spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.6, leading=14, leftIndent=14,
                             bulletIndent=3, spaceAfter=3, alignment=TA_JUSTIFY),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Body-Bold", fontSize=15, leading=18, textColor=NAVY),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Body", fontSize=7.8, leading=10, textColor=MUTED),
    "formula": ParagraphStyle("formula", fontName="Body-Italic", fontSize=9.6, leading=14,
                              alignment=TA_CENTER, textColor=NAVY, spaceBefore=4, spaceAfter=8),
}


def money(x: float, short: bool = False, dec: int = 1) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if short and x >= 1e6:
        s = f"{x / 1e6:,.{dec}f} M"
    else:
        s = f"{x:,.0f}"
    return sign + "$" + s.replace(",", "X").replace(".", ",").replace("X", ".")


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, d: int = 1) -> str:
    return num(x * 100, d) + " %"


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(data: list[list], widths: list[float], header: bool = True, zebra: bool = True,
          bold_first: bool = False, size: float | None = None) -> Table:
    if size:
        local = {k: ParagraphStyle(f"{k}_{size}", parent=S[k], fontSize=size, leading=size * 1.3)
                 for k in ("cell", "cellb", "head")}
    else:
        local = S
    rows = []
    for i, r in enumerate(data):
        row = []
        for j, c in enumerate(r):
            if isinstance(c, str):
                st = "head" if header and i == 0 else ("cellb" if bold_first and j == 0 else "cell")
                row.append(Paragraph(c, local[st]))
            else:
                row.append(c)
        rows.append(row)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, width: float, caption: str) -> KeepTogether:
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    return KeepTogether([img, P(caption, "caption")])


def kpi_grid(items: list[tuple[str, str]], cols: int = 3) -> Table:
    cells = [[P(v, "kpi_v"), P(l, "kpi_l")] for v, l in items]
    grid = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        row += [[""]] * (cols - len(row))
        grid.append(row)
    w = 17.4 * cm / cols
    t = Table(grid, colWidths=[w] * cols)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0, colors.white),
        ("INNERGRID", (0, 0), (-1, -1), 3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def portfolio_figure(scored: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    order = ["A: Alto Valor / Alto Riesgo", "B: Medio / Riesgo Moderado", "C: No Rentable"]
    labels = ["A · Retención prioritaria", "B · Retención automatizada", "C · Salida permitida"]
    colors_ = [SERIES[0], SERIES[2], SERIES[1]]
    crit = scored[scored["prob_fuga"] >= config.CRITICAL_RISK]
    mod = scored[(scored["prob_fuga"] >= config.LOW_RISK) & (scored["prob_fuga"] < config.CRITICAL_RISK)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.5), sharey=True)
    for a, sub, title in [(ax[0], crit, f"Riesgo crítico (P ≥ 0,65): {len(crit)} clientes"),
                          (ax[1], mod, f"Riesgo moderado (0,30 – 0,65): {len(mod)} clientes")]:
        cnt = sub["segmento"].value_counts().reindex(order).fillna(0)
        bars = a.barh(labels[::-1], cnt.values[::-1], color=colors_[::-1], height=0.55)
        for b, v in zip(bars, cnt.values[::-1]):
            a.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {int(v)}", va="center",
                   fontsize=8, color=TEXT_2)
        a.set_title(title)
        a.set_xlabel("Clientes")
        a.grid(axis="y", visible=False)
        a.set_xlim(0, max(cnt.max() * 1.2, 1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, "Deep Retention · Informe Ejecutivo de Proyecto")
        canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, letter[1] - 1.5 * cm, letter[0] - 2 * cm, letter[1] - 1.5 * cm)
    canvas.restoreState()


def build() -> Path:
    meta = json.loads(config.METADATA_PATH.read_text())
    imp = pd.read_csv(config.SHAP_PATH)
    raw = pd.read_csv(config.SCORED_PORTFOLIO_PATH)
    params = FinancialParams()
    scored = evaluate_portfolio(raw, raw["prob_fuga"].to_numpy(), params)
    k = portfolio_kpis(scored)
    naive = naive_strategy(scored, params)
    mat = segment_matrix(scored)
    mt = meta["metricas_prueba"]
    import joblib

    n_in = len(joblib.load(config.PREPROCESSOR_PATH).get_feature_names_out())
    hp = meta["hiperparametros"]
    fig_dir = config.FIGURES_DIR
    portfolio_figure(scored, fig_dir / "cartera_segmentos.png")

    # Estadísticas descriptivas para los factores de abandono.
    by_prod = raw.groupby("NumOfProducts")["Exited"].mean()
    by_age = raw.groupby(pd.cut(raw["Age"], [17, 30, 40, 50, 60, 100]), observed=True)["Exited"].mean()
    by_act = raw.groupby("IsActiveMember")["Exited"].mean()
    by_geo = raw.groupby("Geography")["Exited"].mean()
    by_gender = raw.groupby("Gender")["Exited"].mean()

    seg_c = scored[scored["segmento"] == "C: No Rentable"]
    ahorro = float(np.where(seg_c["Balance"] > 0, params.fee_waiver_cost,
                            np.where(seg_c["HasCrCard"] == 1, params.bonus_cost,
                                     params.fee_waiver_cost)).sum())
    crit = scored[scored["prob_fuga"] >= config.CRITICAL_RISK]
    crit_seg = crit["segmento"].value_counts()
    inc_b = scored[scored["segmento"] == "B: Medio / Riesgo Moderado"]["incentivo"].value_counts()

    story: list = []

    # ------------------------------------------------------------------ PORTADA
    story += [Spacer(1, 3.2 * cm)]
    story.append(P("<font size=10 color='#52514e'>INFORME EJECUTIVO DE PROYECTO</font>", "body"))
    story.append(Paragraph("Deep Retention", ParagraphStyle(
        "t", fontName="Body-Bold", fontSize=32, leading=38, textColor=NAVY)))
    story.append(Paragraph("Sistema de Retención Selectiva de Clientes basado en Redes Neuronales",
                           ParagraphStyle("st", fontName="Body", fontSize=14, leading=19,
                                          textColor=BLUE, spaceAfter=24)))
    cover = [
        ["De:", "Dirección de Estrategia y Rentabilidad de Clientes"],
        ["Para:", "Equipo de Desarrollo e Inteligencia de Negocios"],
        ["Asunto:", "Especificaciones operativas y resultados del sistema “Deep Retention”"],
        ["Autor(es):", ", ".join(AUTORES)],
        ["Materia:", CURSO],
        ["Fecha:", date.today().strftime("%d/%m/%Y")],
        ["Repositorio:", REPO_URL],
    ]
    story.append(table(cover, [3.2 * cm, 13.6 * cm], header=False, zebra=False, bold_first=True))
    story.append(Spacer(1, 1.2 * cm))
    story.append(kpi_grid([
        (money(k["ganancia_neta"], True), "Ganancia neta esperada de la campaña"),
        (pct(k["roi"], 0), "Retorno sobre la inversión (ROI)"),
        (num(mt["roc_auc"], 3), "ROC-AUC de la red neuronal (conjunto de prueba)"),
    ]))
    story.append(PageBreak())

    # --------------------------------------------- 1. RESUMEN DE IMPACTO FINANCIERO
    story.append(P("1. Resumen de Impacto Financiero y Retorno", "h1"))
    story.append(P(
        "La retención indiscriminada de clientes erosiona el margen operativo, porque destina "
        "presupuesto a usuarios que, aun si se quedan, no generan beneficio para el banco. "
        "Deep Retention resuelve este problema en dos pasos: una red neuronal estima la "
        "probabilidad de que cada cliente abandone la entidad y un motor financiero decide si "
        "vale la pena intervenir. Un incentivo solo se autoriza cuando la ganancia esperada "
        "supera su costo. Sobre la cartera analizada, los resultados son los siguientes:"))
    story.append(kpi_grid([
        (num(k["clientes"]).replace(",", "."), "Volumen total de clientes analizados"),
        (money(k["capital_riesgo_critico"], True),
         f"Capital en riesgo crítico: saldo de {num(k['clientes_riesgo_critico'])} clientes con "
         f"probabilidad de fuga ≥ {num(config.CRITICAL_RISK, 2)}"),
        (money(k["capital_intervenido"], True),
         f"Capital rescatable estimado: saldo de los {num(k['clientes_intervenidos'])} clientes "
         "rentables a intervenir"),
        (money(k["inversion"]), "Inversión total en campañas de fidelización"),
        (money(k["ganancia_neta"]), f"Beneficio neto esperado (margen protegido de "
                                    f"{money(k['margen_protegido'])} menos la inversión)"),
        (pct(k["roi"]), "Retorno sobre la inversión (ROI = beneficio neto / inversión)"),
    ]))
    story.append(Spacer(1, 8))
    story.append(P(
        f"Del capital rescatable, el sistema estima que se conservarán efectivamente "
        f"<b>{money(k['saldo_rescatado_esperado'], True)}</b> en saldos, una vez ponderado por la "
        f"probabilidad de fuga de cada cliente y por la efectividad del incentivo asignado. En "
        f"términos de clientes, se espera evitar alrededor de <b>{num(k['clientes_retenidos_esperados'])}</b> "
        f"de las {num(k['fugas_esperadas_intervenidos'])} salidas proyectadas en el grupo intervenido. "
        f"Por otro lado, <b>{num(k['salida_permitida'])}</b> clientes con riesgo relevante quedaron "
        f"catalogados como <i>Salida Permitida</i>, ya que retenerlos costaría más de lo que aportan."))
    story.append(P("Comparación con una estrategia de retención indiscriminada", "h2"))
    story.append(P(
        "Para dimensionar el aporte del filtro de rentabilidad, se comparó el sistema con la "
        "práctica habitual: ofrecer un incentivo a todo cliente con probabilidad de fuga mayor o "
        "igual a 0,50, sin revisar su margen."))
    story.append(table([
        ["Estrategia", "Clientes intervenidos", "Inversión", "Beneficio neto esperado",
         "ROI", "Intervenciones con pérdida"],
        ["Deep Retention (ganancia neta > 0)", num(k["clientes_intervenidos"]), money(k["inversion"]),
         money(k["ganancia_neta"]), pct(k["roi"], 0), "0"],
        ["Indiscriminada (umbral fijo 0,50)", num(naive["clientes_intervenidos"]),
         money(naive["inversion"]), money(naive["ganancia_neta"]), pct(naive["roi"], 0),
         num(naive["clientes_con_perdida"])],
    ], [4.3 * cm, 2.6 * cm, 2.2 * cm, 2.9 * cm, 1.6 * cm, 3.0 * cm]))
    story.append(Spacer(1, 6))
    story.append(P(
        f"Deep Retention genera <b>{money(k['ganancia_neta'] - naive['ganancia_neta'])}</b> más de "
        f"beneficio neto y elimina las {num(naive['clientes_con_perdida'])} intervenciones que, en la "
        "estrategia indiscriminada, cuestan más de lo que recuperan. El ROI promedio de la estrategia "
        "indiscriminada luce algo mayor porque solo atiende a clientes de riesgo muy alto; sin embargo, "
        "deja fuera a clientes de riesgo moderado que siguen siendo rentables y, al mismo tiempo, "
        "financia a clientes deficitarios. Como la meta de la Junta Directiva es maximizar la "
        "ganancia total, el indicador relevante es el beneficio neto y no el ROI promedio."))
    story.append(PageBreak())

    # ---------------------------------- 2. ARQUITECTURA Y DIAGNÓSTICO DE FUGA
    story.append(P("2. Arquitectura de la Solución y Diagnóstico de Fuga", "h1"))
    story.append(P("Datos y preparación", "h2"))
    story.append(P(
        f"Se trabajó con {num(meta['datos']['registros'])} registros de clientes y una tasa de "
        f"abandono de {pct(meta['datos']['tasa_fuga'])}. Los datos se dividieron de forma "
        f"estratificada en entrenamiento ({num(meta['datos']['train'])}), validación "
        f"({num(meta['datos']['val'])}) y prueba ({num(meta['datos']['test'])}). Las variables "
        "numéricas se estandarizaron con <i>StandardScaler</i> y las categóricas (país, género y tipo "
        "de tarjeta) se codificaron con <i>OneHotEncoder</i>. Ambos transformadores se ajustaron "
        "únicamente con el conjunto de entrenamiento, para evitar que información de validación o "
        "prueba se filtrara al modelo. Los identificadores (número de fila, ID y apellido) se "
        "descartaron por no tener valor predictivo."))
    story.append(P(
        f"<b>Control de fuga de datos.</b> La variable <i>Complain</i> coincide con la variable "
        f"objetivo en el {pct(meta['fuga_de_datos']['coincidencia_complain_exited'], 2)} de los "
        f"casos. Al incluirla, incluso una regresión logística simple alcanza un ROC-AUC de "
        f"{num(meta['fuga_de_datos']['roc_auc_logistica_con_complain'], 3)}, un resultado que no se "
        "reproduciría en operación: el reclamo suele registrarse cuando el cliente ya decidió "
        "irse o durante su salida. Por esta razón se excluyó del modelo. El desempeño reportado "
        "corresponde a un modelo que puede usarse antes de que el cliente abandone."))
    story.append(P("Configuración de la red neuronal", "h2"))
    story.append(table([
        ["Componente", "Configuración"],
        ["Tipo", "Perceptrón Multicapa (MLP) en Keras/TensorFlow"],
        ["Entrada", f"{n_in} columnas (8 numéricas estandarizadas, 2 binarias y {n_in - 10} "
                    "indicadoras one-hot)"],
        ["Capa oculta 1", f"{hp['units_1']} neuronas · ReLU · Batch Normalization · "
                          f"Dropout {num(hp['dropout_1'], 1)} · regularización L2"],
        ["Capa oculta 2", f"{hp['units_2']} neuronas · ReLU · Dropout {num(hp['dropout_2'], 1)}"],
        ["Capa de salida", "1 neurona · Sigmoide (probabilidad entre 0 y 1)"],
        ["Optimizador y pérdida", f"Adam (tasa de aprendizaje {hp['learning_rate']}) · entropía "
                                  "cruzada binaria"],
        ["Desbalance de clases", f"class_weight balanceado (no fuga: "
                                 f"{num(meta['class_weight']['0'], 2)}; fuga: "
                                 f"{num(meta['class_weight']['1'], 2)})"],
        ["Sobreajuste", f"Early Stopping sobre val_loss (paciencia {hp['patience']}); mejor época "
                        f"{meta['mejor_epoca']} de {meta['epocas_entrenadas']}"],
        ["Calibración", "Escalado de Platt ajustado con el conjunto de validación"],
        ["Tamaño", f"{num(meta['parametros_capas'])} parámetros"],
    ], [4.2 * cm, 12.8 * cm], bold_first=True))
    story.append(Spacer(1, 6))
    story.append(P(
        "La ponderación de clases evita que la red aprenda a clasificar a todos como clientes "
        "que se quedan, pero tiene un efecto secundario: infla las probabilidades. Como la fórmula "
        "financiera multiplica directamente por la probabilidad, un valor inflado haría que el "
        "banco gastara de más. Por eso se añadió una etapa de calibración que devuelve las "
        "probabilidades a su escala real."))
    story.append(P("Métricas de diagnóstico técnico (conjunto de prueba)", "h2"))
    story.append(table([
        ["Métrica", "Red neuronal", "Referencia", "Lectura"],
        ["ROC-AUC", num(mt["roc_auc"], 3), f"{num(mt['baseline_logistica_roc_auc'], 3)} "
                                           "(reg. logística)",
         "Capacidad para ordenar a los clientes según su riesgo."],
        ["Precision-Recall AUC", num(mt["pr_auc"], 3),
         f"{num(mt['tasa_fuga_real'], 3)} (azar) · {num(mt['baseline_logistica_pr_auc'], 3)} (reg. log.)",
         "Calidad en la clase minoritaria (clientes que se van)."],
        ["Precisión de calibración", f"{pct(1 - mt['ece_calibrado'])} (ECE {num(mt['ece_calibrado'], 3)})",
         f"{pct(1 - mt['ece_sin_calibrar'])} sin calibrar (ECE {num(mt['ece_sin_calibrar'], 3)})",
         "Qué tanto coincide la probabilidad estimada con la tasa real."],
        ["Brier score", num(mt["brier_calibrado"], 3), f"{num(mt['brier_sin_calibrar'], 3)} sin calibrar",
         "Error cuadrático de la probabilidad (menor es mejor)."],
    ], [3.6 * cm, 3.3 * cm, 4.4 * cm, 5.7 * cm], bold_first=True))
    story.append(Spacer(1, 6))
    story.append(P(
        f"La red supera con claridad al modelo lineal de referencia (+{num((mt['roc_auc'] - mt['baseline_logistica_roc_auc']) * 100, 1)} "
        f"puntos de ROC-AUC), lo que confirma que existen relaciones no lineales que justifican el uso de "
        f"una red neuronal. La diferencia entre el ROC-AUC de entrenamiento ({num(mt['train_roc_auc'], 3)}) "
        f"y el de prueba ({num(mt['roc_auc'], 3)}) es moderada, señal de que el Dropout y el Early "
        f"Stopping contuvieron el sobreajuste. Tras la calibración, la probabilidad media estimada "
        f"({pct(mt['prob_media_predicha'])}) es muy cercana a la tasa real de fuga ({pct(mt['tasa_fuga_real'])})."))
    story.append(figure(fig_dir / "roc_pr.png", 16.5 * cm,
                        "Figura 1. Curvas ROC y Precision-Recall de la red frente a una regresión logística."))
    story.append(figure(fig_dir / "calibracion.png", 8.6 * cm,
                        "Figura 2. Curva de calibración antes y después del escalado de Platt."))
    story.append(figure(fig_dir / "curvas_entrenamiento.png", 16.5 * cm,
                        "Figura 3. Evolución de la pérdida y del ROC-AUC. La línea punteada marca la "
                        "época restaurada por Early Stopping."))

    story.append(P("Factores determinantes de abandono", "h2"))
    story.append(P(
        "Para que las decisiones de la red no sean una caja negra, se calcularon valores SHAP sobre "
        "600 clientes del conjunto de prueba. Cada valor indica cuánto sube o baja la probabilidad "
        "de fuga de un cliente por efecto de una variable. Los tres factores más influyentes son:"))
    top3 = imp.head(3)["etiqueta"].tolist()
    story += bullets([
        f"<b>{top3[0]}.</b> Es la variable con mayor impacto. Su efecto no es lineal: los clientes "
        f"con dos productos son los más estables (tasa de fuga de {pct(by_prod[2])}), los que tienen "
        f"uno solo abandonan con más frecuencia ({pct(by_prod[1])}) y quienes tienen tres o cuatro "
        f"presentan tasas de {pct(by_prod[3], 0)} y {pct(by_prod[4], 0)}. Esto último sugiere "
        "productos vendidos que no se ajustan a la necesidad del cliente.",
        f"<b>{top3[1]}.</b> El riesgo aumenta con la edad hasta alcanzar su punto más alto entre los "
        f"50 y 60 años ({pct(by_age.iloc[3])}), frente a {pct(by_age.iloc[0])} en menores de 30. "
        "Es un segmento que suele trasladar ahorros hacia productos de inversión o jubilación.",
        f"<b>{top3[2]}.</b> Los clientes inactivos abandonan casi el doble que los activos "
        f"({pct(by_act[0])} frente a {pct(by_act[1])}). La inactividad es la señal temprana más útil "
        "para el ejecutivo de cuenta.",
    ])
    story.append(P(
        f"Les siguen el país (Alemania registra {pct(by_geo['Germany'])} de fuga, frente a "
        f"{pct(by_geo['France'])} en Francia y {pct(by_geo['Spain'])} en España) y el género "
        f"({pct(by_gender['Female'])} en mujeres frente a {pct(by_gender['Male'])} en hombres). En el "
        "caso del género, se recomienda que el área de cumplimiento valide su uso, ya que la "
        "normativa local podría limitar el empleo de esta variable en decisiones comerciales. "
        "La satisfacción declarada, la antigüedad y la tenencia de tarjeta tienen un peso marginal."))
    story.append(figure(fig_dir / "shap_importancia.png", 12 * cm,
                        "Figura 4. Importancia global de las variables (promedio del valor SHAP absoluto)."))
    story.append(figure(fig_dir / "shap_beeswarm.png", 14.5 * cm,
                        "Figura 5. Efecto de cada variable cliente por cliente. A la derecha del cero, "
                        "la variable aumenta el riesgo."))
    story.append(PageBreak())

    # ---------------------------------------- 3. MATRIZ DE DECISIÓN COMERCIAL
    story.append(P("3. Matriz de Decisión Comercial y Rentabilidad", "h1"))
    story.append(P(
        "El sistema no utiliza un umbral fijo de 0,50. Para cada cliente se evalúan los incentivos "
        "del catálogo a los que es elegible y se calcula la regla establecida por la Junta Directiva:"))
    story.append(P("Ganancia Neta Esperada = (Probabilidad de Fuga × Efectividad del Incentivo × "
                   "Margen del Cliente) − Costo del Incentivo", "formula"))
    story.append(P(
        f"Se elige el incentivo con la mayor ganancia neta y la acción solo se ejecuta si ese valor es "
        f"mayor a cero. Los clientes con probabilidad inferior a {num(params.min_risk, 2)} permanecen "
        "en mantenimiento ordinario, en línea con el segmento D de la matriz. El margen del cliente "
        f"corresponde al margen neto anual que aporta al banco, proyectado a {num(params.horizon_years)} "
        "años de permanencia (los supuestos se detallan en el Anexo A)."))

    def seg(name, col):
        return mat.loc[name, col] if name in mat.index else 0

    A, B, C, D = ("A: Alto Valor / Alto Riesgo", "B: Medio / Riesgo Moderado", "C: No Rentable",
                  "D: Estable")
    story.append(table([
        ["Segmento", "Probabilidad red", "Valor patrimonial", "Decisión del sistema",
         "Incentivo asignado", "Costo ($)", "Margen<br/>proyectado"],
        [f"<b>A: Alto Valor / Alto Riesgo</b><br/>{num(seg(A, 'Clientes'))} clientes",
         f"≥ {num(params.min_risk, 2)} (media {num(seg(A, 'Prob. media'), 2)})",
         f"Saldo superior (≥ {money(params.high_balance)}; medio {money(seg(A, 'Saldo medio'))})",
         "Retención Prioritaria", "Gestor VIP y Tarifa Plana",
         f"{money(params.vip_cost)} c/u<br/>Total {money(seg(A, 'Costo total'))}",
         f"{money(seg(A, 'Ganancia neta esperada'))} netos"],
        [f"<b>B: Medio / Riesgo Moderado</b><br/>{num(seg(B, 'Clientes'))} clientes",
         f"≥ {num(params.min_risk, 2)} (media {num(seg(B, 'Prob. media'), 2)})",
         f"Saldo medio (medio {money(seg(B, 'Saldo medio'))})",
         "Retención Automatizada",
         f"Exoneración de Comisiones ({num(inc_b.get('Exoneración de Comisiones', 0))}) · "
         f"Bonificación Transaccional ({num(inc_b.get('Bonificación Transaccional', 0))})",
         f"{money(params.fee_waiver_cost)} / {money(params.bonus_cost)} c/u<br/>Total "
         f"{money(seg(B, 'Costo total'))}",
         f"{money(seg(B, 'Ganancia neta esperada'))} netos"],
        [f"<b>C: No Rentable</b><br/>{num(seg(C, 'Clientes'))} clientes",
         f"≥ {num(params.min_risk, 2)} (media {num(seg(C, 'Prob. media'), 2)})",
         f"Saldo bajo o deficitario (margen medio {money(seg(C, 'Margen cliente medio'))})",
         "Salida Permitida", "Sin intervención", "$0",
         f"Ahorro operativo de {money(ahorro)}"],
        [f"<b>D: Estable</b><br/>{num(seg(D, 'Clientes'))} clientes",
         f"&lt; {num(params.min_risk, 2)} (media {num(seg(D, 'Prob. media'), 2)})",
         "Cualquier saldo", "Sin Acción Requerida", "Mantenimiento ordinario", "$0", "Estable"],
    ], [2.9 * cm, 2.3 * cm, 2.8 * cm, 2.4 * cm, 2.9 * cm, 2.1 * cm, 2.0 * cm], size=7.6))
    story.append(Spacer(1, 6))
    story.append(P(
        f"La plantilla original asociaba los segmentos A, B y C a probabilidades iguales o superiores a "
        f"0,65. En este sistema, ese valor se usa para medir el capital en riesgo crítico, pero no "
        f"para decidir: un cliente de saldo alto con probabilidad de 0,40 también puede justificar una "
        f"intervención. Dentro de la banda crítica (≥ 0,65) hay {num(len(crit))} clientes: "
        f"{num(crit_seg.get(A, 0))} quedaron en el segmento A, {num(crit_seg.get(B, 0))} en el B y "
        f"{num(crit_seg.get(C, 0))} en el C. Estos últimos tienen una probabilidad de fuga muy alta, "
        "pero su margen es negativo, por lo que la regla bloquea cualquier gasto de fidelización. El "
        f"{pct((seg_c['margen_cliente'] <= 0).mean(), 0)} del segmento C es deficitario; el resto tiene "
        "un margen tan bajo que ningún incentivo resulta rentable."))
    story.append(figure(fig_dir / "cartera_segmentos.png", 16.5 * cm,
                        "Figura 6. Decisión del sistema según la banda de riesgo."))
    story.append(PageBreak())

    # --------------------------------------------- 4. OPERACIÓN EN SUCURSALES
    story.append(P("4. Operación de la Aplicación en Sucursales", "h1"))
    story.append(P(
        "La aplicación se desarrolló en Streamlit y se organiza en cinco módulos: Buscador por "
        "Cliente, Panel de Cartera, Simulador de Parámetros, Evaluación por Lote y Modelo y "
        "Explicabilidad. No requiere conocimientos técnicos y responde en pocos segundos."))
    story.append(P("Procedimiento del gestor de cuentas", "h2"))
    steps = [
        "<b>Preparar la jornada.</b> En el Panel de Cartera, filtrar por país o segmento y descargar la "
        "lista de trabajo, que ordena a los clientes por ganancia neta esperada. Los primeros de la "
        "lista son los que más valor aportan si se retienen.",
        "<b>Consultar al cliente.</b> En el Buscador por Cliente, ingresar el ID o el apellido. Si es "
        "un cliente nuevo, completar el formulario con sus datos.",
        "<b>Leer la recomendación.</b> La aplicación muestra la probabilidad de fuga, el margen "
        "proyectado, el incentivo sugerido y la decisión comercial. La tabla de cálculo detalla la "
        "fórmula aplicada a cada incentivo, para que el gestor entienda por qué se eligió uno.",
        "<b>Preparar la conversación.</b> Abrir la sección “¿Por qué la red asignó este puntaje?” para "
        "ver qué variables elevan el riesgo (por ejemplo, inactividad o un solo producto) y orientar el "
        "diálogo hacia esa causa.",
        "<b>Ejecutar la acción.</b> En los segmentos A y B, ofrecer el incentivo indicado. En el "
        "segmento A, el caso se deriva a un gestor VIP en un plazo máximo de 48 horas. En los segmentos "
        "C y D, brindar la atención habitual sin conceder beneficios adicionales.",
        "<b>Registrar el resultado.</b> Anotar en el CRM si el cliente aceptó la oferta y si "
        "permanece a los 90 días. Este registro es indispensable para medir la efectividad real de "
        "cada incentivo y ajustar el simulador.",
    ]
    for i, t in enumerate(steps, 1):
        story.append(Paragraph(t, S["bullet"], bulletText=f"{i}."))
    story.append(P("Mantenimiento del modelo", "h2"))
    story.append(table([
        ["Actividad", "Frecuencia", "Detalle"],
        ["Actualización de datos y nueva puntuación de la cartera", "Mensual",
         "Cargar el corte de clientes del mes y regenerar la lista de trabajo."],
        ["Monitoreo de desempeño", "Mensual",
         "Comparar las probabilidades con las salidas reales. Revisar ROC-AUC y la deriva de las "
         "variables (PSI)."],
        ["Recalibración de probabilidades", "Trimestral",
         "Reajustar el escalado de Platt con los datos más recientes."],
        ["Reentrenamiento completo de la red", "Semestral o por alerta",
         "Ejecutar <i>python -m src.train</i> si el ROC-AUC cae por debajo de 0,80 o si el PSI de "
         "alguna variable clave supera 0,20."],
        ["Actualización de supuestos financieros", "Trimestral",
         "Reemplazar la efectividad supuesta de los incentivos por la medida en el piloto."],
    ], [5.2 * cm, 3 * cm, 8.8 * cm], bold_first=True))

    # ----------------------------------------- 5. CONCLUSIONES Y PRÓXIMOS PASOS
    story.append(PageBreak())
    story.append(P("5. Conclusiones y Próximos Pasos", "h1"))
    story.append(P(
        f"Deep Retention cumple la directriz de la Junta Directiva: cada peso invertido en "
        f"fidelización está respaldado por una ganancia neta esperada positiva. La red neuronal "
        f"ordena correctamente el riesgo (ROC-AUC de {num(mt['roc_auc'], 3)}), sus probabilidades "
        f"están calibradas y el filtro financiero concentra la inversión en "
        f"{num(k['clientes_intervenidos'])} clientes, con un beneficio neto esperado de "
        f"{money(k['ganancia_neta'], True)}. A la vez, el sistema evita gastar en "
        f"{num(k['salida_permitida'])} clientes que no son rentables."))
    story.append(P(
        "Los resultados financieros dependen de supuestos de margen y de efectividad que todavía "
        "no se han medido en campo. Por eso, antes del despliegue general se propone un piloto "
        "controlado:"))
    story.append(table([
        ["Fase", "Período", "Actividades"],
        ["1. Preparación", "Semanas 1 – 2",
         "Integración con el corte mensual de datos, validación de supuestos con Finanzas y "
         "capacitación de los gestores de las sucursales piloto."],
        ["2. Piloto con grupo de control", "Semanas 3 – 10",
         "En 3 a 5 sucursales, los clientes recomendados se dividen al azar: la mitad recibe el "
         "incentivo y la otra mitad no. Así se mide la efectividad real de cada incentivo."],
        ["3. Evaluación", "Semanas 11 – 12",
         "Comparar las tasas de fuga entre ambos grupos, recalcular el ROI con datos reales y "
         "actualizar el simulador."],
        ["4. Despliegue gradual", "Meses 4 – 5",
         "Extender la aplicación al 50 % de la red de sucursales y conectar el registro de "
         "resultados con el CRM."],
        ["5. Despliegue general", "Mes 6",
         "Cobertura total, con reentrenamiento y monitoreo según el plan de mantenimiento."],
    ], [4.2 * cm, 2.8 * cm, 10 * cm], bold_first=True))
    story.append(Spacer(1, 6))
    story.append(P(
        "Como evolución natural, se sugiere incorporar un <i>modelo de uplift</i> que estime "
        "directamente cuánto cambia la probabilidad de fuga con cada incentivo, a partir de los "
        "datos que genere el piloto. También conviene revisar con el área de cumplimiento el uso "
        "de variables sensibles como el género."))
    story.append(PageBreak())

    # ------------------------------------------------------- ANEXO A: SUPUESTOS
    story.append(P("Anexo A. Supuestos financieros", "h1"))
    story.append(P(
        "El conjunto de datos no incluye el margen por cliente ni el costo de los incentivos, por lo "
        "que se definieron los siguientes supuestos, coherentes con la operación de un banco minorista. "
        "Todos pueden modificarse desde el Simulador de Parámetros de la aplicación."))
    story.append(table([
        ["Concepto", "Valor", "Uso"],
        ["Margen neto de interés sobre el saldo", pct(params.net_interest_margin), "Ingreso anual"],
        ["Comisión anual por producto", money(params.fee_per_product), "Ingreso anual"],
        ["Ingreso anual por tarjeta (Silver / Gold / Platinum / Diamond)",
         " / ".join(money(v) for v in params.card_income.values()), "Ingreso anual, si tiene tarjeta"],
        ["Costo de redención por punto acumulado", "$" + num(params.point_cost, 2), "Costo anual"],
        ["Costo operativo por cliente", money(params.operating_cost), "Costo anual"],
        ["Horizonte de permanencia", f"{num(params.horizon_years)} años", "Margen del cliente"],
        ["Saldo mínimo de cuenta patrimonial", money(params.high_balance), "Elegibilidad Gestor VIP"],
        ["Riesgo mínimo para intervenir", num(params.min_risk, 2), "Segmento D"],
        ["Gestor VIP y Tarifa Plana", f"{money(params.vip_cost)} · efectividad {pct(params.vip_eff, 0)}",
         "Saldo ≥ " + money(params.high_balance)],
        ["Exoneración de Comisiones", f"{money(params.fee_waiver_cost)} · efectividad "
                                      f"{pct(params.fee_waiver_eff, 0)}", "Clientes con saldo"],
        ["Bonificación Transaccional", f"{money(params.bonus_cost)} · efectividad "
                                       f"{pct(params.bonus_eff, 0)}", "Clientes con tarjeta"],
    ], [7.4 * cm, 5 * cm, 4.6 * cm], bold_first=True))
    story.append(Spacer(1, 6))
    story.append(P(
        "<b>Margen anual</b> = saldo × margen de interés + productos × comisión + ingreso por tarjeta "
        "− puntos × costo por punto − costo operativo. <b>Margen del cliente</b> = margen anual × "
        "horizonte de permanencia."))
    story.append(P(
        "<b>Nota metodológica.</b> Las cifras de cartera se calcularon sobre los 10.000 clientes del "
        "archivo, como haría el banco en operación. Las métricas técnicas del modelo se reportan "
        "solo sobre el conjunto de prueba, que la red nunca vio durante el entrenamiento."))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Informe Ejecutivo · Deep Retention",
                            author=", ".join(AUTORES))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUTPUT


if __name__ == "__main__":
    print(f"Informe generado en: {build()}")
