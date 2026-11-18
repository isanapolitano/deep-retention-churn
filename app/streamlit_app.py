"""Deep Retention · Aplicación interactiva para ejecutivos de cuenta.

Ejecutar desde la raíz del repositorio:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.financial import (  # noqa: E402
    CARD_TYPES,
    FinancialParams,
    evaluate_portfolio,
    naive_strategy,
    portfolio_kpis,
    segment_matrix,
)
from src.inference import ChurnPredictor  # noqa: E402

st.set_page_config(page_title="Deep Retention", page_icon="🏦", layout="wide")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, WARNING, SERIOUS, CRITICAL, GRAY = "#0ca30c", "#fab219", "#ec835a", "#d03b3b", "#8a8984"
SEGMENT_COLORS = {
    "A: Alto Valor / Alto Riesgo": BLUE,
    "B: Medio / Riesgo Moderado": AQUA,
    "C: No Rentable": ORANGE,
    "D: Estable": GRAY,
}
DECISION_STYLE = {
    "Retención Prioritaria": ("✅", GOOD),
    "Retención Automatizada": ("✅", GOOD),
    "Salida Permitida": ("⛔", CRITICAL),
    "Sin Acción Requerida": ("ℹ️", GRAY),
}


# --------------------------------------------------------------------- utils
def money(x: float, short: bool = False) -> str:
    """Formato monetario en español: $ 1.234.567 o $ 1,2 M."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if short and x >= 1e6:
        return f"{sign}$ {x / 1e6:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    if short and x >= 1e3:
        return f"{sign}$ {x / 1e3:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}$ {x:,.0f}".replace(",", ".")


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %".replace(".", ",")


@st.cache_resource(show_spinner="Cargando la red neuronal...")
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor.load()


@st.cache_data(show_spinner="Puntuando la cartera...")
def get_portfolio() -> pd.DataFrame:
    if config.SCORED_PORTFOLIO_PATH.exists():
        return pd.read_csv(config.SCORED_PORTFOLIO_PATH)
    from src.data_prep import load_raw

    df = load_raw()
    return df.assign(prob_fuga=get_predictor().predict_proba(df))


@st.cache_data
def get_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text()) if config.METADATA_PATH.exists() else {}


# ---------------------------------------------------- parámetros del simulador
DEFAULTS = FinancialParams()
PARAM_KEYS = {
    "net_interest_margin": DEFAULTS.net_interest_margin * 100,
    "fee_per_product": DEFAULTS.fee_per_product,
    "point_cost": DEFAULTS.point_cost,
    "operating_cost": DEFAULTS.operating_cost,
    "horizon_years": DEFAULTS.horizon_years,
    "high_balance": DEFAULTS.high_balance,
    "min_risk": DEFAULTS.min_risk,
    "vip_cost": DEFAULTS.vip_cost,
    "vip_eff": DEFAULTS.vip_eff,
    "fee_waiver_cost": DEFAULTS.fee_waiver_cost,
    "fee_waiver_eff": DEFAULTS.fee_waiver_eff,
    "bonus_cost": DEFAULTS.bonus_cost,
    "bonus_eff": DEFAULTS.bonus_eff,
    **{f"card_{c}": DEFAULTS.card_income[c] for c in CARD_TYPES},
}
for k, v in PARAM_KEYS.items():
    st.session_state.setdefault(f"p_{k}", float(v))


def reset_params() -> None:
    for k, v in PARAM_KEYS.items():
        st.session_state[f"p_{k}"] = float(v)


def current_params() -> FinancialParams:
    s = st.session_state
    return FinancialParams(
        net_interest_margin=s.p_net_interest_margin / 100,
        fee_per_product=s.p_fee_per_product,
        card_income={c: s[f"p_card_{c}"] for c in CARD_TYPES},
        point_cost=s.p_point_cost,
        operating_cost=s.p_operating_cost,
        horizon_years=s.p_horizon_years,
        high_balance=s.p_high_balance,
        min_risk=s.p_min_risk,
        vip_cost=s.p_vip_cost, vip_eff=s.p_vip_eff,
        fee_waiver_cost=s.p_fee_waiver_cost, fee_waiver_eff=s.p_fee_waiver_eff,
        bonus_cost=s.p_bonus_cost, bonus_eff=s.p_bonus_eff,
    )


# ------------------------------------------------------------------- carga
try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

portfolio_raw = get_portfolio()
params = current_params()
scored = evaluate_portfolio(portfolio_raw, portfolio_raw["prob_fuga"].to_numpy(), params)
kpis = portfolio_kpis(scored)
meta = get_metadata()

# ------------------------------------------------------------------ encabezado
st.title("🏦 Deep Retention")
st.caption(
    "Sistema de retención selectiva de clientes. La red neuronal estima la probabilidad de fuga "
    "y el motor financiero solo autoriza un incentivo cuando la **ganancia neta esperada es "
    "mayor a cero**."
)
if params.to_dict() != DEFAULTS.to_dict():
    st.info("Se están usando supuestos modificados en el **Simulador de Parámetros**.", icon="🧮")

tab_client, tab_portfolio, tab_sim, tab_batch, tab_model = st.tabs([
    "🔎 Buscador por Cliente",
    "📊 Panel de Cartera",
    "🧮 Simulador de Parámetros",
    "📁 Evaluación por Lote",
    "🧠 Modelo y Explicabilidad",
])


# ============================================================ 1. BUSCADOR
def client_form() -> pd.DataFrame | None:
    with st.form("nuevo_cliente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            geo = st.selectbox("País", ["France", "Germany", "Spain"])
            gender = st.selectbox("Género", ["Female", "Male"])
            age = st.number_input("Edad", 18, 100, 45)
            tenure = st.number_input("Antigüedad (años)", 0, 10, 3)
            credit = st.number_input("Puntaje crediticio", 300, 850, 650)
        with c2:
            balance = st.number_input("Saldo ($)", 0.0, 1_000_000.0, 125_000.0, step=1_000.0)
            salary = st.number_input("Salario estimado ($)", 0.0, 500_000.0, 90_000.0, step=1_000.0)
            products = st.number_input("Número de productos", 1, 4, 1)
            points = st.number_input("Puntos acumulados", 0, 1_000, 500)
        with c3:
            card = st.selectbox("Tipo de tarjeta", CARD_TYPES, index=1)
            has_card = st.checkbox("Tiene tarjeta de crédito", True)
            active = st.checkbox("Cliente activo", False)
            sat = st.slider("Satisfacción (1-5)", 1, 5, 3)
        ok = st.form_submit_button("Evaluar cliente", type="primary")
    if ok:
        st.session_state["new_client"] = pd.DataFrame([{
            "CustomerId": 0, "Surname": "Cliente nuevo", "CreditScore": credit, "Geography": geo,
            "Gender": gender, "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": products, "HasCrCard": int(has_card), "IsActiveMember": int(active),
            "EstimatedSalary": salary, "Satisfaction Score": sat, "Card Type": card,
            "Point Earned": points,
            }])
    # Se guarda en sesión para que el resultado no desaparezca al pulsar otros botones.
    return st.session_state.get("new_client")


def show_client(row_df: pd.DataFrame, proba: float) -> None:
    res = evaluate_portfolio(row_df, np.array([proba]), params).iloc[0]
    icon, color = DECISION_STYLE[res["decision"]]

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;border-left:6px solid {color};"
        f"background:rgba(127,127,127,0.08);margin-bottom:12px'>"
        f"<div style='font-size:0.85rem;opacity:0.75'>Decisión comercial sugerida</div>"
        f"<div style='font-size:1.5rem;font-weight:700'>{icon} {res['decision']}</div>"
        f"<div style='opacity:0.85'>{res['segmento']} · Incentivo: <b>{res['incentivo']}</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilidad de fuga (red neuronal)", f"{proba:.2f}", help=res["banda_riesgo"])
    c2.metric(f"Margen del cliente ({params.horizon_years:g} años)", money(res["margen_cliente"]),
              help=f"Margen neto anual: {money(res['margen_anual'])}")
    c3.metric("Costo del incentivo", money(res["costo_incentivo"]))
    c4.metric("Ganancia neta esperada", money(res["ganancia_neta_esperada"]))
    st.progress(float(proba), text=f"Riesgo de abandono: {res['banda_riesgo']}")

    # Detalle de la regla de decisión para cada incentivo del catálogo.
    st.markdown("##### Cálculo de la regla financiera")
    elig = {
        "vip": row_df["Balance"].iloc[0] >= params.high_balance,
        "fee_waiver": row_df["Balance"].iloc[0] > 0,
        "bonus": row_df["HasCrCard"].iloc[0] == 1,
    }
    rows = []
    for inc in params.incentives():
        eng = proba * inc.effectiveness * res["margen_cliente"] - inc.cost
        rows.append({
            "Incentivo": inc.name,
            "Elegible": "Sí" if elig[inc.key] else "No",
            "Fórmula": f"({proba:.2f} × {inc.effectiveness:.2f} × {money(res['margen_cliente'])})"
                       f" − {money(inc.cost)}",
            "Ganancia neta esperada": money(eng) if elig[inc.key] else "—",
            "Resultado": ("✔ Elegido" if res["incentivo"] == inc.name else
                          ("Rentable" if elig[inc.key] and eng > 0 else "No rentable")
                          if elig[inc.key] else "No aplica al perfil"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if proba < params.min_risk:
        st.caption(f"El riesgo es inferior a {params.min_risk:.2f}: el cliente sigue en mantenimiento "
                   "ordinario aunque alguna opción resulte rentable.")
    elif res["margen_cliente"] <= 0:
        st.caption("El cliente es deficitario (margen ≤ 0): ningún incentivo puede generar ganancia, "
                   "por lo que se cataloga como **Salida Permitida** y se bloquea el gasto.")

    with st.expander("🧠 ¿Por qué la red asignó este puntaje? (SHAP)"):
        if st.button("Calcular explicación", key=f"shap_{row_df['CustomerId'].iloc[0]}"):
            with st.spinner("Calculando contribuciones..."):
                sv = predictor.explain(row_df, n_permutations=8)
            contrib = pd.DataFrame({
                "Variable": [config.FEATURE_LABELS[f] for f in config.FEATURES],
                "Valor": [str(row_df[f].iloc[0]) for f in config.FEATURES],
                "Contribución": sv.values[0],
            }).sort_values("Contribución", key=np.abs, ascending=True)
            fig = go.Figure(go.Bar(
                x=contrib["Contribución"], y=contrib["Variable"] + " = " + contrib["Valor"],
                orientation="h",
                marker_color=[CRITICAL if v > 0 else BLUE for v in contrib["Contribución"]],
                hovertemplate="%{y}<br>Contribución: %{x:+.3f}<extra></extra>",
            ))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                              title=f"Base de la cartera: {float(sv.base_values[0]):.2f} → "
                                    f"cliente: {proba:.2f}",
                              xaxis_title="Contribución a la probabilidad de fuga")
            st.plotly_chart(fig, width="stretch")
            st.caption("Rojo: la variable aumenta el riesgo. Azul: lo reduce.")


with tab_client:
    mode = st.radio("Tipo de consulta", ["Cliente de la cartera", "Cliente nuevo"], horizontal=True)
    if mode == "Cliente de la cartera":
        c1, c2 = st.columns([1, 2])
        with c1:
            query = st.text_input("Buscar por ID de cliente o apellido", placeholder="Ej.: 15634602")
        matches = portfolio_raw
        if query.strip():
            q = query.strip()
            matches = portfolio_raw[
                portfolio_raw["CustomerId"].astype(str).str.contains(q)
                | portfolio_raw["Surname"].str.contains(q, case=False, na=False)
            ]
        if matches.empty:
            st.warning("No se encontró ningún cliente con ese criterio.")
        else:
            with c2:
                options = matches["CustomerId"].tolist()[:500]
                cid = st.selectbox(
                    f"Resultados ({len(matches)})", options,
                    format_func=lambda i: f"{i} · {portfolio_raw.loc[portfolio_raw.CustomerId == i, 'Surname'].iloc[0]}",
                )
            row = portfolio_raw[portfolio_raw["CustomerId"] == cid]
            with st.expander("Ficha del cliente", expanded=False):
                st.dataframe(row[["CustomerId", "Surname"] + config.FEATURES]
                             .rename(columns=config.FEATURE_LABELS), hide_index=True)
            show_client(row, float(row["prob_fuga"].iloc[0]))
    else:
        new = client_form()
        if new is not None:
            try:
                show_client(new, float(predictor.predict_proba(new)[0]))
            except ValueError as e:
                st.error(f"Datos inválidos: {e}")


# ======================================================== 2. PANEL DE CARTERA
with tab_portfolio:
    f1, f2, f3 = st.columns(3)
    geo_f = f1.multiselect("País", sorted(scored["Geography"].unique()))
    seg_f = f2.multiselect("Segmento", list(SEGMENT_COLORS))
    tier_f = f3.multiselect("Nivel de saldo", sorted(scored["nivel_saldo"].unique()))
    view = scored
    if geo_f:
        view = view[view["Geography"].isin(geo_f)]
    if seg_f:
        view = view[view["segmento"].isin(seg_f)]
    if tier_f:
        view = view[view["nivel_saldo"].isin(tier_f)]
    k = portfolio_kpis(view)

    r1 = st.columns(4)
    r1[0].metric("Clientes analizados", f"{k['clientes']:,}".replace(",", "."))
    r1[1].metric("Capital en riesgo crítico", money(k["capital_riesgo_critico"], True),
                 help=f"Saldo de {k['clientes_riesgo_critico']} clientes con P(fuga) ≥ {config.CRITICAL_RISK}")
    r1[2].metric("Capital rescatable estimado", money(k["capital_intervenido"], True),
                 help="Saldo de los clientes rentables a intervenir (ganancia neta esperada > 0)")
    r1[3].metric("Saldo que se espera conservar", money(k["saldo_rescatado_esperado"], True),
                 help="Σ saldo × P(fuga) × efectividad del incentivo, solo clientes intervenidos")
    r2 = st.columns(4)
    r2[0].metric("Clientes a intervenir", f"{k['clientes_intervenidos']:,}".replace(",", "."))
    r2[1].metric("Inversión en fidelización", money(k["inversion"], True))
    r2[2].metric("Ganancia neta esperada", money(k["ganancia_neta"], True),
                 help=f"Margen protegido {money(k['margen_protegido'])} − inversión")
    r2[3].metric("ROI", pct(k["roi"], 0), help="Ganancia neta / inversión")
    st.caption(
        f"Efectividad proyectada: se espera retener ≈ {k['clientes_retenidos_esperados']:.0f} de las "
        f"{k['fugas_esperadas_intervenidos']:.0f} fugas esperadas entre los clientes intervenidos "
        f"({pct(k['clientes_retenidos_esperados'] / max(k['fugas_esperadas_intervenidos'], 1))}). "
        f"{k['salida_permitida']} clientes en riesgo quedan como Salida Permitida."
    )

    c1, c2 = st.columns(2)
    seg_counts = (view.groupby("segmento").agg(clientes=("CustomerId", "size"),
                                               saldo=("Balance", "sum"))
                  .reindex(list(SEGMENT_COLORS)).dropna().reset_index())
    fig = px.bar(seg_counts, x="clientes", y="segmento", orientation="h", color="segmento",
                 color_discrete_map=SEGMENT_COLORS, text="clientes",
                 labels={"clientes": "Clientes", "segmento": ""}, title="Clientes por segmento")
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    c1.plotly_chart(fig, width="stretch")

    band = (view.groupby("banda_riesgo")["Balance"].sum()
            .reindex(["Bajo (< 0.30)", "Moderado (0.30 – 0.65)", "Crítico (≥ 0.65)"]).fillna(0)
            .reset_index())
    fig = px.bar(band, x="banda_riesgo", y="Balance", color="banda_riesgo",
                 color_discrete_map={"Bajo (< 0.30)": GRAY, "Moderado (0.30 – 0.65)": WARNING,
                                     "Crítico (≥ 0.65)": CRITICAL},
                 labels={"Balance": "Saldo total ($)", "banda_riesgo": ""},
                 title="Saldo total por banda de riesgo")
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=50, b=10))
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    fig = px.histogram(view, x="prob_fuga", nbins=40, color="segmento",
                       color_discrete_map=SEGMENT_COLORS,
                       labels={"prob_fuga": "Probabilidad de fuga", "segmento": "Segmento"},
                       title="Distribución de la probabilidad de fuga")
    fig.update_layout(height=340, bargap=0.05, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25), yaxis_title="Clientes")
    c3.plotly_chart(fig, width="stretch")

    inc = (view[view["incentivo_key"] != "none"].groupby("incentivo")
           .agg(clientes=("CustomerId", "size"), costo=("costo_incentivo", "sum"),
                ganancia=("ganancia_neta_esperada", "sum")).reset_index())
    if not inc.empty:
        fig = go.Figure()
        fig.add_bar(name="Inversión", x=inc["incentivo"], y=inc["costo"], marker_color=ORANGE)
        fig.add_bar(name="Ganancia neta esperada", x=inc["incentivo"], y=inc["ganancia"],
                    marker_color=BLUE)
        fig.update_layout(barmode="group", title="Inversión vs. ganancia por incentivo", height=340,
                          margin=dict(l=10, r=10, t=50, b=10), yaxis_title="$",
                          legend=dict(orientation="h", y=-0.2))
        c4.plotly_chart(fig, width="stretch")

    st.markdown("##### Matriz de decisión comercial")
    mat = segment_matrix(view)
    show = mat.copy()
    for c in ["Saldo medio", "Margen cliente medio", "Costo total", "Ganancia neta esperada"]:
        show[c] = show[c].map(money)
    for c in ["Prob. mínima", "Prob. media"]:
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    show["Clientes"] = show["Clientes"].astype(int)
    st.dataframe(show, width="stretch")

    st.markdown("##### Lista de trabajo: clientes a contactar (ordenados por ganancia neta esperada)")
    work = (view[view["incentivo_key"] != "none"]
            .sort_values("ganancia_neta_esperada", ascending=False)
            [["CustomerId", "Surname", "Geography", "Balance", "prob_fuga", "margen_cliente",
              "incentivo", "costo_incentivo", "ganancia_neta_esperada", "segmento"]])
    st.dataframe(
        work.head(200), hide_index=True, width="stretch",
        column_config={
            "CustomerId": "ID", "Surname": "Apellido", "Geography": "País",
            "Balance": st.column_config.NumberColumn("Saldo", format="$ %.0f"),
            "prob_fuga": st.column_config.ProgressColumn("P(fuga)", min_value=0, max_value=1,
                                                         format="%.2f"),
            "margen_cliente": st.column_config.NumberColumn("Margen cliente", format="$ %.0f"),
            "incentivo": "Incentivo",
            "costo_incentivo": st.column_config.NumberColumn("Costo", format="$ %.0f"),
            "ganancia_neta_esperada": st.column_config.NumberColumn("Ganancia neta", format="$ %.0f"),
            "segmento": "Segmento",
        },
    )
    st.download_button("⬇️ Descargar lista completa (CSV)", work.to_csv(index=False).encode("utf-8"),
                       "clientes_a_intervenir.csv", "text/csv")


# ========================================================= 3. SIMULADOR
with tab_sim:
    st.markdown("Ajuste los supuestos y observe el impacto en la rentabilidad neta de la campaña. "
                "Los cambios se aplican a toda la aplicación.")
    st.button("↺ Restablecer valores por defecto", on_click=reset_params)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Catálogo de incentivos**")
        st.number_input("Gestor VIP: costo ($)", 0.0, 5000.0, step=10.0, key="p_vip_cost")
        st.slider("Gestor VIP: efectividad", 0.0, 1.0, step=0.01, key="p_vip_eff")
        st.number_input("Exoneración de comisiones: costo ($)", 0.0, 2000.0, step=10.0,
                        key="p_fee_waiver_cost")
        st.slider("Exoneración: efectividad", 0.0, 1.0, step=0.01, key="p_fee_waiver_eff")
        st.number_input("Bonificación transaccional: costo ($)", 0.0, 2000.0, step=5.0,
                        key="p_bonus_cost")
        st.slider("Bonificación: efectividad", 0.0, 1.0, step=0.01, key="p_bonus_eff")
    with s2:
        st.markdown("**Margen del cliente**")
        st.slider("Margen neto de interés sobre el saldo (%)", 0.0, 5.0, step=0.1,
                  key="p_net_interest_margin")
        st.number_input("Comisión anual por producto ($)", 0.0, 500.0, step=5.0,
                        key="p_fee_per_product")
        st.number_input("Costo operativo anual por cliente ($)", 0.0, 1000.0, step=10.0,
                        key="p_operating_cost")
        st.number_input("Costo por punto acumulado ($)", 0.0, 1.0, step=0.01, key="p_point_cost")
        st.slider("Horizonte de permanencia (años)", 1.0, 10.0, step=0.5, key="p_horizon_years")
    with s3:
        st.markdown("**Ingreso anual por tarjeta y políticas**")
        for c in CARD_TYPES:
            st.number_input(f"Tarjeta {c.title()} ($)", 0.0, 1000.0, step=5.0, key=f"p_card_{c}")
        st.number_input("Saldo mínimo para cuenta patrimonial ($)", 0.0, 300_000.0, step=5_000.0,
                        key="p_high_balance")
        st.slider("Riesgo mínimo para intervenir", 0.0, 0.9, step=0.05, key="p_min_risk")

    base = portfolio_kpis(evaluate_portfolio(portfolio_raw, portfolio_raw["prob_fuga"].to_numpy(),
                                             DEFAULTS))
    st.divider()
    st.markdown("##### Escenario actual vs. escenario base")
    m = st.columns(5)
    m[0].metric("Clientes intervenidos", kpis["clientes_intervenidos"],
                kpis["clientes_intervenidos"] - base["clientes_intervenidos"])
    m[1].metric("Inversión", money(kpis["inversion"], True),
                money(kpis["inversion"] - base["inversion"], True), delta_color="inverse")
    m[2].metric("Ganancia neta", money(kpis["ganancia_neta"], True),
                money(kpis["ganancia_neta"] - base["ganancia_neta"], True))
    m[3].metric("ROI", pct(kpis["roi"], 0), pct(kpis["roi"] - base["roi"], 0))
    m[4].metric("Capital rescatable", money(kpis["saldo_rescatado_esperado"], True),
                money(kpis["saldo_rescatado_esperado"] - base["saldo_rescatado_esperado"], True))

    naive = naive_strategy(scored, params)
    st.markdown("##### Retención selectiva vs. retención indiscriminada (umbral fijo 0.5)")
    comp = pd.DataFrame({
        "Estrategia": ["Deep Retention (ganancia neta > 0)", "Indiscriminada (P ≥ 0.5)"],
        "Clientes intervenidos": [kpis["clientes_intervenidos"], naive["clientes_intervenidos"]],
        "Inversión": [money(kpis["inversion"]), money(naive["inversion"])],
        "Ganancia neta esperada": [money(kpis["ganancia_neta"]), money(naive["ganancia_neta"])],
        "ROI": [pct(kpis["roi"], 0), pct(naive["roi"], 0)],
        "Intervenciones con pérdida": [0, naive["clientes_con_perdida"]],
    })
    st.dataframe(comp, hide_index=True, width="stretch")

    st.markdown("##### Sensibilidad al costo de los incentivos")
    factors = np.round(np.arange(0.5, 3.01, 0.25), 2)
    rows = []
    for f in factors:
        pp = current_params()
        pp.vip_cost *= f
        pp.fee_waiver_cost *= f
        pp.bonus_cost *= f
        kk = portfolio_kpis(evaluate_portfolio(portfolio_raw, portfolio_raw["prob_fuga"].to_numpy(), pp))
        rows.append({"Factor de costo": f, "Ganancia neta": kk["ganancia_neta"],
                     "Clientes intervenidos": kk["clientes_intervenidos"]})
    sens = pd.DataFrame(rows)
    cA, cB = st.columns(2)
    fig = px.line(sens, x="Factor de costo", y="Ganancia neta", markers=True,
                  title="Ganancia neta esperada según el costo de los incentivos")
    fig.update_traces(line_color=BLUE)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="$",
                      xaxis_title="Multiplicador sobre el costo actual")
    cA.plotly_chart(fig, width="stretch")
    fig = px.line(sens, x="Factor de costo", y="Clientes intervenidos", markers=True,
                  title="Clientes intervenidos según el costo de los incentivos")
    fig.update_traces(line_color=ORANGE)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Multiplicador sobre el costo actual")
    cB.plotly_chart(fig, width="stretch")
    st.caption("Al encarecer los incentivos, el motor deja de intervenir a los clientes menos "
               "rentables en lugar de incurrir en pérdidas: la ganancia baja, pero nunca se "
               "autoriza un gasto con retorno negativo.")


# ====================================================== 4. EVALUACIÓN POR LOTE
with tab_batch:
    st.markdown(
        "Cargue un archivo CSV con clientes nuevos. Debe incluir las columnas: "
        + ", ".join(f"`{c}`" for c in config.FEATURES)
        + ". Las columnas `CustomerId` y `Surname` son opcionales."
    )
    template = portfolio_raw[["CustomerId", "Surname"] + config.FEATURES].head(5)
    st.download_button("⬇️ Descargar plantilla de ejemplo", template.to_csv(index=False).encode("utf-8"),
                       "plantilla_clientes.csv", "text/csv")
    up = st.file_uploader("Archivo CSV", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            if "CustomerId" not in new.columns:
                new["CustomerId"] = np.arange(1, len(new) + 1)
            if "Surname" not in new.columns:
                new["Surname"] = ""
            unknown = {c: sorted(set(new[c].astype(str)) - set(portfolio_raw[c].astype(str)))
                       for c in config.CATEGORICAL_FEATURES if c in new.columns}
            unknown = {c: v for c, v in unknown.items() if v}
            if unknown:
                st.warning(f"Categorías no vistas en el entrenamiento (se tratarán como neutras): {unknown}")
            proba = predictor.predict_proba(new)
            res = evaluate_portfolio(new, proba, params)
            st.success(f"{len(res)} clientes evaluados.")
            st.dataframe(res[["CustomerId", "Surname", "prob_fuga", "margen_cliente", "decision",
                              "incentivo", "costo_incentivo", "ganancia_neta_esperada"]],
                         hide_index=True, width="stretch")
            st.download_button("⬇️ Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                               "resultados_deep_retention.csv", "text/csv")
        except ValueError as e:
            st.error(f"No se pudo procesar el archivo: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Error inesperado al leer el archivo: {e}")


# ================================================== 5. MODELO Y EXPLICABILIDAD
with tab_model:
    if meta:
        mt = meta["metricas_prueba"]
        c = st.columns(5)
        c[0].metric("ROC-AUC (prueba)", f"{mt['roc_auc']:.3f}")
        c[1].metric("PR-AUC (prueba)", f"{mt['pr_auc']:.3f}")
        c[2].metric("Brier (calibrado)", f"{mt['brier_calibrado']:.3f}")
        c[3].metric("Error de calibración (ECE)", f"{mt['ece_calibrado']:.3f}",
                    f"{mt['ece_calibrado'] - mt['ece_sin_calibrar']:+.3f} vs. sin calibrar",
                    delta_color="inverse")
        c[4].metric("Regresión logística (ROC-AUC)", f"{mt['baseline_logistica_roc_auc']:.3f}",
                    help="Modelo de referencia para comparar")
        st.markdown(
            f"**Arquitectura:** MLP {meta['hiperparametros']['units_1']} → "
            f"{meta['hiperparametros']['units_2']} → 1 · ReLU, Batch Normalization, Dropout "
            f"{meta['hiperparametros']['dropout_1']}/{meta['hiperparametros']['dropout_2']}, "
            f"L2 {meta['hiperparametros']['l2']} · salida Sigmoide · optimizador Adam · "
            f"{meta['parametros_capas']:,} parámetros · Early Stopping en la época "
            f"{meta['mejor_epoca']} de {meta['epocas_entrenadas']}."
        )
        st.warning(
            f"**Fuga de datos controlada:** la variable `Complain` coincide con `Exited` en el "
            f"{pct(meta['fuga_de_datos']['coincidencia_complain_exited'], 2)} de los casos. "
            f"Con ella, incluso una regresión logística alcanza un ROC-AUC de "
            f"{meta['fuga_de_datos']['roc_auc_logistica_con_complain']:.3f}, un resultado irreal. "
            "Por eso se excluyó del modelo.", icon="⚠️")
    figs = config.FIGURES_DIR
    c1, c2 = st.columns(2)
    for col, name, cap in [
        (c1, "shap_importancia.png", "Importancia global de variables"),
        (c2, "shap_beeswarm.png", "Dirección del efecto por cliente"),
        (c1, "roc_pr.png", "Curvas ROC y Precision-Recall"),
        (c2, "calibracion.png", "Calibración de probabilidades"),
        (c1, "curvas_entrenamiento.png", "Curvas de entrenamiento"),
    ]:
        if (figs / name).exists():
            col.image(str(figs / name), caption=cap, width="stretch")
