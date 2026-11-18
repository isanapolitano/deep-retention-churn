"""Motor financiero: margen del cliente, catálogo de incentivos y regla de decisión.

Regla de negocio (directriz de la Junta Directiva):

    Ganancia Neta Esperada = (P(fuga) x Efectividad x Margen del Cliente) - Costo del Incentivo

La acción comercial solo se ejecuta si la Ganancia Neta Esperada es mayor a cero.
No se usa un umbral fijo de 0.5: la decisión depende del valor de cada cliente.

Todos los supuestos monetarios son anuales, en dólares, y se pueden modificar
desde el simulador de la aplicación.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import config

CARD_TYPES = ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]


@dataclass
class Incentive:
    key: str
    name: str
    cost: float          # costo unitario para el banco ($ por cliente)
    effectiveness: float  # probabilidad de que el cliente se quede gracias al incentivo
    description: str


@dataclass
class FinancialParams:
    # --- Supuestos de margen (anual) ---
    net_interest_margin: float = 0.015   # margen neto de interés sobre el saldo (1.5 %)
    fee_per_product: float = 45.0        # comisiones anuales por producto contratado
    card_income: dict = field(default_factory=lambda: {
        "SILVER": 25.0, "GOLD": 50.0, "PLATINUM": 80.0, "DIAMOND": 120.0,
    })                                   # ingreso anual por tarjeta (si tiene tarjeta)
    point_cost: float = 0.10             # costo de redención por punto acumulado
    operating_cost: float = 180.0        # costo operativo anual de servir al cliente
    horizon_years: float = 3.0           # horizonte de permanencia esperado
    # --- Segmentación por saldo ---
    high_balance: float = 120_000.0      # desde aquí: "Saldo Superior" (cuenta patrimonial)
    # --- Política ---
    min_risk: float = config.LOW_RISK    # por debajo: mantenimiento ordinario
    # --- Catálogo de incentivos ---
    vip_cost: float = 450.0
    vip_eff: float = 0.40
    fee_waiver_cost: float = 150.0
    fee_waiver_eff: float = 0.25
    bonus_cost: float = 80.0
    bonus_eff: float = 0.15

    def incentives(self) -> list[Incentive]:
        return [
            Incentive("vip", "Gestor VIP y Tarifa Plana", self.vip_cost, self.vip_eff,
                      "Ejecutivo asignado y tarifa plana. Solo cuentas patrimoniales."),
            Incentive("fee_waiver", "Exoneración de Comisiones", self.fee_waiver_cost,
                      self.fee_waiver_eff, "Exoneración de comisiones por 12 meses. "
                      "Clientes con saldo."),
            Incentive("bonus", "Bonificación Transaccional", self.bonus_cost, self.bonus_eff,
                      "Bonificación por consumo con tarjeta. Clientes con tarjeta."),
        ]

    def to_dict(self) -> dict:
        return asdict(self)


def balance_tier(balance: pd.Series, params: FinancialParams) -> pd.Series:
    return pd.Series(
        np.select(
            [balance >= params.high_balance, balance > 0],
            ["Saldo Superior", "Saldo Medio"],
            default="Saldo Bajo / Cero",
        ),
        index=balance.index,
    )


def annual_margin(df: pd.DataFrame, params: FinancialParams) -> pd.Series:
    """Margen neto anual que el cliente aporta al banco (puede ser negativo)."""
    card = df["Card Type"].map(params.card_income).fillna(0.0) * df["HasCrCard"]
    return (
        df["Balance"] * params.net_interest_margin
        + df["NumOfProducts"] * params.fee_per_product
        + card
        - df["Point Earned"] * params.point_cost
        - params.operating_cost
    )


def _eligibility(df: pd.DataFrame, params: FinancialParams) -> dict[str, np.ndarray]:
    return {
        "vip": (df["Balance"] >= params.high_balance).to_numpy(),
        "fee_waiver": (df["Balance"] > 0).to_numpy(),
        "bonus": (df["HasCrCard"] == 1).to_numpy(),
    }


def evaluate_portfolio(df: pd.DataFrame, proba: np.ndarray, params: FinancialParams) -> pd.DataFrame:
    """Aplica la regla de ganancia neta esperada a cada cliente.

    Devuelve el DataFrame original con columnas nuevas: probabilidad, margen,
    incentivo elegido, costo, ganancia neta esperada, decisión y segmento.
    """
    out = df.copy()
    p = np.clip(np.asarray(proba, dtype=float), 0, 1)
    out["prob_fuga"] = p
    out["margen_anual"] = annual_margin(out, params)
    out["margen_cliente"] = out["margen_anual"] * params.horizon_years
    out["nivel_saldo"] = balance_tier(out["Balance"], params)

    m = out["margen_cliente"].to_numpy()
    elig = _eligibility(out, params)
    incentives = params.incentives()

    eng = np.full((len(out), len(incentives)), -np.inf)
    for j, inc in enumerate(incentives):
        val = p * inc.effectiveness * m - inc.cost
        eng[:, j] = np.where(elig[inc.key], val, -np.inf)

    best = eng.argmax(axis=1)
    best_eng = eng[np.arange(len(out)), best]
    has_option = np.isfinite(best_eng)

    act = has_option & (best_eng > 0) & (p >= params.min_risk)

    names = np.array([i.name for i in incentives])
    costs = np.array([i.cost for i in incentives])
    effs = np.array([i.effectiveness for i in incentives])
    keys = np.array([i.key for i in incentives])

    out["incentivo"] = np.where(act, names[best], "Sin intervención")
    out["incentivo_key"] = np.where(act, keys[best], "none")
    out["costo_incentivo"] = np.where(act, costs[best], 0.0)
    out["efectividad"] = np.where(act, effs[best], 0.0)
    # Mejor ganancia neta posible (aunque sea negativa) para explicar la decisión.
    out["ganancia_neta_mejor_opcion"] = np.where(has_option, best_eng, np.nan)
    out["ganancia_neta_esperada"] = np.where(act, best_eng, 0.0)
    # Margen esperado que se protege (antes de restar el costo).
    out["margen_rescatado"] = np.where(act, p * effs[best] * m, 0.0)
    # Saldo que se espera conservar gracias a la intervención.
    out["saldo_rescatado"] = np.where(act, p * effs[best] * out["Balance"], 0.0)

    low = p < params.min_risk
    segment = np.select(
        [low, act & (keys[best] == "vip"), act],
        ["D: Estable", "A: Alto Valor / Alto Riesgo", "B: Medio / Riesgo Moderado"],
        default="C: No Rentable",
    )
    out["segmento"] = segment
    out["decision"] = np.select(
        [low, act & (keys[best] == "vip"), act],
        ["Sin Acción Requerida", "Retención Prioritaria", "Retención Automatizada"],
        default="Salida Permitida",
    )
    out["banda_riesgo"] = np.select(
        [p >= config.CRITICAL_RISK, p >= config.LOW_RISK],
        ["Crítico (≥ 0.65)", "Moderado (0.30 – 0.65)"],
        default="Bajo (< 0.30)",
    )
    return out


def portfolio_kpis(scored: pd.DataFrame) -> dict:
    """Indicadores consolidados para el panel y el informe ejecutivo."""
    act = scored["incentivo_key"] != "none"
    critical = scored["prob_fuga"] >= config.CRITICAL_RISK
    investment = float(scored["costo_incentivo"].sum())
    protected = float(scored["margen_rescatado"].sum())
    net = float(scored["ganancia_neta_esperada"].sum())
    return {
        "clientes": int(len(scored)),
        "clientes_riesgo_critico": int(critical.sum()),
        "capital_riesgo_critico": float(scored.loc[critical, "Balance"].sum()),
        "clientes_intervenidos": int(act.sum()),
        "capital_intervenido": float(scored.loc[act, "Balance"].sum()),
        "saldo_rescatado_esperado": float(scored["saldo_rescatado"].sum()),
        "clientes_retenidos_esperados": float((scored["prob_fuga"] * scored["efectividad"]).sum()),
        "fugas_esperadas_intervenidos": float(scored.loc[act, "prob_fuga"].sum()),
        "inversion": investment,
        "margen_protegido": protected,
        "ganancia_neta": net,
        "roi": net / investment if investment > 0 else 0.0,
        "salida_permitida": int((scored["segmento"] == "C: No Rentable").sum()),
    }


def naive_strategy(scored: pd.DataFrame, params: FinancialParams, threshold: float = 0.5) -> dict:
    """Escenario de comparación: retención indiscriminada con umbral fijo.

    A todo cliente con P(fuga) >= umbral se le da el incentivo más barato al que
    es elegible (o la bonificación), sin mirar su rentabilidad.
    """
    p = scored["prob_fuga"].to_numpy()
    m = scored["margen_cliente"].to_numpy()
    target = p >= threshold
    inc = {i.key: i for i in params.incentives()}
    use_vip = scored["Balance"].to_numpy() >= params.high_balance
    use_fee = scored["Balance"].to_numpy() > 0
    cost = np.where(use_vip, inc["vip"].cost, np.where(use_fee, inc["fee_waiver"].cost, inc["bonus"].cost))
    eff = np.where(use_vip, inc["vip"].effectiveness,
                   np.where(use_fee, inc["fee_waiver"].effectiveness, inc["bonus"].effectiveness))
    gain = p * eff * m - cost
    investment = float(cost[target].sum())
    net = float(gain[target].sum())
    return {
        "clientes_intervenidos": int(target.sum()),
        "inversion": investment,
        "ganancia_neta": net,
        "roi": net / investment if investment > 0 else 0.0,
        "clientes_con_perdida": int((gain[target] <= 0).sum()),
    }


def segment_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """Matriz de decisión comercial agregada por segmento."""
    order = ["A: Alto Valor / Alto Riesgo", "B: Medio / Riesgo Moderado", "C: No Rentable", "D: Estable"]
    g = scored.groupby("segmento")
    mat = pd.DataFrame({
        "Clientes": g.size(),
        "Prob. mínima": g["prob_fuga"].min(),
        "Prob. media": g["prob_fuga"].mean(),
        "Saldo medio": g["Balance"].mean(),
        "Margen cliente medio": g["margen_cliente"].mean(),
        "Costo total": g["costo_incentivo"].sum(),
        "Ganancia neta esperada": g["ganancia_neta_esperada"].sum(),
        "Incentivo principal": g["incentivo"].agg(lambda s: s.value_counts().index[0]),
    }).reindex(order)
    return mat.dropna(how="all")
