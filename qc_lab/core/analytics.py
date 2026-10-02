"""
Estadisticas de rendimiento y construccion de series para los graficos.

Incluye las metricas que QuantConnect muestra en su barra de resultados:

    Equity, Total Return, Net Profit, Fees, Volume, PSR, Sharpe, Sortino,
    Max Drawdown, CAGR, Calmar, VaR/CVaR, Profit Factor, Win Rate, Exposure.

Todas se calculan sobre retornos DIARIOS. Si la curva llega en resolucion de
minuto se reduce a diario primero: mezclar granularidades daria Sharpe y PSR sin
sentido.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DIAS_ANIO = 252


# --------------------------------------------------------------------------- #
# Utilidades estadisticas
# --------------------------------------------------------------------------- #
def _normal_cdf(x: float) -> float:
    """Funcion de distribucion acumulada normal estandar (via erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def psr(retornos: np.ndarray, sharpe: float, sr_objetivo: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado).

        PSR = Phi( (SR - SR0) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2) )

    Devuelve la probabilidad de que el Sharpe real sea superior al objetivo
    `sr_objetivo`. Con `sr_objetivo = 0` responde a la pregunta habitual:
    "¿hay evidencia estadistica de que esta estrategia beats el cero?".
    """
    n = retornos.size
    if n < 3 or not np.isfinite(sharpe):
        return float("nan")
    media = float(np.mean(retornos))
    sigma = float(np.std(retornos, ddof=1))
    if sigma <= 0:
        return float("nan")

    # Asimetria y curtosis (curtosis NO centrada en exceso).
    std = sigma
    g3 = float(np.mean(((retornos - media) / std) ** 3))
    g4 = float(np.mean(((retornos - media) / std) ** 4))

    denom = 1.0 - g3 * sharpe + ((g4 - 1.0) / 4.0) * (sharpe**2)
    if denom <= 0:
        return float("nan")
    return _normal_cdf((sharpe - sr_objetivo) * math.sqrt(n - 1) / math.sqrt(denom))


def max_drawdown(equity: pd.Series) -> tuple[float, Any, Any]:
    """Drawdown maximo (negativo) y las fechas de pico y valle."""
    if equity.empty:
        return 0.0, None, None
    pico = equity.cummax()
    dd = equity / pico - 1.0
    valle = dd.idxmin()
    valor = float(dd.min())
    pico_fecha = equity.loc[:valle].idxmax() if valle is not None else None
    return valor, pico_fecha, valle


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Serie de drawdown en cada instante (0 en los maximos historicos)."""
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def _retornos_diarios(curva: pd.DataFrame) -> pd.Series:
    """Retornos diarios a partir de la curva de equity."""
    if curva.empty or "equity" not in curva.columns:
        return pd.Series(dtype=float)
    equity = curva["equity"].astype(float)
    if equity.index.freqstr is None:
        # Reduccion a diario: se toma el ultimo valor de cada sesion.
        diario = equity.resample("D").last().dropna()
    else:
        diario = equity
    return diario.pct_change().dropna()


# --------------------------------------------------------------------------- #
# Serie principal de equity
# --------------------------------------------------------------------------- #
def curva_principal(curva: pd.DataFrame) -> pd.Series:
    """Equity diario de la estrategia (fin de cada sesion)."""
    if curva.empty or "equity" not in curva.columns:
        return pd.Series(dtype=float)
    return curva["equity"].astype(float).resample("D").last().dropna()


def serie_drawdown(curva: pd.DataFrame) -> pd.Series:
    """Drawdown diario en porcentaje."""
    equity = curva_principal(curva)
    if equity.empty:
        return equity
    return drawdown_series(equity) * 100.0


def serie_exposicion(curva: pd.DataFrame) -> pd.Series:
    """Porcentaje del capital invertido (0-100)."""
    if curva.empty or "holdings" not in curva.columns:
        return pd.Series(dtype=float)
    base = curva[["holdings", "equity"]].resample("D").last().dropna()
    denom = base["equity"].replace(0.0, np.nan)
    return (base["holdings"] / denom * 100.0).ffill().fillna(0.0)


def serie_margen(curva: pd.DataFrame) -> pd.DataFrame:
    """Margen de cartera y reparto caja / invertido, en dolares."""
    if curva.empty or "margin" not in curva.columns:
        return pd.DataFrame()
    base = curva[["equity", "cash", "holdings", "margin"]].resample("D").last().dropna()
    base["cash_pct"] = np.where(base["equity"] > 0, base["cash"] / base["equity"] * 100.0, 0.0)
    return base


# --------------------------------------------------------------------------- #
# Estadisticas completas
# --------------------------------------------------------------------------- #
def calcular_estadisticas(
    curva: pd.DataFrame,
    fills: pd.DataFrame,
    trades: pd.DataFrame,
    info: Any,
    benchmark: pd.Series,
    capital_inicial: float,
    ejecucion: str = "Daily",
    warmup: int = 0,
) -> Dict[str, Any]:
    """Calcula el diccionario completo de metricas que consume el dashboard."""
    capital_inicial = float(capital_inicial or 0.0)
    equity = curva_principal(curva)
    retornos = equity.pct_change().dropna() if not equity.empty else pd.Series(dtype=float)

    if equity.empty:
        return {"error": "Curva de equity vacia"}

    equity_final = float(equity.iloc[-1])
    equity_inicial = float(equity.iloc[0]) if equity.iloc[0] else capital_inicial
    base = capital_inicial if capital_inicial else equity_inicial

    beneficio_neto = equity_final - base
    retorno_total = (equity_final / base - 1.0) if base else 0.0

    # --- riesgo / retorno ---
    n = len(retornos)
    media = float(retornos.mean()) if n else 0.0
    sigma = float(retornos.std(ddof=1)) if n > 1 else 0.0
    sharpe = (media / sigma * math.sqrt(DIAS_ANIO)) if sigma > 0 else 0.0

    downside = retornos[retornos < 0]
    sigma_down = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (media / sigma_down * math.sqrt(DIAS_ANIO)) if sigma_down > 0 else 0.0

    anios = max(n / DIAS_ANIO, 1e-9)
    cagr = ((equity_final / base) ** (1.0 / anios) - 1.0) if base > 0 and equity_final > 0 else -1.0

    dd_max, fecha_pico, fecha_valle = max_drawdown(equity)
    dd_diario = drawdown_series(equity)
    dd_medio = float(dd_diario.mean()) if len(dd_diario) else 0.0
    duracion_max = float(dd_diario.min() / media * DIAS_ANIO) if media > 0 else 0.0
    calmar = (cagr / abs(dd_max)) if dd_max < 0 else 0.0

    var95 = float(np.percentile(retornos, 5)) if n else 0.0
    cvar95 = float(retornos[retornos <= var95].mean()) if n and (retornos <= var95).any() else var95

    # --- PSR y probabilidad de Styled Sharpe --------------------------- #
    valor_psr = psr(retornos.to_numpy(), sharpe, 0.0)

    # --- benchmark ---------------------------------------------------- #
    bench_sharpe = 0.0
    beta = 0.0
    alfa = 0.0
    info_tracking = 0.0
    bench_curva = pd.Series(dtype=float)
    if not benchmark.empty:
        seg = benchmark.loc[:equity.index[-1]].dropna()
        seg = seg.reindex(equity.index, method="ffill").dropna()
        if len(seg) > 5 and not equity.empty:
            bench_curva = seg
            ret_bench = seg.pct_change().dropna()
            comunes = retornos.index.intersection(ret_bench.index)
            if len(comunes) > 5:
                rb = ret_bench.loc[comunes].to_numpy()
                rs = retornos.loc[comunes].to_numpy()
                sb = float(np.std(rb, ddof=1))
                if sb > 0:
                    bench_sharpe = float(np.mean(rb) / sb * math.sqrt(DIAS_ANIO))
                    cov = float(np.cov(rs, rb, ddof=1)[0][1])
                    beta = cov / (sb**2)
                    alfa = float(np.mean(rs) - beta * np.mean(rb)) * DIAS_ANIO
                    dif = rs - rb
                    info_tracking = float(np.mean(dif) / np.std(dif, ddof=1)) if dif.std(ddof=1) > 0 else 0.0

    # --- costes y actividad ------------------------------------------- #
    comision_total = float(curva["fees"].iloc[-1]) if "fees" in curva.columns and not curva.empty else 0.0
    volumen = float(fills["value"].abs().sum()) if not fills.empty else 0.0
    n_ejecuciones = int(len(fills)) if not fills.empty else 0

    # --- operaciones cerradas ----------------------------------------- #
    n_ops = int(len(trades)) if not trades.empty else 0
    if n_ops:
        # Resumen de las operaciones round-trip FIFO.
        pnl = trades["pnl"].astype(float)
        ganancias = float(pnl[pnl > 0].sum())
        perdidas = float(-pnl[pnl < 0].sum())
        profit_factor = (ganancias / perdidas) if perdidas > 0 else float("inf") if ganancias > 0 else 0.0
        win_rate = float((pnl > 0).mean())
        mejor = float(pnl.max())
        peor = float(pnl.min())
        duracion_media = float(
            (pd.to_datetime(trades["close_time"]) - pd.to_datetime(trades["open_time"]))
            .dt.total_seconds().mean() / 86400.0
        )
    else:
        profit_factor = 0.0
        win_rate = 0.0
        mejor = peor = 0.0
        duracion_media = 0.0

    # --- posicion abierta -------------------------------------------- #
    # Ganancia NO REALIZADA de las posiciones aun abiertas.
    #
    # No se puede aproximar con `holdings - cash`: esa resta es el valor de la
    # cartera, no su ganancia. Si el motor ha calculado el coste de las posiciones
    # (`coste`), la formula exacta es valor_de_mercado - coste. La caida a
    # `holdings - cash` solo se conserva para resultados antiguos sin esa columna.
    sin_realizar = 0.0
    if not curva.empty:
        if "coste" in curva.columns:
            sin_realizar = float(curva["holdings"].iloc[-1] - curva["coste"].iloc[-1])
        elif "holdings" in curva.columns and "cash" in curva.columns:
            sin_realizar = float(curva["holdings"].iloc[-1] - curva["cash"].iloc[-1])

    exposicion = serie_exposicion(curva)
    # `serie_exposicion` devuelve 0-100 (unidad natural del grafico); aqui se
    # guarda como FRACCION para que todas las metricas de ratio sean coherentes
    # entre si y se formateen con `fmt_pct`.
    exposicion_media = (float(exposicion.mean()) / 100.0) if len(exposicion) else 0.0

    margen = serie_margen(curva)
    margen_medio = float(margen["margin"].mean()) if len(margen) else 0.0

    return {
        "equity_inicial": base,
        "equity_final": equity_final,
        "beneficio_neto": beneficio_neto,
        "retorno_total": retorno_total,
        "comisiones": comision_total,
        "sin_realizar": sin_realizar,
        "volumen": volumen,
        "n_ejecuciones": n_ejecuciones,
        "turnover": (volumen / base) if base else 0.0,

        "sharpe": sharpe,
        "sortino": sortino,
        "psr": valor_psr,
        "cagr": cagr,
        "calmar": calmar,
        "volatilidad": sigma * math.sqrt(DIAS_ANIO),
        "dd_max": dd_max,
        "dd_medio": dd_medio,
        "dd_fecha_pico": fecha_pico,
        "dd_fecha_valle": fecha_valle,
        "dd_duracion": duracion_max,
        "var95": var95,
        "cvar95": cvar95,
        "sesgo": float((((retornos - media) / sigma) ** 3).mean()) if sigma > 0 else 0.0,

        "sharpe_benchmark": bench_sharpe,
        "beta": beta,
        "alfa": alfa,
        "tracking_error": info_tracking,

        "n_operaciones": n_ops,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "mejor_operacion": mejor,
        "peor_operacion": peor,
        "duracion_media_dias": duracion_media,

        "exposicion_media": exposicion_media,
        "margen_medio": margen_medio,
        "n_periodos": n,
        "ejecucion": ejecucion,
        "warmup_barras": warmup,
    }


# --------------------------------------------------------------------------- #
# Formateo para la interfaz
# --------------------------------------------------------------------------- #
def fmt_dinero(valor: float, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "n/d"
    signo = "-" if valor < 0 else ""
    return f"{signo}${abs(valor):,.{decimales}f}"


def fmt_pct(valor: float, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "n/d"
    return f"{valor * 100:,.{decimales}f}%"


def fmt_num(valor: float, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "n/d"
    return f"{valor:,.{decimales}f}"


def fmt_compacto(valor: float) -> str:
    """Formato compacto para volumen: 1.2M, 850K."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "n/d"
    v = float(valor)
    for limite, sufijo in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= limite:
            return f"{v / limite:,.2f}{sufijo}"
    return f"{v:,.0f}"