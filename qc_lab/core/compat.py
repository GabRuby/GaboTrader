"""
Puente de compatibilidad entre QuantLab y la interfaz heredada de `app.py`.

La UI actual (graficas, tabla comparativa, panel de IA y zona de borrador) consume
un contrato muy concreto:

  * Un DataFrame con las columnas `Equity`, `Benchmark`, `Drawdown` y `Position`.
  * Un diccionario de metricas con claves inglesas y porcentajes ya multiplicados.

El motor devuelve algo mas rico y en otro idioma: un `ResultadoBacktest` con 37
metricas en fracciones y series en resolucion de mercado. Este modulo traduce UNO
en OTRO sin recalcular nada: todas las cifras salen de `estadisticas`, de modo que
la interfaz y el motor nunca pueden discrepar.

Es codigo de andamiaje deliberadamente. Su unico trabajo es que la migracion sea
progresiva: cuando la UI nueva se apoye directamente en `ResultadoBacktest`, este
archivo desaparece entero.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from qc_lab.core.engine import ResultadoBacktest


def _pct(fraccion: Any) -> float:
    """De fraccion a porcentaje, que es la unidad que usa el contrato heredado."""
    try:
        valor = float(fraccion)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(valor):
        return 0.0
    return valor * 100.0


def _alinear_diario(serie: pd.Series, indice: pd.DatetimeIndex, capital: float) -> pd.Series:
    """
    Lleva una serie de PRECIOS a una curva de capital diaria iniciada en `capital`.

    La alineacion es por dia, no por marca de tiempo exacta: una serie diaria de
    referencia tiene que poder pintarse encima de una curva que viene en minuto.
    """
    if serie is None or serie.empty:
        return pd.Series(dtype=float)
    precios = pd.Series(
        pd.to_numeric(serie, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(serie.index)).normalize(),
    )
    precios = precios[~precios.index.duplicated(keep="last")].dropna().sort_index()
    if precios.empty:
        return pd.Series(dtype=float)

    destino = pd.DatetimeIndex(pd.to_datetime(indice)).normalize()
    alineado = precios.reindex(destino.union(precios.index)).ffill().reindex(destino).dropna()
    if alineado.empty or not alineado.iloc[0]:
        return pd.Series(dtype=float)
    return alineado * (capital / alineado.iloc[0])


def marco_heredado(
    res: ResultadoBacktest,
    capital: float,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame `Equity / Benchmark / Drawdown / Position` que espera la UI.

    La serie se entrega DIARIA aunque el motor haya corrido en minuto: es la
    granularidad con la que estan calculadas las metricas del panel y la que la
    grafica histórica dibujaba. La curva completa queda en `res.curva`.

    `Benchmark` es el buy & hold DEL ACTIVO NEGOCIADO, que es la referencia que la
    interfaz muestra junto a cada estrategia ("Benchmark (TSLA)"). Es una eleccion
    distinta de la que usa `estadisticas` para beta y alfa, que si compara contra
    el indice: aqui se mantiene la semantica visible del panel.
    """
    diario = res.curva["equity"].astype(float).resample("D").last().dropna()
    marco = pd.DataFrame({"Equity": diario})
    marco["Drawdown"] = (marco["Equity"] / marco["Equity"].cummax() - 1.0) * 100.0

    referencia = benchmark if benchmark is not None else res.precio_activo
    alineado = _alinear_diario(referencia, marco.index, capital)
    marco["Benchmark"] = alineado if not alineado.empty else marco["Equity"]

    # La posicion se expone como 0/1 porque la tabla y las vistas solo necesitan
    # saber cuanto esta invertido; el detalle viene en `res.diagnostico`.
    if "invested" in res.curva.columns:
        posicion = res.curva["invested"].astype(float).resample("D").last()
    else:
        posicion = pd.Series(1.0, index=res.curva.index)
    marco["Position"] = posicion.reindex(marco.index).ffill().fillna(0.0)

    return marco


def metricas_heredadas(
    res: ResultadoBacktest,
    *,
    simbolo: str,
    capital: float,
    parametros: Optional[Dict[str, Any]] = None,
    comision_rate: float = 0.0005,
) -> Dict[str, Any]:
    """
    Traduce las 37 metricas del motor al diccionario de 17 claves de la UI.

    Todas las claves que la interfaz ya leia siguen existiendo con el MISMO
    significado; lo que cambia es la unidad (porcentaje en vez de fraccion) y que
    las comisiones y el volumen salen de ejecuciones reales en lugar de estimarse.
    """
    e = res.estadisticas or {}
    parametros = dict(parametros or {})

    def _int(nombre: str, defecto: int = 0) -> int:
        try:
            return int(round(float(parametros.get(nombre, defecto))))
        except (TypeError, ValueError):
            return defecto

    marco = marco_heredado(res, capital)
    return {
        # --- resultado ------------------------------------------------ #
        "equity": float(e.get("equity_final", capital)),
        "net_profit": float(e.get("beneficio_neto", 0.0)),
        "total_return": _pct(e.get("retorno_total", 0.0)),
        "max_drawdown": _pct(e.get("dd_max", 0.0)),
        "win_rate": _pct(e.get("win_rate", 0.0)),
        "sharpe": float(e.get("sharpe", 0.0) or 0.0),

        # --- actividad ------------------------------------------------ #
        "trades": int(e.get("n_operaciones", 0)),
        "fills": int(e.get("n_ejecuciones", 0)),
        "fees": float(e.get("comisiones", 0.0)),
        "volume": float(e.get("volumen", 0.0)),
        "bars": int(res.barras_evaluadas or len(marco)),

        # --- contexto ------------------------------------------------- #
        "source": res.info.dialecto,
        "ticker": simbolo,
        "capital": float(capital),
        "fast_ma": _int("fast_ma"),
        "slow_ma": _int("slow_ma"),
        "comision_rate": float(comision_rate),

        # --- benchmark (buy & hold del activo negociado) -------------- #
        "benchmark_return": _pct(_retorno_marco(marco, "Benchmark")),
        "benchmark_drawdown": _pct(_drawdown_marco(marco, "Benchmark")),
    }


def _retorno_marco(marco: pd.DataFrame, columna: str) -> float:
    """Retorno acumulado en fraccion de una columna del marco heredado."""
    if columna not in marco or marco.empty:
        return 0.0
    serie = marco[columna].astype(float).dropna()
    if len(serie) < 2 or not serie.iloc[0]:
        return 0.0
    return float(serie.iloc[-1] / serie.iloc[0] - 1.0)


def _drawdown_marco(marco: pd.DataFrame, columna: str) -> float:
    """Drawdown maximo en fraccion de una columna del marco heredado."""
    if columna not in marco or marco.empty:
        return 0.0
    serie = marco[columna].astype(float).dropna()
    if serie.empty:
        return 0.0
    return float((serie / serie.cummax() - 1.0).min())


def contrato_heredado(
    res: ResultadoBacktest,
    *,
    simbolo: str,
    capital: float,
    parametros: Optional[Dict[str, Any]] = None,
    comision_rate: float = 0.0005,
    benchmark: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Devuelve el par `(df, metrics)` que la interfaz heredada ya sabe leer."""
    if not res.exito:
        raise ValueError(res.error or "La estrategia no se pudo ejecutar.")
    marco = marco_heredado(res, capital, benchmark=benchmark)
    metricas = metricas_heredadas(
        res, simbolo=simbolo, capital=capital, parametros=parametros, comision_rate=comision_rate,
    )
    return marco, metricas