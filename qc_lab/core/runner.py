"""
Fachada unica de ejecucion de QuantLab.

Este modulo es la puerta de entrada que deben usar TODOS los consumidores
(estrategia principal, comparativas y borrador). Encapsula las decisiones que
hoy estan repartidas por el codigo y que el motor debe respetar siempre igual:

  * Que estrategia se ejecuta: un `.py` del proyecto o un borrador en memoria.
  * Que configuracion se aplica: capital, fechas, comision, apalancamiento.
  * Que parametros llegan al codigo de la estrategia.
  * Que simbolo y benchmark se usan cuando la estrategia no los declara.

Regla de la casa: `qc_lab` es la unica fuente de verdad de la ejecucion. Este
modulo NO simula nada; elige dialecto, aplica la configuracion y devuelve el
mismo `ResultadoBacktest` que produce `engine.ejecutar_backtest`, tanto si la
estrategia es Lean como si es vectorial.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from qc_lab.core import datafeed
from qc_lab.core.engine import ResultadoBacktest, ejecutar_backtest
from qc_lab.core.strategy_parser import (
    DIALECTO_VECTOR,
    EstrategiaInfo,
    analizar_archivo,
    analizar_fuente,
)

# Periodo por defecto cuando la estrategia no declara fechas.
INICIO_POR_DEFECTO = "2020-01-01"
FIN_POR_DEFECTO = "2026-09-30"
CAPITAL_POR_DEFECTO = 100000.0


@dataclass
class ConfigEjecucion:
    """
    Configuracion comun de una corrida, sea del dialecto que sea.

    Los campos en `None` significan "deja lo que la propia estrategia decida":
    una estrategia Lean que fija `set_cash` o `set_start_date` manda sobre el
    valor por defecto, igual que en el motor original.
    """

    simbolo: Optional[str] = None
    capital: Optional[float] = None
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    apalancamiento: float = 1.0
    comision: Optional[float] = None
    benchmark: Optional[str] = None
    parametros: Dict[str, Any] = field(default_factory=dict)
    semilla: int = datafeed.DEFAULT_SEED
    etiqueta: Optional[str] = None

    def resuelta(self, info: EstrategiaInfo) -> "ConfigEjecucion":
        """Completa los huecos con lo que declare la estrategia."""
        return ConfigEjecucion(
            simbolo=(self.simbolo or (info.simbolos[0] if info.simbolos else "SPY")).upper(),
            capital=float(self.capital if self.capital is not None
                          else (info.capital or CAPITAL_POR_DEFECTO)),
            fecha_inicio=str(self.fecha_inicio or info.fecha_inicio or INICIO_POR_DEFECTO),
            fecha_fin=str(self.fecha_fin or info.fecha_fin or FIN_POR_DEFECTO),
            apalancamiento=float(self.apalancamiento or 1.0),
            comision=self.comision,
            benchmark=(self.benchmark or info.benchmark_efectivo or "SPY").upper(),
            parametros=dict(self.parametros or {}),
            semilla=self.semilla,
            etiqueta=self.etiqueta or info.nombre_archivo,
        )


def _verificar_simbolo(simbolo: str) -> Optional[str]:
    """Error de negocio si el simbolo no existe en el data feed de QuantLab."""
    if simbolo in datafeed.available_tickers():
        return None
    return (
        f"{simbolo} no está en el universo de datos de QuantLab. "
        f"Disponibles: {', '.join(datafeed.available_tickers())}."
    )


def ejecutar_estrategia(
    ruta: Optional[Union[str, Path]] = None,
    *,
    fuente: Optional[str] = None,
    config: Optional[ConfigEjecucion] = None,
    info: Optional[EstrategiaInfo] = None,
) -> ResultadoBacktest:
    """
    Ejecuta una estrategia y devuelve siempre un `ResultadoBacktest`.

    Hay que pasar `ruta` (archivo del proyecto) o `fuente` (codigo en memoria,
    por ejemplo el borrador de la IA). Nunca lanza por un fallo de negocio: un
    archivo con un error debe produzir un resultado con `exito=False` y su
    mensaje, para que la interfaz pueda mostrarlo sin romperse.
    """
    cfg = config or ConfigEjecucion()

    # --- 1) localizar la estrategia ----------------------------------- #
    if ruta is None and not (fuente or "").strip():
        etiqueta = (config.etiqueta if config else None) or "La estrategia"
        return ResultadoBacktest(info=info or EstrategiaInfo(), exito=False,
                                error=f"{etiqueta}: el código está vacío.")

    if ruta is not None:
        ruta = Path(ruta)
        info = info or analizar_archivo(ruta)
    else:
        nombre = cfg.etiqueta or "borrador"
        info = info or analizar_fuente(fuente or "", nombre=nombre)
        if not info.valido:
            # El dialecto se fuerza a vectorial para que el ejecutor produzca el
            # error REAL ("no compila" frente a "no define run_strategy") en lugar
            # de un mensaje generico del analizador.
            info.dialecto = DIALECTO_VECTOR

    cfg = cfg.resuelta(info)

    # --- 2) validacion del universo ------------------------------------ #
    if info.dialecto == DIALECTO_VECTOR:
        problema = _verificar_simbolo(cfg.simbolo or "SPY")
        if problema:
            return ResultadoBacktest(info=info, exito=False, error=problema)

    # --- 3) ejecucion -------------------------------------------------- #
    if ruta is not None:
        return ejecutar_backtest(
            ruta,
            parametros=cfg.parametros,
            capital=cfg.capital,
            fecha_inicio=cfg.fecha_inicio,
            fecha_fin=cfg.fecha_fin,
            apalancamiento=cfg.apalancamiento,
            semilla=cfg.semilla,
            info=info,
            simbolo=cfg.simbolo,
            benchmark=cfg.benchmark,
            comision=cfg.comision,
        )

    # Un borrador es codigo en memoria. El motor Lean espera una clase, asi que
    # se materializa en un temporal: el ejecutor de archivos queda igual de
    # simple y el borrador puede ser de cualquiera de los dos dialectos.
    with tempfile.TemporaryDirectory(prefix="quantlab_borrador_") as temporal:
        ruta_temporal = Path(temporal) / f"{info.nombre_archivo or 'borrador'}.py"
        ruta_temporal.write_text(fuente or "", encoding="utf-8")
        if info.dialecto == DIALECTO_VECTOR:
            from qc_lab.core.vector_runner import ejecutar_backtest_vector

            return ejecutar_backtest_vector(
                fuente or "",
                info=info,
                parametros=cfg.parametros,
                capital=cfg.capital,
                fecha_inicio=cfg.fecha_inicio,
                fecha_fin=cfg.fecha_fin,
                simbolo=cfg.simbolo,
                benchmark=cfg.benchmark,
                comision_rate=cfg.comision if cfg.comision is not None else 0.0005,
                semilla=cfg.semilla,
                etiqueta=cfg.etiqueta or "La estrategia",
            )
        return ejecutar_backtest(
            ruta_temporal,
            parametros=cfg.parametros,
            capital=cfg.capital,
            fecha_inicio=cfg.fecha_inicio,
            fecha_fin=cfg.fecha_fin,
            apalancamiento=cfg.apalancamiento,
            semilla=cfg.semilla,
            info=info,
            simbolo=cfg.simbolo,
            benchmark=cfg.benchmark,
            comision=cfg.comision,
        )


def ejecutar_fuente(
    fuente: str,
    *,
    etiqueta: str = "La estrategia",
    **campos: Any,
) -> ResultadoBacktest:
    """Atajo para codigo en memoria (borrador de la IA)."""
    return ejecutar_estrategia(fuente=fuente, config=ConfigEjecucion(etiqueta=etiqueta, **campos))