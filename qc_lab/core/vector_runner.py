"""
Ejecucion del dialecto VECTOR: `run_strategy(data, ...) -> DataFrame`.

El proyecto arrastra dos formas de escribir una estrategia:

  * LEAN   -> clase que hereda de `QCAlgorithm`. Es el dialecto nativo de qc_lab
              y lo ejecuta `engine.ejecutar_backtest`.
  * VECTOR -> funcion `run_strategy(data, ...)` que recibe un DataFrame de
              precios y devuelve las columnas `Signal` / `Position`. Es el
              dialecto que historicamente ejecutaba `app.py` con su propio motor.

Este modulo hace que el segundo dialecto pase por el MISMO data feed, la MISMA
contabilidad y las MISMAS estadisticas que el primero, de modo que cualquier
estrategia del proyecto -venga del dialecto que venga- devuelve exactamente el
mismo `ResultadoBacktest` y por tanto las mismas 37 metricas.

La contabilidad no es un `cumprod` sobre retornos: se mantiene caja y titulos
explicitamente y las comisiones se descuentan del efectivo en cada ejecucion, tal
como hace el motor Lean. Eso permite reconstruir operaciones FIFO exactas y un
beneficio no realizado correcto (`valor de mercado - coste`).

La decision de exposicion de la barra `t` se aplica al CIERRE de `t-1`: la
funcion ha visto los precios hasta `t-1` y la orden se ejecuta a ese precio de
cierre, de modo que el retorno que captura esa exposicion es el de `t`. Es la
misma convencion anti-look-ahead del motor Lean.
"""

from __future__ import annotations

import inspect
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qc_lab.core import datafeed
from qc_lab.core.analytics import calcular_estadisticas
from qc_lab.core.strategy_parser import EstrategiaInfo

if TYPE_CHECKING:  # pragma: no cover
    from qc_lab.core.engine import ResultadoBacktest

# Comisión por defecto de una ejecución: 0.05 % del nocional, el mismo valor que
# usaba el motor legacy de app.py para que las comparativas no cambien de escala.
COMISION_POR_DEFECTO = 0.0005

# Tolerancia de nocional por debajo de la cual un rebalanceo se considera nulo.
_TOLERANCIA = 1e-9


class ErrorVector(ValueError):
    """Fallo de negocio ya redactado para mostrarse tal cual en la interfaz."""


# --------------------------------------------------------------------------- #
# Carga de la funcion de entrada
# --------------------------------------------------------------------------- #
def cargar_run_strategy(fuente: str, etiqueta: str = "La estrategia") -> Any:
    """Compila `fuente` en un espacio aislado y devuelve su `run_strategy`."""
    if not (fuente or "").strip():
        raise ErrorVector(f"{etiqueta} está vacío.")

    espacio: Dict[str, Any] = {"__name__": "__qc_vector__", "__file__": f"<{etiqueta}>"}
    try:
        exec(compile(fuente, f"<{etiqueta}>", "exec"), espacio)  # noqa: S102
    except SyntaxError as exc:
        raise ErrorVector(f"Error de sintaxis en la línea {exc.lineno}: {exc.msg}") from exc
    except ModuleNotFoundError as exc:
        # Un .py exportado de QuanConnect depende del runtime de AlgorithmImports.
        raise ErrorVector(
            f"{etiqueta} usa `{exc.name}` de QuantConnect, que solo existe dentro del "
            "runtime de Lean. Cópiala al editor y expón la lógica como "
            "`run_strategy(data, fast_ma, slow_ma)` para poder compararla aquí."
        ) from exc
    except Exception as exc:
        raise ErrorVector(f"{etiqueta} falló al definirse: {type(exc).__name__}: {exc}") from exc

    entrada = espacio.get("run_strategy")
    if not callable(entrada):
        # Si el runtime de Lean ya esta instalado en el proceso (porque antes se
        # ejecuto una estrategia Lean), el `from AlgorithmImports import *` de un
        # borrador NO falla: simplemente no aparece `run_strategy`. Sin este aviso
        # el usuario veria un error generico en vez de la pista que si le sirve.
        if "AlgorithmImports" in fuente:
            raise ErrorVector(
                f"{etiqueta} usa `AlgorithmImports` de QuantConnect, que solo existe "
                "dentro del runtime de Lean. Cópiala al editor y expón la lógica como "
                "`run_strategy(data, fast_ma, slow_ma)` para poder compararla aquí."
            )
        raise ErrorVector(f"{etiqueta} no define una función `run_strategy(data, ...)`.")
    return entrada


def _kwargs_admitidos(entrada: Any, parametros: Dict[str, Any]) -> Dict[str, Any]:
    """Se queda solo con los parametros que la firma de `run_strategy` acepta."""
    if not parametros:
        return {}
    try:
        nombres = set(inspect.signature(entrada).parameters)
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in parametros.items() if k in nombres}


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def _marco_precios(ticker: str, inicio: str, fin: str, semilla: int) -> pd.DataFrame:
    """OHLCV del data feed de qc_lab, con los alias en mayuscula que espera el dialecto vector."""
    barras = datafeed.daily_bars(ticker, inicio, fin, semilla)
    if barras.empty:
        raise ErrorVector(
            f"El data feed no tiene velas de {ticker} entre {inicio} y {fin}."
        )
    marco = barras.rename(columns=str.capitalize)
    marco["Date"] = marco.index
    return marco


# --------------------------------------------------------------------------- #
# Posicion objetivo
# --------------------------------------------------------------------------- #
def _posicion_objetivo(salida: pd.DataFrame) -> pd.Series:
    """
    Deriva la fraccion de cartera objetivo en cada barra.

    Prioridad: `Position` si existe (la estrategia ya fijo la exposicion), si no
    `Signal` desplazada una barra para no mirar el futuro, y si no ninguna de las
    dos, la exposicion por defecto de mantener la posicion.
    """
    if "Position" in salida.columns:
        posicion = pd.to_numeric(salida["Position"], errors="coerce")
    elif "Signal" in salida.columns:
        posicion = pd.to_numeric(salida["Signal"], errors="coerce").shift(1)
    else:
        posicion = pd.Series(1.0, index=salida.index)
    return posicion.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)


# --------------------------------------------------------------------------- #
# Simulacion de la cartera
# --------------------------------------------------------------------------- #
class _Cartera:
    """
    Caja + titulos a mercado con comisiones por ejecucion y lotes FIFO.

    Reproduce las mismas magnitudes que expone el motor Lean en `curva`:
    equity, cash, holdings, invested, margin, fees, warmup, realizado y coste.
    """

    def __init__(self, capital: float, comision_rate: float) -> None:
        self.efectivo = float(capital)
        self.capital = float(capital)
        self.titulos = 0.0
        self.comision_rate = float(comision_rate)
        self.comisiones = 0.0
        self.realizado = 0.0
        # Lotes abiertos: [fecha, cantidad, precio]
        self.lotes: List[List[Any]] = []

    # --- contabilidad -------------------------------------------------- #
    def valor(self, precio: float) -> Tuple[float, float, float]:
        """Devuelve (equity, efectivo, valor de las posiciones) al precio dado."""
        posiciones = self.titulos * precio
        return self.efectivo + posiciones, self.efectivo, posiciones

    def coste_basis(self) -> float:
        """Coste de mercado de las posiciones abiertas (para el no realizado exacto)."""
        return sum(lote[1] * lote[2] for lote in self.lotes)

    # --- ejecucion ----------------------------------------------------- #
    def operar(self, objetivo: float, precio: float, fecha: Any,
               simbolo: str, ejecuciones: List[tuple], operaciones: List[tuple]) -> None:
        """
        Rebalancia a `objetivo` de la cartera al precio dado.

        Emite un solo evento por barra con el delta total de la reasignacion, que
        es exactamente lo que hace `SetHoldings` en Lean: un unico envio de orden
        por decision, no uno por titulo comprado y otro por vendido.
        """
        _, _, posiciones = self.valor(precio)
        patrimonio = self.efectivo + posiciones
        delta = patrimonio * objetivo - posiciones
        if abs(delta) <= _TOLERANCIA or precio <= 0:
            return

        comision = self.comision_rate * abs(delta)
        cantidad = delta / precio
        if cantidad < 0:
            pnl = self._cerrar_lotes(-cantidad, precio, fecha, simbolo, operaciones, comision)
            self.realizado += pnl
        else:
            self.lotes.append([fecha, cantidad, precio])

        self.titulos += cantidad
        self.efectivo -= delta + comision
        self.comisiones += comision
        ejecuciones.append((
            fecha, simbolo, "BUY" if delta > 0 else "SELL", cantidad, precio,
            delta, comision, self.efectivo,
        ))

    def _cerrar_lotes(self, cantidad: float, precio: float, fecha: Any, simbolo: str,
                      operaciones: List[tuple], comision: float) -> float:
        """Liquida lotes FIFO. La comision de la barra se prorratea entre ellos."""
        restante = cantidad
        prorrateo = comision * (cantidad / self.titulos if self.titulos else 1.0)
        pnl_total = 0.0
        while restante > _TOLERANCIA and self.lotes:
            fecha_abierta, cantidad_abierta, precio_abierto = self.lotes[0]
            usado = min(restante, cantidad_abierta)
            parte = prorrateo * (usado / cantidad) if cantidad else prorrateo
            pnl = (precio - precio_abierto) * usado - parte
            pnl_total += pnl
            operaciones.append((
                fecha_abierta, fecha, simbolo, usado,
                precio_abierto, precio, pnl,
            ))
            restante -= usado
            if usado >= cantidad_abierta - _TOLERANCIA:
                self.lotes.pop(0)
            else:
                self.lotes[0][1] = cantidad_abierta - usado
        return pnl_total


def _simular(precios: pd.Series, objetivo: pd.Series, capital: float,
             comision_rate: float, simbolo: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recorre el historico aplicando la exposicion objetivo y devuelve
    (curva, ejecuciones, operaciones) con el esquema que consume analytics.
    """
    cartera = _Cartera(capital, comision_rate)
    filas: List[tuple] = []
    ejecuciones: List[tuple] = []
    operaciones: List[tuple] = []

    objetivos = objetivo.reindex(precios.index).fillna(0.0).to_numpy(dtype=float)
    cierre = precios.to_numpy(dtype=float)
    fechas = precios.index

    for i, momento in enumerate(fechas):
        # La orden se decide con los datos hasta la barra anterior y se ejecuta a
        # ese cierre: por eso el objetivo de `i` es el de `i - 1`.
        objetivo_barra = objetivos[i - 1] if i > 0 else 0.0
        precio_orden = cierre[i - 1] if i > 0 else cierre[i]
        cartera.operar(float(objetivo_barra), float(precio_orden), momento, simbolo,
                       ejecuciones, operaciones)

        equity, efectivo, posiciones = cartera.valor(float(cierre[i]))
        filas.append((
            momento, equity, efectivo, posiciones, bool(posiciones > 0),
            0.0, cartera.comisiones, False, cartera.realizado, cartera.coste_basis(),
        ))

    curva = pd.DataFrame(
        filas,
        columns=["time", "equity", "cash", "holdings", "invested",
                 "margin", "fees", "warmup", "realizado", "coste"],
    ).set_index("time")
    fills = pd.DataFrame(
        ejecuciones,
        columns=["time", "symbol", "side", "qty", "price", "value", "fee", "cash"],
    )
    trades = pd.DataFrame(
        operaciones,
        columns=["open_time", "close_time", "symbol", "qty", "open_price", "close_price", "pnl"],
    )
    return curva, fills, trades


# --------------------------------------------------------------------------- #
# Ejecucion
# --------------------------------------------------------------------------- #
def ejecutar_backtest_vector(
    fuente: str,
    info: EstrategiaInfo,
    parametros: Optional[Dict[str, Any]] = None,
    capital: Optional[float] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    simbolo: Optional[str] = None,
    benchmark: Optional[str] = None,
    comision_rate: float = COMISION_POR_DEFECTO,
    semilla: int = datafeed.DEFAULT_SEED,
    etiqueta: str = "La estrategia",
) -> "ResultadoBacktest":
    """
    Ejecuta una estrategia `run_strategy` y devuelve un `ResultadoBacktest`.

    La firma replica la de `engine.ejecutar_backtest` para que el despachador
    del motor pueda tratar los dos dialectos igual.
    """
    from qc_lab.core.engine import ResultadoBacktest

    inicio = time.perf_counter()
    logs: List[Dict[str, Any]] = []

    def log(nivel: str, mensaje: str, momento: Any = None) -> None:
        logs.append({"tiempo": momento, "nivel": nivel, "mensaje": mensaje, "orden": len(logs)})

    def fallo(error: str, traza: str = "") -> ResultadoBacktest:
        res = ResultadoBacktest(info=info, exito=False, error=error, traza=traza, logs=logs)
        res.duracion_s = time.perf_counter() - inicio
        return res

    # --- 1) funcion de entrada ---------------------------------------- #
    try:
        entrada = cargar_run_strategy(fuente, etiqueta)
    except ErrorVector as exc:
        log("ERROR", str(exc))
        return fallo(str(exc))

    # --- 2) configuracion efectiva ------------------------------------ #
    capital_efectivo = float(capital if capital is not None else (info.capital or 100000.0))
    inicio_efectivo = str(fecha_inicio or info.fecha_inicio or "2020-01-01")
    fin_efectivo = str(fecha_fin or info.fecha_fin or "2026-09-30")
    universo = datafeed.available_tickers()
    ticker = (simbolo or (info.simbolos[0] if info.simbolos else "SPY")).upper()
    if ticker not in universo:
        return fallo(
            f"{ticker} no está en el universo de datos de QuantLab. "
            f"Disponibles: {', '.join(universo)}."
        )
    bench = (benchmark or info.benchmark_efectivo or "SPY").upper()
    if bench not in universo:
        bench = "SPY"

    log("INFO", f"Dialecto vector: {ticker} de {inicio_efectivo} a {fin_efectivo}, "
                f"capital {capital_efectivo:,.2f}, comision {comision_rate:.4%}")

    # --- 3) datos ------------------------------------------------------ #
    try:
        marco = _marco_precios(ticker, inicio_efectivo, fin_efectivo, semilla)
    except ErrorVector as exc:
        log("ERROR", str(exc))
        return fallo(str(exc))

    # --- 4) ejecucion de la estrategia -------------------------------- #
    kwargs = _kwargs_admitidos(entrada, dict(parametros or {}))
    if kwargs:
        log("INFO", f"Parametros aplicados: " + ", ".join(f"{k}={v}" for k, v in kwargs.items()))
    try:
        salida = entrada(marco, **kwargs)
    except TypeError:
        # La firma puede no aceptar los parametros opcionales: se reintenta limpio.
        try:
            salida = entrada(marco)
            kwargs = {}
        except Exception as exc:
            log("ERROR", f"run_strategy fallo: {type(exc).__name__}: {exc}")
            return fallo(f"`run_strategy` falló: {type(exc).__name__}: {exc}",
                         traceback.format_exc()[-3000:])
    except Exception as exc:
        log("ERROR", f"run_strategy fallo: {type(exc).__name__}: {exc}")
        return fallo(f"`run_strategy` falló: {type(exc).__name__}: {exc}",
                     traceback.format_exc()[-3000:])

    if not isinstance(salida, pd.DataFrame):
        return fallo("`run_strategy` debe devolver un DataFrame de pandas.")
    if salida.empty:
        return fallo("`run_strategy` devolvio un DataFrame sin filas.")

    # --- 5) contabilidad ----------------------------------------------- #
    precios = pd.to_numeric(salida["Close"], errors="coerce")
    precios.index = salida.index
    objetivo = _posicion_objetivo(salida)
    curva, fills, trades = _simular(precios.dropna(), objetivo, capital_efectivo,
                                    comision_rate, ticker)
    if curva.empty:
        return fallo("La serie de precios quedo vacia tras limpiar.")

    # --- 6) series de referencia y estadisticas ----------------------- #
    ventana = curva.index
    benchmark_serie = pd.Series(dtype=float)
    frame_b = datafeed.daily_bars(bench, inicio_efectivo, fin_efectivo, semilla)
    if not frame_b.empty:
        benchmark_serie = frame_b["close"].astype(float).reindex(ventana).ffill().dropna()

    estadisticas = calcular_estadisticas(
        curva=curva, fills=fills, trades=trades, info=info,
        benchmark=benchmark_serie, capital_inicial=capital_efectivo,
        ejecucion="Daily", warmup=0,
    )

    # El diagnostico conserva lo que la estrategia calculo, para inspeccionarlo
    # sin volver a ejecutar el codigo.
    diagnostico = salida.select_dtypes(include=[np.number]).copy()
    diagnostico = diagnostico.reindex(curva.index)

    log("INFO", f"Backtest vectorial completado: {len(curva):,} barras, "
                f"{len(fills)} ejecuciones, {len(trades)} operaciones cerradas")

    return ResultadoBacktest(
        info=info,
        exito=True,
        curva=curva,
        operaciones=trades,
        ejecuciones=fills,
        estadisticas=estadisticas,
        logs=logs,
        plots={},
        benchmark=benchmark_serie,
        precio_activo=precios.reindex(curva.index).ffill(),
        diagnostico=diagnostico,
        duracion_s=time.perf_counter() - inicio,
        barras_evaluadas=len(curva),
    )