"""
Motor de backtest: ejecuta un archivo de estrategia `.py` real.

El flujo completo es:

  1. Analisis estatico del `.py` (fechas, capital, universo, parametros).
  2. Inyeccion de parametros: se reescribe el codigo fuente para sustituir los
     valores que el usuario mueve en los sliders. Asi se prueban 参数 reales sin
     tocar el archivo del usuario en disco.
  3. Importacion del modulo mediante `AlgorithmImports` sintetico y obtencion de
     la clase que herede de `QCAlgorithm`.
  4. `initialize()` en una instancia instrumentada.
  5. Recorrido vela a vela: eventos de calendario, `on_data`, ejecucion de
     ordenes, `on_order_event`, y registro de la curva de capital.
  6. `on_end_of_algorithm()` y calculo de estadisticas.
"""

from __future__ import annotations

import ast
import time as _time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qc_lab.core import datafeed
from qc_lab.core import lean_runtime as lean
from qc_lab.core.strategy_parser import (
    DIALECTO_VECTOR,
    EstrategiaInfo,
    analizar_archivo,
)

# Ventana de datos que se precarga ANTES de la fecha de inicio, por resolucion.
# Es holgada a proposito: cubre el warm-up de LEAN (hasta 210 velas diarias) y las
# ventanas de `history()` mas largas que usan las estrategias.
PRECARGA_DIAS = {"Daily": 800, "Hourly": 120, "Minute": 20}


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoBacktest:
    """Todo lo que produce una ejecucion: curvas, operaciones, metricas y logs."""

    info: EstrategiaInfo
    exito: bool = True
    error: str = ""
    traza: str = ""

    curva: pd.DataFrame = field(default_factory=pd.DataFrame)
    operaciones: pd.DataFrame = field(default_factory=pd.DataFrame)
    ejecuciones: pd.DataFrame = field(default_factory=pd.DataFrame)
    estadisticas: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    plots: Dict[str, Dict[str, list]] = field(default_factory=dict)
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    precio_activo: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    diagnostico: pd.DataFrame = field(default_factory=pd.DataFrame)
    duracion_s: float = 0.0
    barras_evaluadas: int = 0

    @property
    def nombre(self) -> str:
        return self.info.nombre_archivo

    @property
    def clase(self) -> str:
        return self.info.clase


# --------------------------------------------------------------------------- #
# Inyeccion de parametros en el codigo fuente
# --------------------------------------------------------------------------- #
class _Inyector(ast.NodeTransformer):
    """
    Sustituye literales numericos por los valores fijados en los sliders.

    Dos patrones, que son los que usan estas estrategias:

        pct_stop_loss = 0.040                 -> constante local en on_data
        self.max_escalones = 5               -> atributo del algoritmo

    Solo se tocan literales EXACTAMENTE iguales al valor original, de modo que un
    mismo numero usado para dos cosas distintas no se corrompe: si el nombre no
    esta en el diccionario de overrides, el nodo se deja intacto.
    """

    def __init__(self, overrides: Dict[str, float]):
        self.overrides = overrides
        self.aplicados: List[str] = []

    def _reemplazar(self, nodo: ast.AST, nombre: Optional[str]) -> ast.AST:
        if nombre is None or nombre not in self.overrides:
            return nodo
        if not isinstance(nodo, ast.Constant) or not isinstance(nodo.value, (int, float)):
            return nodo
        if isinstance(nodo.value, bool):
            return nodo
        valor = self.overrides[nombre]
        # Solo se sustituye si el literal es del mismo tipo (int -> int).
        if isinstance(nodo.value, int) and not float(valor).is_integer():
            return nodo
        self.aplicados.append(nombre)
        return ast.Constant(value=valor if isinstance(nodo.value, int) else float(valor))

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        destino = node.targets[0]
        if isinstance(destino, ast.Name):
            nombre = destino.id
        elif isinstance(destino, ast.Attribute):
            nombre = destino.attr
        else:
            return node
        nuevo = self._reemplazar(node.value, nombre)
        if nuevo is not node.value:
            node.value = nuevo
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        nombre = None
        if isinstance(node.target, ast.Name):
            nombre = node.target.id
        elif isinstance(node.target, ast.Attribute):
            nombre = node.target.attr
        if nombre:
            nuevo = self._reemplazar(node.value, nombre)
            if nuevo is not node.value:
                node.value = nuevo
        return node


def aplicar_parametros(fuente: str, overrides: Dict[str, float]) -> Tuple[str, List[str]]:
    """Reescribe la fuente aplicando los overrides. Devuelve (fuente, aplicados)."""
    limpios = {k: v for k, v in (overrides or {}).items() if v is not None}
    if not limpios:
        return fuente, []
    arbol = ast.parse(fuente)
    inyector = _Inyector(limpios)
    nuevo = inyector.visit(arbol)
    ast.fix_missing_locations(nuevo)
    try:
        return ast.unparse(nuevo), sorted(set(inyector.aplicados))
    except Exception:
        # Si `ast.unparse` no sabe representar algun nodo, se devuelve el
        # original: es preferible ignorar un override que romper la ejecucion.
        return fuente, []


# --------------------------------------------------------------------------- #
# Contenedores de valores
# --------------------------------------------------------------------------- #
class SecurityContainer(dict):
    """Dict con `contains_key`, igual que `Securities` en LEAN."""

    def contains_key(self, key: Any) -> bool:
        try:
            return lean._as_symbol(key) in self
        except Exception:
            return False

    def get(self, key: Any, default: Any = None):
        try:
            return self[lean._as_symbol(key)]
        except KeyError:
            return default


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class LeanEngine:
    """Recorre el tiempo y ejecuta la estrategia instrumentada."""

    def __init__(self, info: EstrategiaInfo, parameters: Optional[Dict[str, Any]] = None,
                 capital: Optional[float] = None,
                 fecha_inicio: Optional[str] = None,
                 fecha_fin: Optional[str] = None,
                 apalancamiento: float = 1.0,
                 semilla: int = datafeed.DEFAULT_SEED):
        self.info = info
        self.parameters = dict(parameters or {})
        self.capital = float(capital if capital is not None else (info.capital or 100000.0))
        self.fecha_inicio = pd.Timestamp(fecha_inicio or info.fecha_inicio or "2020-01-01")
        self.fecha_fin = pd.Timestamp(fecha_fin or info.fecha_fin or "2026-09-30")
        if self.fecha_fin <= self.fecha_inicio:
            self.fecha_fin = self.fecha_inicio + pd.DateOffset(years=1)
        self.leverage = float(apalancamiento)
        self.semilla = int(semilla)

        # --- servicios que la estrategia consume a traves de `self` ---
        self.portfolio = lean.Portfolio()
        self.securities = SecurityContainer()
        self.schedule = lean.ScheduleManager(self)
        self.date_rules = lean.DateRules
        self.time_rules = lean.TimeRules
        self.current_slice: Optional[lean.Slice] = None

        self._symbols: Dict[str, lean.Symbol] = {}
        self._indicadores: Dict[str, lean.Indicator] = {}
        self._pending: List[lean.Order] = []
        self._orden_id = 0
        self._plots: Dict[str, Dict[str, list]] = {}
        self._logs: List[Dict[str, Any]] = []
        self._current_time: Any = None

        self._cash = self.capital
        self._start_date = self.fecha_inicio
        self._end_date = self.fecha_fin
        self._benchmark: Optional[lean.Symbol] = lean.Symbol(self.info.benchmark_efectivo)
        self._warmup_count = 0
        self._warmup_res = "Daily"
        self.leverage = float(apalancamiento)

        # --- datos precargados (se rellenan bajo demanda) ---
        self._datos: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._resolucion_ejecucion = "Daily"
        self._resoluciones: Dict[lean.Symbol, str] = {}

        # --- registros de salida ---
        self._curva: List[Tuple] = []
        self._fills: List[Tuple] = []
        self._trades: List[Tuple] = []
        self._lotes: Dict[str, List[List[float]]] = {}
        self._warmup_restante = 0

    # ------------------------------------------------------------------ #
    # Datos
    # ------------------------------------------------------------------ #
    def _rango_preparacion(self, resolucion: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        dias = PRECARGA_DIAS.get(resolucion, 800)
        return self.fecha_inicio - pd.Timedelta(days=dias), self.fecha_fin + pd.Timedelta(days=2)

    def datos(self, simbolo: lean.Symbol, resolucion: str) -> pd.DataFrame:
        """Velas del simbolo, precargadas bajo demanda (el data feed tiene cache)."""
        clave = (simbolo.value, resolucion)
        if clave not in self._datos:
            ini, fin = self._rango_preparacion(resolucion)
            self._datos[clave] = datafeed.bars(
                simbolo.value, resolucion, ini, fin, self.semilla
            )
        return self._datos[clave]

    def precargar(self, simbolos: List[str]) -> None:
        """Calienta la cache antes de la ejecucion (no es obligatorio, acelera)."""
        for t in simbolos:
            sym = lean.Symbol(t)
            for res in ("Daily", "Minute"):
                try:
                    self.datos(sym, res)
                except Exception:
                    pass

    def register_security(self, security: lean.Security, resolucion: str) -> None:
        self._resoluciones[security.symbol] = resolucion
        # La resolucion de ejecucion es la MAS FINA de todas las declaradas.
        if resolucion == "Minute":
            self._resolucion_ejecucion = "Minute"
        self.datos(security.symbol, resolucion)
        self.datos(security.symbol, "Daily")

    def build_indicator_series(self, nombre: str, simbolo: lean.Symbol,
                               periodo: int, resolucion: str) -> pd.Series:
        frame = self.datos(simbolo, resolucion)
        if frame.empty:
            return pd.Series(dtype=float)
        return lean._Rolling.compute(nombre, frame, periodo)

    def last_price(self, simbolo: lean.Symbol) -> float:
        sec = self.securities.get(simbolo)
        if sec is not None and sec.price > 0:
            return sec.price
        # Fuera de una vela en curso se usa el ultimo cierre conocido.
        for res in (self._resolucion_ejecucion, "Daily"):
            frame = self.datos(simbolo, res)
            if not frame.empty:
                return float(frame["close"].iloc[-1])
        return 0.0

    def history(self, simbolo: lean.Symbol, count: int, resolucion: str) -> pd.DataFrame:
        """Ultimas `count` velas hasta el instante actual."""
        frame = self.datos(simbolo, resolucion)
        if frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        pos = frame.index.searchsorted(self._current_time, side="right")
        inicio = max(0, pos - max(count, 0))
        # `.copy()` es imprescindible: las estrategias anaden columnas al resultado.
        return frame.iloc[inicio:pos].copy()

    # ------------------------------------------------------------------ #
    # Ordenes
    # ------------------------------------------------------------------ #
    def submit(self, simbolo: lean.Symbol, cantidad: float, direccion: str,
               tipo: str = "Market", limit_price: float = 0.0) -> lean.Order:
        self._orden_id += 1
        orden = lean.Order(self._orden_id, simbolo, abs(float(cantidad)), direccion, tipo, limit_price)
        orden.time = self._current_time
        self._pending.append(orden)
        return orden

    def _comision(self, sec: Optional[lean.Security], cantidad: float, valor: float) -> float:
        if sec is None:
            return 0.0
        modelo = sec.fee_model
        if hasattr(modelo, "order_fee"):
            fee = modelo.order_fee(None, cantidad, valor)
        elif hasattr(modelo, "get_order_fee"):
            fee = modelo.get_order_fee(None, cantidad, valor)
        else:
            return 0.0
        return float(fee.value.amount)

    def _procesar_ordenes(self) -> None:
        if not self._pending:
            return
        pendientes, self._pending = self._pending, []

        for orden in pendientes:
            simbolo = orden.symbol
            sec = self.securities.get(simbolo)
            precio = self.last_price(simbolo)
            if sec is None or precio <= 0:
                self.emit_log("WARNING",
                              f"Orden ignorada: sin datos de {simbolo.value}", self._current_time)
                continue

            # Deslizamiento
            deslizamiento = 0.0
            if sec.slippage_model is not None and hasattr(sec.slippage_model, "get_slippage_approximation"):
                deslizamiento = float(sec.slippage_model.get_slippage_approximation(simbolo) or 0.0)

            es_compra = orden.direction == lean.OrderDirection.BUY
            precio_ejecucion = precio * (1.0 + deslizamiento) if es_compra else precio * (1.0 - deslizamiento)

            cantidad = orden.quantity
            if cantidad <= 0:
                continue

            if es_compra:
                presupuesto = self.portfolio.cash * self.leverage
                comision = self._comision(sec, cantidad, cantidad * precio_ejecucion)
                coste = cantidad * precio_ejecucion + comision
                if coste > presupuesto:
                    # Ajuste proporcional iterativo hasta que quepa en caja.
                    for _ in range(8):
                        if cantidad <= 0 or coste <= presupuesto:
                            break
                        nueva = int(cantidad * presupuesto / coste)
                        if nueva >= cantidad:
                            nueva = cantidad - 1
                        cantidad = max(nueva, 0)
                        comision = self._comision(sec, cantidad, cantidad * precio_ejecucion)
                        coste = cantidad * precio_ejecucion + comision
                    if cantidad <= 0:
                        self.emit_log(
                            "WARNING",
                            f"Compra cancelada por falta de caja: {orden.quantity:.0f} {simbolo.value} "
                            f"(necesita ${coste:,.2f}, hay ${presupuesto:,.2f})",
                            self._current_time,
                        )
                        continue
                    self.emit_log(
                        "WARNING",
                        f"Compra parcial: {orden.quantity:.0f} -> {cantidad:.0f} {simbolo.value} "
                        f"por falta de caja",
                        self._current_time,
                    )
                self._aplicar_compra(simbolo, cantidad, precio_ejecucion, comision)
            else:
                tenencia = self.portfolio[simbolo]
                cantidad = min(cantidad, tenencia.quantity)
                if cantidad <= 1e-9:
                    continue
                comision = self._comision(sec, cantidad, cantidad * precio_ejecucion)
                self._aplicar_venta(simbolo, cantidad, precio_ejecucion, comision)

            # Evento de orden para `on_order_event`
            orden.status = lean.OrderStatus.FILLED
            orden.fill_quantity = cantidad
            orden.fill_price = precio_ejecucion
            orden.fee = lean.FeeOrder(comision)
            evento = lean.OrderEvent(
                simbolo, lean.OrderStatus.FILLED, cantidad, precio_ejecucion,
                orden.fee, orden, self._current_time,
            )
            if hasattr(self._algoritmo, "on_order_event"):
                self._algoritmo.on_order_event(evento)

    def _aplicar_compra(self, simbolo: lean.Symbol, cantidad: float, precio: float, comision: float) -> None:
        pos = self.portfolio[simbolo]
        pos.current_price = precio
        qty_previa = pos.quantity
        coste = cantidad * precio
        if qty_previa + cantidad > 0:
            pos.average_price = (pos.average_price * qty_previa + coste) / (qty_previa + cantidad)
        pos.quantity = qty_previa + cantidad
        self.portfolio.cash -= coste + comision
        self.portfolio.total_fees += comision
        self._registrar_fill(simbolo, cantidad, precio, comision, "BUY")
        self._registrar_compra_lote(simbolo, cantidad, precio)

    def _aplicar_venta(self, simbolo: lean.Symbol, cantidad: float, precio: float, comision: float) -> None:
        pos = self.portfolio[simbolo]
        pos.current_price = precio
        coste = cantidad * precio
        realizado = (precio - pos.average_price) * cantidad
        pos.quantity -= cantidad
        self.portfolio.cash += coste - comision
        self.portfolio.total_fees += comision
        self.portfolio.realized_net_profit += realizado - comision
        if abs(pos.quantity) < 1e-9:
            pos.quantity = 0.0
            pos.average_price = 0.0
        self._cerrar_lotes(simbolo, cantidad, precio)
        self._registrar_fill(simbolo, -cantidad, precio, comision, "SELL")

    def _registrar_fill(self, simbolo: lean.Symbol, cantidad: float, precio: float,
                        comision: float, lado: str) -> None:
        self._fills.append((self._current_time, simbolo.value, lado, cantidad, precio,
                            cantidad * precio, comision, self.portfolio.cash))

    # --- Lotes de tipo FIFO para reconstruir las operaciones cerradas ---- #
    def _cerrar_lotes(self, simbolo: lean.Symbol, cantidad: float, precio: float) -> None:
        cola = self._lotes.setdefault(simbolo.value, [])
        restante = cantidad
        while restante > 1e-9 and cola:
            fecha_abierta, cantidad_abierta, precio_abierto, comision_abierta = cola[0]
            usado = min(restante, cantidad_abierta)
            beneficio = (precio - precio_abierto) * usado
            self._trades.append((
                fecha_abierta, self._current_time, simbolo.value,
                usado, precio_abierto, precio, beneficio,
            ))
            restante -= usado
            if usado >= cantidad_abierta - 1e-9:
                cola.pop(0)
            else:
                cola[0] = [fecha_abierta, cantidad_abierta - usado, precio_abierto, comision_abierta]

    def _registrar_compra_lote(self, simbolo: lean.Symbol, cantidad: float, precio: float) -> None:
        self._lotes.setdefault(simbolo.value, []).append([self._current_time, cantidad, precio, 0.0])

    # ------------------------------------------------------------------ #
    # Diagnostico
    # ------------------------------------------------------------------ #
    def emit_log(self, nivel: str, mensaje: str, tiempo: Any) -> None:
        self._logs.append({
            "tiempo": tiempo,
            "nivel": nivel,
            "mensaje": mensaje,
            "orden": len(self._logs),
        })

    def plot(self, titulo: str, serie: str, valor: float) -> None:
        self._plots.setdefault(titulo, {}).setdefault(serie, []).append((self._current_time, valor))

    def _registrar(self, momento: Any, en_warmup: bool = False) -> None:
        """
        Anota una fila de la curva de capital.

        Se guarda el valor TOTAL de la cartera (caja + posiciones a precio de
        mercado) que es lo que QuantConnect llama *Portfolio Holdings Equity*.
        Para estrategias con apalancamiento el apalancamiento real puede diferir
        ligeramente del 1x si el broker permitiera mas, pero con apalancamiento 1.0
        la curva coincide exactamente con el equity negociable.
        """
        p = self.portfolio
        for sym, pos in p.securities.items():
            if sym in self.securities and pos.quantity:
                pos.current_price = self.last_price(sym)
        self._curva.append((
            momento,
            p.total_portfolio_value,
            p.cash,
            p.total_holdings_value,
            bool(p.invested),
            p.total_margin_used,
            p.total_fees,
            bool(en_warmup),
            p.realized_net_profit,
            # Coste de las posiciones abiertas (valor de mercado a precio medio).
            # Con este campo la ganancia no realizada es EXACTA:
            #   no_realizado = valor_mercado - coste
            # y no hay que aproximarla con `holdings - cash`, que no es la
            # ganancia sino simplemente el valor de la cartera.
            p.total_cost_basis,
        ))

    # El motor reemplaza `portfolio`, `securities` y `plot` por los suyos.
    @property
    def _algoritmo(self) -> Any:
        return getattr(self, "_instancia", None)


# --------------------------------------------------------------------------- #
# Carga del modulo de estrategia
# --------------------------------------------------------------------------- #
def _cargar_clase(fuente: str, ruta: Path) -> type:
    """
    Ejecuta el codigo de la estrategia y devuelve la clase QCAlgorithm.

    Se registra `AlgorithmImports` en `sys.modules` para que el
    `from AlgorithmImports import *` de la estrategia resuelva contra el runtime
    de este proyecto y el archivo se ejecute sin modificarlo.
    """
    lean.install_algorithm_imports()
    espacio: Dict[str, Any] = {
        "__name__": "__qc_estrategia__",
        "__file__": str(ruta),
        "__builtins__": __builtins__,
    }
    codigo = compile(fuente, str(ruta), "exec")
    exec(codigo, espacio)  # noqa: S102 - ejecutar la estrategia es el objetivo

    candidatas = [
        obj for obj in espacio.values()
        if isinstance(obj, type)
        and issubclass(obj, lean.QCAlgorithm)
        and obj is not lean.QCAlgorithm
    ]
    if not candidatas:
        raise RuntimeError(
            "El archivo no define ninguna clase que herede de QCAlgorithm"
        )
    # Si hay varias, se prefiere la que implementa `on_data`.
    for cls in candidatas:
        if "on_data" in cls.__dict__:
            return cls
    return candidatas[0]


def _instrumentar(algoritmo: lean.QCAlgorithm, motor: LeanEngine) -> lean.QCAlgorithm:
    """Inyecta en la instancia todos los servicios del runtime."""
    algoritmo._engine = motor
    algoritmo._securities = motor.securities
    algoritmo._symbols = motor._symbols
    algoritmo._indicators = motor._indicadores
    algoritmo._pending_orders = motor._pending
    algoritmo._plot_data = motor._plots
    algoritmo._logs = motor._logs
    algoritmo._current_time = motor._current_time
    algoritmo._cash = motor._cash
    algoritmo._start_date = motor._start_date
    algoritmo._end_date = motor._end_date
    algoritmo._benchmark = motor._benchmark
    algoritmo._warmup_count = motor._warmup_count
    algoritmo._warmup_resolution = motor._warmup_res
    algoritmo._leverage = motor.leverage

    # El `portfolio` y el `plot` del algoritmo pasan a ser los del motor, para que
    # la cuenta unica y los registros sean consistentes.
    algoritmo.portfolio = motor.portfolio
    algoritmo.securities = motor.securities
    algoritmo.schedule = motor.schedule
    algoritmo.log = lambda msg: motor.emit_log("INFO", str(msg), motor._current_time)
    algoritmo.debug = lambda msg: motor.emit_log("DEBUG", str(msg), motor._current_time)
    algoritmo.error = lambda msg: motor.emit_log("ERROR", str(msg), motor._current_time)
    algoritmo.warn = lambda msg: motor.emit_log("WARNING", str(msg), motor._current_time)
    algoritmo.plot = lambda titulo, serie, valor=None, *a, **k: motor.plot(titulo, serie, valor)
    return algoritmo


def _tiempo_resuelto(resolucion_indicador: str, actual: pd.Timestamp,
                     ejecucion: str) -> pd.Timestamp:
    """
    Instante contra el que se resuelve un indicador.

    Un indicador DIARIO consultado dentro de una vela de MINUTO debe usar la
    ultima vela diaria COMPLETA: usar la del dia en curso introduciria el precio
    intradia en una media que en LEAN todavia no esta cerrada.
    """
    if resolucion_indicador == "Daily" and ejecucion == "Minute":
        return actual.normalize() - pd.Timedelta(days=1)
    return actual


# --------------------------------------------------------------------------- #
# Bucle principal
# --------------------------------------------------------------------------- #
def ejecutar_backtest(
    ruta: str | Path,
    parametros: Optional[Dict[str, Any]] = None,
    capital: Optional[float] = None,
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
    apalancamiento: float = 1.0,
    semilla: int = datafeed.DEFAULT_SEED,
    info: Optional[EstrategiaInfo] = None,
    simbolo: Optional[str] = None,
    benchmark: Optional[str] = None,
    comision: Optional[float] = None,
) -> ResultadoBacktest:
    """Ejecuta una estrategia y devuelve curvas, operaciones, metricas y logs.

    Es el punto de entrada UNICO del proyecto: despacha al motor Lean si el
    archivo expone una clase `QCAlgorithm` y al ejecutor vectorial si expone
    `run_strategy(data, ...)`. Los dos caminos devuelven el mismo
    `ResultadoBacktest`, con las mismas metricas calculadas por `analytics`.
    """
    inicio = _time.perf_counter()
    ruta = Path(ruta)
    info = info or analizar_archivo(ruta)

    if not info.valido:
        res = ResultadoBacktest(info=info, exito=False, error=info.error or "Estrategia no valida")
        res.duracion_s = _time.perf_counter() - inicio
        return res

    try:
        fuente = ruta.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        res = ResultadoBacktest(info=info, exito=False, error=f"No se pudo leer el archivo: {exc}")
        res.duracion_s = _time.perf_counter() - inicio
        return res

    # --- dialecto vectorial: misma contabilidad y mismas metricas ------- #
    if info.dialecto == DIALECTO_VECTOR:
        from qc_lab.core.vector_runner import COMISION_POR_DEFECTO, ejecutar_backtest_vector

        return ejecutar_backtest_vector(
            fuente,
            info=info,
            parametros=parametros,
            capital=capital,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            simbolo=simbolo,
            benchmark=benchmark,
            comision_rate=COMISION_POR_DEFECTO if comision is None else comision,
            semilla=semilla,
            etiqueta=info.nombre_archivo,
        )

    # --- 1) Inyeccion de parametros --------------------------------- #
    fuente_modificada, aplicados = aplicar_parametros(fuente, parametros or {})

    motor = LeanEngine(
        info=info,
        parameters=parametros or {},
        capital=capital,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        apalancamiento=apalancamiento,
        semilla=semilla,
    )

    try:
        clase = _cargar_clase(fuente_modificada, ruta)
    except Exception as exc:
        res = ResultadoBacktest(
            info=info, exito=False,
            error=f"Error al importar la estrategia: {exc}",
            traza=traceback.format_exc()[-3000:],
        )
        res.duracion_s = _time.perf_counter() - inicio
        res.logs = motor._logs
        return res

    # --- 2) Calentamiento de la cache de datos ---------------------- #
    try:
        motor.precargar(list(info.simbolos) + [info.benchmark_efectivo])
    except Exception:
        pass

    # --- 3) initialize() --------------------------------------------- #
    try:
        instancia = clase()
        _instrumentar(instancia, motor)
        motor._instancia = instancia
        instancia.initialize()
    except Exception as exc:
        res = ResultadoBacktest(
            info=info, exito=False,
            error=f"Error en initialize(): {exc}",
            traza=traceback.format_exc()[-3000:],
        )
        res.logs = motor._logs
        res.duracion_s = _time.perf_counter() - inicio
        return res

    # Tras `initialize()` la estrategia puede haber cambiado fechas, capital y
    # warm-up con `set_start_date`, `set_end_date`, `set_cash` o `set_warm_up`.
    # El motor se sincroniza con esos valores definitivos: sin esta copia se
    # construiria la linea de tiempo con las fechas PREVIAS a `initialize()` y las
    # capacidades/numeric del usuario se ignorarian.
    motor._start_date = pd.Timestamp(instancia._start_date)
    motor._end_date = pd.Timestamp(instancia._end_date)
    motor._cash = float(instancia._cash)

    # El warm-up y el rango definitivos solo se conocen tras `initialize()`.
    inicio_real = pd.Timestamp(motor._start_date)
    fin_real = pd.Timestamp(motor._end_date)
    if fin_real <= inicio_real:
        fin_real = inicio_real + pd.DateOffset(years=1)
    motor._warmup_count = instancia._warmup_count
    motor._warmup_res = instancia._warmup_resolution
    ejecucion = motor._resolucion_ejecucion

    # --- 4) Construir la linea de tiempo ----------------------------- #
    try:
        linea = _construir_timeline(motor, ejecucion, inicio_real, fin_real)
    except Exception as exc:
        res = ResultadoBacktest(info=info, exito=False,
                                error=f"No hay datos para el periodo solicitado: {exc}",
                                logs=motor._logs)
        res.duracion_s = _time.perf_counter() - inicio
        return res

    if linea.empty:
        res = ResultadoBacktest(info=info, exito=False,
                                error="El periodo solicitado no contiene velas.",
                                logs=motor._logs)
        res.duracion_s = _time.perf_counter() - inicio
        return res

    # --- 5) Fin del warm-up ------------------------------------------- #
    # El warm-up se cuenta en SU PROPIA resolucion, no en la de ejecucion. Una
    # estrategia puede ejecutar en minuto y pedir 200 velas DIARIAS de warm-up:
    # si se compararan contra las velas de minuto, el contadorjamas llegaria a
    # cero, `is_warming_up` se quedaria en True y `on_data` no se llamaria nunca.
    fin_warmup = _calcular_fin_warmup(motor, inicio_real, ejecucion)
    motor._warmup_restante = max(motor._warmup_count, 0)
    instancia.is_warmingup_estado = False

    # --- 6) Mapas rapidos de acceso a los datos ------------------------ #
    ctx = _construir_contexto(motor, linea, ejecucion, fin_warmup)

    # --- 7) Recorrido vela a vela -------------------------------------- #
    instancia.time = linea[0]
    instancia.is_warming_up = bool(linea[0] < fin_warmup)

    n_barras = len(linea)
    for i in range(n_barras):
        actual = pd.Timestamp(linea[i])
        motor._current_time = actual
        instancia._current_time = actual
        instancia.time = actual

        # --- velas disponibles en este instante ---
        slice_barras: Dict[lean.Symbol, lean.TradeBar] = {}
        for s in ctx["simbolos"]:
            if ctx["hay_dato"][s][i]:
                p = ctx["pos"][s][i]
                a = ctx["arrays"][s]
                barra = lean.TradeBar(a["open"][p], a["high"][p], a["low"][p],
                                      a["close"][p], a["volume"][p], actual)
                slice_barras[s] = barra
                sec = motor.securities.get(s)
                if sec is not None:
                    sec.update(barra)

        motor.current_slice = lean.Slice(actual, slice_barras, motor._symbols)
        instancia.is_warming_up = bool(ctx["en_warmup"][i])

        # --- indicadores ---
        for ind, valores, posiciones in ctx["indicadores"]:
            ind.actualizar_rapido(valores, int(posiciones[i]), actual)

        try:
            _disparar_calendario(motor, instancia, actual, ejecucion)

            if not instancia.is_warming_up and hasattr(instancia, "on_data"):
                instancia.on_data(motor.current_slice)
        except Exception as exc:
            motor.emit_log("ERROR", f"Excepcion en la vela {actual}: {exc}", actual)
            motor.emit_log("DEBUG", traceback.format_exc()[-1500:], actual)

        # --- ejecucion de ordenes ---
        try:
            motor._procesar_ordenes()
        except Exception as exc:
            motor.emit_log("ERROR", f"Error ejecutando ordenes: {exc}", actual)

        # --- registro de la curva de capital ---
        motor._registrar(actual, bool(ctx["en_warmup"][i]))

    # --- 8) cierre ------------------------------------------------------ #
    try:
        if hasattr(instancia, "on_end_of_algorithm"):
            instancia.on_end_of_algorithm()
    except Exception as exc:
        motor.emit_log("WARNING", f"Error en on_end_of_algorithm(): {exc}", linea[-1])

    return _empaquetar(motor, instancia, info, linea, ejecucion,
                       _time.perf_counter() - inicio, aplicados, n_barras)


def _calcular_fin_warmup(motor: LeanEngine, inicio: pd.Timestamp, ejecucion: str) -> pd.Timestamp:
    """
    Instante en el que termina el warm-up, contado en su propia resolucion.

    Devuelve el timestamp de la vela `warmup_count` posterior a la fecha de inicio.
    """
    n = int(motor._warmup_count or 0)
    if n <= 0:
        return inicio
    res = motor._warmup_res
    referencia = next(iter(motor._symbols.values()), None)
    if referencia is not None:
        frame = motor.datos(referencia, res)
        idx = frame.index[frame.index >= inicio]
    else:
        idx = datafeed.trading_days(inicio.strftime("%Y-%m-%d"),
                                    inicio.strftime("%Y-%m-%d")).append(
            pd.DatetimeIndex([inicio]))
    if len(idx) >= n:
        return pd.Timestamp(idx[n - 1])
    return pd.Timestamp(idx[-1]) if len(idx) else inicio


def _construir_contexto(motor: LeanEngine, linea: pd.DatetimeIndex,
                        ejecucion: str, fin_warmup: pd.Timestamp) -> Dict[str, Any]:
    """
    Precalcula todo lo que el bucle necesita, en vectores numpy.

    Razon: el bucle puede llegar a cientos de miles de velas. Acceder con
    `DataFrame.iloc[fila]` dentro del bucle cuesta ~100 microsegundos por fila,
    mientras que el acceso a un array numpy cuesta ~50 nanosegundos. Con los
    vectores precalculados el recorrido baja de ~100 s a pocos segundos.
    """
    linea_val = linea.values

    arrays: Dict[lean.Symbol, Dict[str, np.ndarray]] = {}
    posicion: Dict[lean.Symbol, np.ndarray] = {}
    hay_dato: Dict[lean.Symbol, np.ndarray] = {}

    for s in motor._symbols.values():
        frame = motor.datos(s, ejecucion)
        if frame.empty:
            arrays[s] = {}
            posicion[s] = np.zeros(len(linea), dtype=np.intp)
            hay_dato[s] = np.zeros(len(linea), dtype=bool)
            continue
        arrays[s] = {c: frame[c].to_numpy(dtype=float)
                     for c in ("open", "high", "low", "close", "volume")}
        idx_val = frame.index.values
        pos = frame.index.searchsorted(linea_val)
        seguro = np.minimum(pos, len(idx_val) - 1)
        posicion[s] = pos
        hay_dato[s] = (pos < len(idx_val)) & (idx_val[seguro] == linea_val)

    # Posicion de cada indicador en cada vela, tambien precalculada.
    indicadores = []
    for ind in motor._indicadores.values():
        serie = ind.serie
        if serie is None or len(serie) == 0:
            valores = np.array([], dtype=float)
            posiciones = np.full(len(linea), -1, dtype=np.intp)
        else:
            valores = serie.to_numpy(dtype=float)
            idx_ref = serie.index
            if ind.resolution == "Daily" and ejecucion == "Minute":
                # Ultima vela diaria COMPLETA (excluye el dia en curso).
                corte = linea.normalize() - pd.Timedelta(days=1)
                posiciones = idx_ref.searchsorted(corte.values, side="right") - 1
            else:
                posiciones = idx_ref.searchsorted(linea_val, side="right") - 1
        indicadores.append((ind, valores, posiciones))

    en_warmup = linea < fin_warmup

    return {
        "simbolos": list(motor._symbols.values()),
        "arrays": arrays,
        "pos": posicion,
        "hay_dato": hay_dato,
        "indicadores": indicadores,
        "en_warmup": en_warmup,
    }


def _construir_timeline(motor: LeanEngine, ejecucion: str,
                        inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DatetimeIndex:
    """Union de las marcas de tiempo de todos los simbolos en el periodo."""
    indices = []
    for s in motor._symbols.values():
        frame = motor.datos(s, ejecucion)
        if frame.empty:
            continue
        indices.append(frame.index[(frame.index >= inicio) & (frame.index <= fin)])
    if not indices:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set().union(*[set(i) for i in indices])))


def _disparar_calendario(motor: LeanEngine, instancia: lean.QCAlgorithm,
                         actual: pd.Timestamp, ejecucion: str) -> None:
    """Ejecuta los handlers programados que tocan esta vela."""
    items = motor.schedule.items
    if not items:
        return
    if ejecucion == "Daily":
        # En velas diarias solo hay un instante por dia: se disparan todos.
        for item in items:
            item.handler()
        return
    minuto_sesion = (actual.hour * 60 + actual.minute) - lean.MARKET_OPEN_MINUTE
    for item in items:
        if minuto_sesion == item.time_rule.offset_minutes:
            item.handler()


def _empaquetar(motor: LeanEngine, instancia: lean.QCAlgorithm, info: EstrategiaInfo,
                linea: pd.DatetimeIndex, ejecucion: str, duracion: float,
                aplicados: List[str], barras: int) -> ResultadoBacktest:
    """Construye el resultado final a partir de los registros del motor."""
    from qc_lab.core.analytics import calcular_estadisticas

    curva = pd.DataFrame(
        motor._curva,
        columns=["time", "equity", "cash", "holdings", "invested",
                 "margin", "fees", "warmup", "realizado", "coste"],
    )
    if not curva.empty:
        curva = curva.set_index("time")

    fills = pd.DataFrame(
        motor._fills,
        columns=["time", "symbol", "side", "qty", "price", "value", "fee", "cash"],
    )
    trades = pd.DataFrame(
        motor._trades,
        columns=["open_time", "close_time", "symbol", "qty", "open_price",
                 "close_price", "pnl"],
    )

    # --- curvas de referencia ---------------------------------------- #
    benchmark = pd.Series(dtype=float)
    precio_activo = pd.Series(dtype=float)
    if not curva.empty and ejecucion == "Daily":
        ventana = curva.loc[(curva.index >= linea[0]) & (curva.index <= linea[-1])]
        cols = [c for c in ("equity", "cash", "holdings", "invested", "margin", "fees",
                            "warmup", "coste", "realizado")
                if c in ventana.columns]
        curva = ventana[cols]

    bench_sym = motor._benchmark or lean.Symbol("SPY")
    frame_b = motor.datos(bench_sym, "Daily")
    if not frame_b.empty:
        seg = frame_b[(frame_b.index >= linea[0]) & (frame_b.index <= linea[-1])]
        benchmark = seg["close"].astype(float)

    riesgo = info.tickers_riesgo
    if riesgo:
        frame_a = motor.datos(lean.Symbol(riesgo[0]), "Daily")
        if not frame_a.empty:
            seg_a = frame_a[(frame_a.index >= linea[0]) & (frame_a.index <= linea[-1])]
            precio_activo = seg_a["close"].astype(float)

    estadisticas = calcular_estadisticas(
        curva=curva, fills=fills, trades=trades, info=info,
        benchmark=benchmark, capital_inicial=motor._cash,
        ejecucion=ejecucion, warmup=len(linea),
    )

    diagnostico = pd.DataFrame()
    plots = motor._plots

    return ResultadoBacktest(
        info=info,
        exito=True,
        curva=curva,
        operaciones=trades,
        ejecuciones=fills,
        estadisticas=estadisticas,
        logs=motor._logs,
        plots=plots,
        benchmark=benchmark,
        precio_activo=precio_activo,
        diagnostico=diagnostico,
        duracion_s=duracion,
        barras_evaluadas=barras,
    )