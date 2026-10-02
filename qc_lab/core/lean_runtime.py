"""
Runtime minimo compatible con QuantConnect LEAN.

Este modulo implementa la parte de la API de LEAN que realmente usan las
estrategias del proyecto, de modo que los archivos `.py` se ejecutan SIN
modificarse: se importa `AlgorithmImports`, se instancia la clase que herede de
`QCAlgorithm` y el motor la recorre vela a vela.

Cobertura de la API:

  * Ciclo de vida  : initialize / on_data / on_order_event / on_end_of_algorithm
  * Datos          : add_equity, securities, history, Slice/TradeBar, Resolution
  * Indicadores    : sma, ema, atr, rsi, macd... con `.is_ready` y `.current.value`
  * Cartera        : portfolio.cash / invested / holdings / fees / margen
  * Ordenes        : market_order, limit_order, liquidate, set_holdings
  * Costes         : InteractiveBrokersFeeModel, ConstantSlippageModel...
  * Planificador   : schedule.on + date_rules + time_rules
  * Parametros     : get_parameter (para los barrido A/B desde el dashboard)
  * Diagnostico    : log, plot, Slice

Las decisiones que difieren de LEAN y conviene conocer:
  * Los datos provienen de `datafeed` (velas sinteticas deterministas).
  * Los indicadores de resolucion superior usan la ultima vela DIARIA COMPLETA,
    de modo que una consulta intradia no ve informacion del dia en curso.
  * Las compras nunca exceden el poder de compra (apalancamiento 1.0 por defecto).
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

MINUTES_PER_SESSION = 390
MARKET_OPEN_MINUTE = 570          # 09:30
MARKET_CLOSE_MINUTE = 960         # 16:00


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
class Resolution:
    """Resoluciones soportadas. Equivalente al enum de LEAN."""

    TICK = "Tick"
    SECOND = "Second"
    MINUTE = "Minute"
    HOURLY = "Hourly"
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"

    @staticmethod
    def parse(value: Any) -> str:
        """Normaliza 'daily', Resolution.DAILY o 'D' al nombre canonico."""
        if isinstance(value, Resolution):
            return value
        txt = str(getattr(value, "name", value)).strip().lower()
        alias = {
            "d": "Daily", "1d": "Daily", "day": "Daily", "daily": "Daily",
            "m": "Minute", "1m": "Minute", "min": "Minute", "minute": "Minute",
            "h": "Hourly", "1h": "Hourly", "hour": "Hourly", "hourly": "Hourly",
            "w": "Weekly", "week": "Weekly", "weekly": "Weekly",
        }
        return alias.get(txt, txt.capitalize())

    @staticmethod
    def minutes_of(res: str) -> int:
        return {"Daily": 1440, "Hourly": 60}.get(Resolution.parse(res), 1)


# --------------------------------------------------------------------------- #
# Symbol
# --------------------------------------------------------------------------- #
class Symbol:
    """Identificador de un valor. Comparable por ticker (case-insensitive)."""

    __slots__ = ("value", "_market", "_symbol")

    def __init__(self, ticker: str, market: str = "usa"):
        self.value = str(ticker).upper()
        self._market = market
        self._symbol = str(ticker).lower()

    # --- protocolo de igualdad / hash ---
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Symbol):
            return self._symbol == other._symbol and self._market == other._market
        if isinstance(other, str):
            return self._symbol == other.lower()
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._symbol, self._market))

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    @staticmethod
    def create(ticker: str, market: str = "usa") -> "Symbol":
        return Symbol(ticker, market)


def _as_symbol(value: Any) -> Symbol:
    """Coerce flexible: acepta Symbol o str (muchas estrategias indexan por ambos)."""
    return value if isinstance(value, Symbol) else Symbol(str(value))


# --------------------------------------------------------------------------- #
# Velas
# --------------------------------------------------------------------------- #
class TradeBar:
    """Vela OHLCV inmutable."""

    __slots__ = ("open", "high", "low", "close", "volume", "time")

    def __init__(self, open_: float, high: float, low: float, close: float,
                 volume: float = 0.0, time: Any = None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.time = time

    @property
    def price(self) -> float:
        return self.close

    @property
    def value(self) -> float:
        return self.close

    def __repr__(self) -> str:
        return f"TradeBar(O:{self.open:.2f} H:{self.high:.2f} L:{self.low:.2f} C:{self.close:.2f})"


class Slice:
    """Coleccion de velas disponibles en un instante concreto (mapa por Symbol)."""

    __slots__ = ("time", "_bars", "_symbols")

    def __init__(self, time: Any, bars: Dict[Symbol, TradeBar], symbols: Dict[str, Symbol]):
        self.time = time
        self._bars = bars
        self._symbols = symbols

    def contains_key(self, key: Any) -> bool:
        sym = self._symbols.get(str(key).lower()) if isinstance(key, str) else key
        return sym in self._bars

    def __getitem__(self, key: Any) -> TradeBar:
        sym = self._symbols.get(str(key).lower()) if isinstance(key, str) else key
        return self._bars[sym]

    def __contains__(self, key: Any) -> bool:
        return self.contains_key(key)

    def __iter__(self):
        return iter(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def keys(self):
        return self._bars.keys()

    @property
    def values(self):
        return self._bars.values()

    @property
    def items(self):
        return self._bars.items()

    def get(self, key: Any, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default

    def count(self, key: Any) -> int:
        return 1 if self.contains_key(key) else 0


# --------------------------------------------------------------------------- #
# Seguridad y cartera
# --------------------------------------------------------------------------- #
class Security:
    """Valor emitido por `add_equity`."""

    __slots__ = ("symbol", "fee_model", "slippage_model", "fill_model",
                 "leverage_model", "price", "bar", "_engine", "name")

    def __init__(self, symbol: Symbol, engine: "LeanEngine"):
        self.symbol = symbol
        self.name = symbol.value
        self.fee_model = ZeroFeeModel()
        self.slippage_model = ZeroSlippageModel()
        self.fill_model = None
        self.leverage_model = None
        self.price = 0.0
        self.bar: Optional[TradeBar] = None
        self._engine = engine

    # --- configuracion (cadena fluida, igual que LEAN) ---
    def set_fee_model(self, model: Any) -> "Security":
        self.fee_model = model
        return self

    def set_slippage_model(self, model: Any) -> "Security":
        self.slippage_model = model
        return self

    def set_fill_model(self, model: Any) -> "Security":
        self.fill_model = model
        return self

    def set_leverage(self, leverage: float) -> "Security":
        self._engine.leverage = leverage
        return self

    def set_name(self, name: str) -> "Security":
        self.name = name
        return self

    def set_data_normalization_mode(self, *a, **k) -> "Security":
        return self

    def set_margin_model(self, model: Any) -> "Security":
        return self

    def set_exchange(self, *a, **k) -> "Security":
        return self

    def update(self, bar: TradeBar) -> None:
        self.bar = bar
        self.price = bar.close


class Holding:
    """Posicion sobre un valor."""

    __slots__ = ("symbol", "quantity", "average_price", "current_price")

    def __init__(self, symbol: Symbol):
        self.symbol = symbol
        self.quantity = 0.0
        self.average_price = 0.0
        self.current_price = 0.0

    @property
    def holdings_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def market_value(self) -> float:
        return self.holdings_value

    @property
    def price(self) -> float:
        return self.current_price

    @property
    def value(self) -> float:
        return self.holdings_value

    @property
    def invested(self) -> bool:
        return abs(self.quantity) > 1e-12

    @property
    def quantity_long(self) -> float:
        return max(self.quantity, 0.0)

    @property
    def quantity_short(self) -> float:
        return min(self.quantity, 0.0)

    @property
    def avg_price(self) -> float:
        return self.average_price

    def __repr__(self) -> str:
        return f"{self.symbol.value}: {self.quantity:,.4f} @ {self.average_price:,.4f}"


class Portfolio:
    """Cartera simulada: caja, posiciones, comisiones y margen."""

    def __init__(self):
        self.cash: float = 0.0
        self.total_fees: float = 0.0
        self.invested_unrealized: float = 0.0
        self.invested_realized: float = 0.0
        self.realized_net_profit: float = 0.0
        # `unrealized_net_profit` es una propiedad calculada a partir de las
        # posiciones; el valor manual solo se guarda por si una estrategia lo
        # asigna explicitamente.
        self._unrealized_manual: float = 0.0
        self._holdings: Dict[Symbol, Holding] = {}

    # --- acceso ---
    def __getitem__(self, key: Any) -> Holding:
        sym = _as_symbol(key)
        if sym not in self._holdings:
            self._holdings[sym] = Holding(sym)
        return self._holdings[sym]

    def __contains__(self, key: Any) -> bool:
        sym = _as_symbol(key)
        h = self._holdings.get(sym)
        return bool(h and h.invested)

    def __iter__(self):
        return iter(self._holdings)

    @property
    def securities(self) -> Dict[Symbol, Holding]:
        return self._holdings

    @property
    def invested(self) -> bool:
        return any(h.invested for h in self._holdings.values())

    @property
    def total_holdings_value(self) -> float:
        return sum(h.holdings_value for h in self._holdings.values())

    @property
    def holdings_value(self) -> float:
        return self.total_holdings_value

    @property
    def total_portfolio_value(self) -> float:
        return self.cash + self.total_holdings_value

    @property
    def total_cost_basis(self) -> float:
        """
        Coste de las posiciones abiertas: valor de mercado a precio medio de compra.

        Permite obtener la ganancia NO REALIZADA exacta
        (`total_holdings_value - total_cost_basis`) sin aproximarla.
        """
        return sum(h.average_price * abs(h.quantity) for h in self._holdings.values())

    @property
    def unrealized_net_profit(self) -> float:
        """Ganancia no realizada de las posiciones abiertas."""
        return self.total_holdings_value - self.total_cost_basis

    @unrealized_net_profit.setter
    def unrealized_net_profit(self, value: float) -> None:
        # Atributo calculado: se acepta la asignacion (las estrategias lo tocan)
        # pero el valor real se deriva de las posiciones.
        self._unrealized_manual = float(value)

    @property
    def portfolio_value(self) -> float:
        return self.total_portfolio_value

    @property
    def total_margin_used(self) -> float:
        """Margen usado = exposicionnocional menos caja (nunca negativo)."""
        return max(self.total_holdings_value - self.cash, 0.0)

    @property
    def margin_used(self) -> float:
        return self.total_margin_used

    @property
    def total_portfolio_margin_used(self) -> float:
        return self.total_margin_used

    @property
    def buying_power(self) -> float:
        return self.cash

    @property
    def avg_price(self) -> float:
        invested = [h for h in self._holdings.values() if h.invested]
        if not invested:
            return 0.0
        total = sum(h.holdings_value for h in invested)
        return sum(h.average_price * h.quantity for h in invested) / max(sum(h.quantity for h in invested), 1e-12) if total else 0.0

    def get(self, key: Any, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        return self._holdings.items()

    def keys(self):
        return self._holdings.keys()

    def values(self):
        return self._holdings.values()

    def set_cash(self, amount: float) -> None:
        self.cash = float(amount)


# --------------------------------------------------------------------------- #
# Modelos de costes
# --------------------------------------------------------------------------- #
class FeeOrder:
    """Importe de una comision."""

    __slots__ = ("value",)

    def __init__(self, amount: float):
        self.value = types.SimpleNamespace(amount=float(amount))

    def __float__(self) -> float:
        return self.value.amount

    def __repr__(self) -> str:
        return f"FeeOrder(${self.value.amount:,.4f})"


class InteractiveBrokersFeeModel:
    """
    Comisiones Interactive Brokers (EE.UU., acciones):

        0,005 $ por accion, minimo 1,00 $ y maximo 0,5 % del valor negociado.

    El minimo de 1 $ es exactamente el motivo por el que las estrategias de
    cuenta pequena ($500) necesitan un tamano minimo de orden.
    """

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0, max_pct: float = 0.005):
        self.per_share = per_share
        self.minimum = minimum
        self.max_pct = max_pct

    def order_fee(self, order: Any = None, qty: float = 0.0, value: float = 0.0) -> FeeOrder:
        fee = max(self.per_share * abs(qty), self.minimum)
        cap = self.max_pct * abs(value)
        return FeeOrder(min(fee, cap) if cap > 0 else fee)

    # Alias habituales de LEAN
    def get_order_fee(self, order: Any = None, qty: float = 0.0, value: float = 0.0) -> FeeOrder:
        return self.order_fee(order, qty, value)


class ConstantFeeModel:
    def __init__(self, amount: float = 0.0):
        self.amount = float(amount)

    def order_fee(self, order: Any = None, qty: float = 0.0, value: float = 0.0) -> FeeOrder:
        return FeeOrder(self.amount)

    get_order_fee = order_fee


class PercentFeeModel:
    def __init__(self, percent: float = 0.0):
        self.percent = float(percent)

    def order_fee(self, order: Any = None, qty: float = 0.0, value: float = 0.0) -> FeeOrder:
        return FeeOrder(abs(value) * self.percent)

    get_order_fee = order_fee


class ZeroFeeModel(ConstantFeeModel):
    def __init__(self):
        super().__init__(0.0)


FeeModel = InteractiveBrokersFeeModel
ConstantFee = ConstantFeeModel
PercentFee = PercentFeeModel


class ConstantSlippageModel:
    """Deslizamiento fijo en puntos basicos del precio."""

    def __init__(self, bps: float = 0.0):
        self.bps = float(bps)

    def get_slippage_approximation(self, asset: Any = None) -> float:
        return self.bps / 10000.0

    # Alias
    def slippage(self, asset: Any = None) -> float:
        return self.get_slippage_approximation(asset)


class ZeroSlippageModel(ConstantSlippageModel):
    def __init__(self):
        super().__init__(0.0)


class MarketFillModel:
    def __init__(self, *a, **k):
        pass


SlippageModel = ConstantSlippageModel


# --------------------------------------------------------------------------- #
# Ordenes
# --------------------------------------------------------------------------- #
class OrderStatus:
    NEW = "New"
    SUBMITTED = "Submitted"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    INVALID = "Invalid"
    NONE = "None"


class OrderDirection:
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class Order:
    __slots__ = ("id", "symbol", "quantity", "type", "direction", "status",
                 "fill_quantity", "fill_price", "fee", "time", "tag", "limit_price")

    def __init__(self, order_id: int, symbol: Symbol, quantity: float,
                 direction: str, order_type: str = "Market", limit_price: float = 0.0):
        self.id = order_id
        self.symbol = symbol
        self.quantity = float(quantity)
        self.type = order_type
        self.direction = direction
        self.status = OrderStatus.NEW
        self.fill_quantity = 0.0
        self.fill_price = 0.0
        self.fee = FeeOrder(0.0)
        self.time: Any = None
        self.tag = ""
        self.limit_price = limit_price


class OrderEvent:
    """Evento de ejecucion que entrega el motor a `on_order_event`."""

    __slots__ = ("symbol", "status", "fill_quantity", "fill_price", "order_fee",
                 "order", "time", "direction")

    def __init__(self, symbol: Symbol, status: str, fill_quantity: float,
                 fill_price: float, order_fee: Any, order: Order, time: Any):
        self.symbol = symbol
        self.status = status
        self.fill_quantity = float(fill_quantity)
        self.fill_price = float(fill_price)
        self.order_fee = order_fee
        self.order = order
        self.time = time
        self.direction = order.direction


# --------------------------------------------------------------------------- #
# Indicadores
# --------------------------------------------------------------------------- #
class IndicatorPoint:
    __slots__ = ("time", "value")

    def __init__(self, time: Any, value: float):
        self.time = time
        self.value = float(value)

    def __float__(self) -> float:
        return self.value


class Indicator:
    """
    Indicador tecnico con la misma semantica de LEAN:
    `.is_ready` pasa a True cuando hay historia suficiente y `.current.value`
    devuelve el valor vigente (NaN si aun no esta listo).
    """

    def __init__(self, name: str, symbol: Symbol, period: int, resolution: str, engine: "LeanEngine"):
        self.name = name
        self.symbol = symbol
        self.period = int(period)
        self.resolution = resolution
        self._engine = engine
        self.serie = engine.build_indicator_series(name, symbol, period, resolution)
        self.current = IndicatorPoint(None, float("nan"))
        self._ready = False

    def update_state(self, time: Any, resolved_time: Any) -> None:
        """Ruta lenta (por indice temporal). Se mantiene por compatibilidad."""
        valor = float("nan")
        idx = self.serie.index
        if len(idx) and resolved_time is not None:
            pos = idx.searchsorted(resolved_time, side="right") - 1
            if pos >= 0:
                v = self.serie.iloc[pos]
                if pd.notna(v):
                    valor = float(v)
        self.current = IndicatorPoint(resolved_time, valor)
        self._ready = bool(pd.notna(valor))

    def actualizar_rapido(self, valores: np.ndarray, posicion: int, time: Any) -> None:
        """
        Ruta rapida del bucle de backtest.

        El motor precalcula la posicion de la vela dentro de la serie del
        indicador, de modo que aqui solo hay un indexado sobre un array numpy.
        """
        if posicion < 0 or posicion >= len(valores):
            self.current = IndicatorPoint(time, float("nan"))
            self._ready = False
            return
        valor = float(valores[posicion])
        self.current = IndicatorPoint(time, valor)
        self._ready = bool(np.isfinite(valor))

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def warmup(self) -> int:
        return self.period

    def __repr__(self) -> str:
        return f"{self.name}({self.symbol.value}, {self.period}, {self.resolution})"


class _Rolling:
    """Calcula la serie completa de un indicador una sola vez (vectorizado)."""

    @staticmethod
    def compute(name: str, frame: pd.DataFrame, period: int) -> pd.Series:
        period = max(int(period), 1)
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        n = name.lower()
        if n in ("sma", "simplemovingaverage", "sma.simplemovingaverage"):
            return close.rolling(period, min_periods=period).mean()
        if n in ("ema", "exponentialmovingaverage"):
            return close.ewm(span=period, adjust=False, min_periods=period).mean()
        if n in ("wma", "weightedmovingaverage"):
            pesos = np.arange(1, period + 1, dtype=float)
            return close.rolling(period, min_periods=period).apply(
                lambda w: float(np.dot(w, pesos) / pesos.sum()), raw=True)
        if n in ("rsi", "relativestrengthindex"):
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            media_g = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            media_l = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
            rs = media_g / media_l.replace(0.0, np.nan)
            return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0).where(media_l.notna() | media_g.notna())
        # OJO: `n` ya viene en minusculas, asi que el alias debe ir tambien.
        # Con "averageTrueRange" en mayusculas esta rama era inalcanzable y el
        # ATR caia en la SMA por defecto, silenciosamente.
        if n in ("atr", "averagetruerange", "averageTrueRange".lower()):
            pc = close.shift(1)
            tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        if n in ("stddev", "standarddeviation"):
            return close.rolling(period, min_periods=period).std(ddof=1)
        if n in ("macd",):
            # periods 12/26/9 por defecto
            p_fast, p_slow, p_signal = (period, period * 2 + 2, 9)
            macd = close.ewm(span=p_fast, adjust=False).mean() - close.ewm(span=p_slow, adjust=False).mean()
            return macd.ewm(span=p_signal, adjust=False).mean()
        # Por defecto, media movil simple (identico a SMA)
        return close.rolling(period, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Reglas de planificacion
# --------------------------------------------------------------------------- #
class DateRules:
    @staticmethod
    def every_day(symbol: Any = None) -> "_DateRule":
        return _DateRule(symbol)

    @staticmethod
    def every_week(symbol: Any = None) -> "_DateRule":
        return _DateRule(symbol)

    @staticmethod
    def every_month(symbol: Any = None) -> "_DateRule":
        return _DateRule(symbol)


class TimeRules:
    @staticmethod
    def after_market_open(symbol: Any = None, minutes: int = 0) -> "_TimeRule":
        return _TimeRule(minutes_after_open=int(minutes))

    @staticmethod
    def before_market_close(symbol: Any = None, minutes: int = 0) -> "_TimeRule":
        return _TimeRule(minutes_after_open=MINUTES_PER_SESSION - int(minutes))

    @staticmethod
    def before_market_open(symbol: Any = None, minutes: int = 0) -> "_TimeRule":
        return _TimeRule(-int(minutes))

    @staticmethod
    def every_day(symbol: Any = None) -> "_TimeRule":
        return _TimeRule(0)

    @staticmethod
    def every_week(symbol: Any = None) -> "_TimeRule":
        return _TimeRule(0)

    @staticmethod
    def at(day: int, hour: int = 0, minute: int = 0) -> "_TimeRule":
        return _TimeRule(0, absolute=(day, hour, minute))


@dataclass
class _DateRule:
    symbol: Any = None


@dataclass
class _TimeRule:
    """Regla horaria. `minutes_after_open` se cuenta desde las 09:30."""

    minutes_after_open: int = 0
    absolute: Optional[tuple] = None

    @property
    def offset_minutes(self) -> int:
        """Desplazamiento en minutos de sesion respecto a la apertura."""
        return int(self.minutes_after_open)


@dataclass
class ScheduleFunction:
    date_rule: _DateRule
    time_rule: _TimeRule
    handler: Callable
    name: str = ""


class ScheduleManager:
    def __init__(self, engine: "LeanEngine"):
        self._engine = engine
        self._items: List[ScheduleFunction] = []

    def on(self, date_rule: Any, time_rule: Any, handler: Callable, name: str = "") -> None:
        self._items.append(ScheduleFunction(date_rule, time_rule, handler, name))

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> List[ScheduleFunction]:
        return self._items


# --------------------------------------------------------------------------- #
# QCAlgorithm
# --------------------------------------------------------------------------- #
class QCAlgorithm:
    """
    Clase base compatible con LEAN.

    El motor inyecta la referencia al runtime mediante `_engine` antes de llamar a
    `initialize()`, de modo que todos los servicios (datos, cartera, planificador)
    estan disponibles en ese momento.
    """

    # --- referencias al runtime (asignadas por LeanEngine) ---
    _engine: "LeanEngine"
    _securities: Dict[Symbol, Security]
    _symbols: Dict[str, Symbol]
    _indicators: Dict[str, Indicator]
    _pending_orders: List[Order]
    _plot_data: Dict[str, Dict[str, List[Any]]]
    _logs: List[Dict[str, Any]]
    _current_time: Any
    _cash: float
    _start_date: Any
    _end_date: Any
    _benchmark: Any
    _warmup_count: int
    _warmup_resolution: str
    _leverage: float

    # --- estado de negocio ---
    portfolio: Portfolio
    securities: Dict[Symbol, Security]
    schedule: ScheduleManager
    date_rules: type = DateRules
    time_rules: type = TimeRules
    time: Any = None
    is_warming_up: bool = False

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:  # pragma: no cover - lo define la estrategia
        pass

    def on_data(self, data: Slice) -> None:  # pragma: no cover
        pass

    def on_order_event(self, order_event: OrderEvent) -> None:  # pragma: no cover
        pass

    def on_end_of_algorithm(self) -> None:  # pragma: no cover
        pass

    def on_start(self) -> None:  # pragma: no cover
        pass

    def on_secure_changed(self, obj) -> None:  # pragma: no cover
        pass

    def on_market_close(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------ #
    # Configuracion
    # ------------------------------------------------------------------ #
    def set_start_date(self, year: int, month: int, day: int) -> None:
        self._start_date = pd.Timestamp(int(year), int(month), int(day))

    def set_end_date(self, year: int, month: int, day: int) -> None:
        self._end_date = pd.Timestamp(int(year), int(month), int(day))

    def set_cash(self, amount: float) -> None:
        self._cash = float(amount)
        self.portfolio.set_cash(float(amount))

    def set_start(self, timestamp: Any) -> None:
        self.set_start_date(pd.Timestamp(timestamp).year, pd.Timestamp(timestamp).month,
                            pd.Timestamp(timestamp).day)

    def set_end(self, timestamp: Any) -> None:
        self.set_end_date(pd.Timestamp(timestamp).year, pd.Timestamp(timestamp).month,
                          pd.Timestamp(timestamp).day)

    def set_warm_up(self, count: int, resolution: Any = "Daily") -> None:
        self._warmup_count = int(count)
        self._warmup_resolution = Resolution.parse(resolution)

    def set_benchmark(self, symbol: Any) -> None:
        self._benchmark = _as_symbol(symbol)

    def set_name(self, name: str) -> None:
        self._algorithm_name = name

    def set_account_currency(self, *a, **k) -> None:
        pass

    def set_brokerage_model(self, *a, **k) -> None:
        pass

    def set_algorithm_id(self, *a, **k) -> None:
        pass

    def set_security_identifier(self, *a, **k) -> None:
        pass

    def set_time_zone(self, *a, **k) -> None:
        pass

    def set_fill_model(self, *a, **k) -> None:
        pass

    def set_margin_model(self, *a, **k) -> None:
        pass

    def add_equity(self, ticker: str, resolution: Any = "Minute", *a, **k) -> Security:
        sym = self._register_symbol(ticker)
        sec = Security(sym, self._engine)
        self._securities[sym] = sec
        self._engine.register_security(sec, Resolution.parse(resolution))
        return sec

    def add_index(self, ticker: str, resolution: Any = "Daily", *a, **k) -> Security:
        return self.add_equity(ticker, resolution)

    def add_fx(self, ticker: str, resolution: Any = "Daily", *a, **k) -> Security:
        return self.add_equity(ticker, resolution)

    def add_future(self, ticker: str, resolution: Any = "Daily", *a, **k) -> Security:
        return self.add_equity(ticker, resolution)

    def add_crypto(self, ticker: str, resolution: Any = "Minute", *a, **k) -> Security:
        return self.add_equity(ticker, resolution)

    def add(self, ticker: str, resolution: Any = "Minute", *a, **k) -> Security:
        return self.add_equity(ticker, resolution)

    def add_data(self, *a, **k) -> None:
        pass

    def unify(self, *a, **k) -> None:
        pass

    def _register_symbol(self, ticker: Any) -> Symbol:
        sym = _as_symbol(ticker)
        if sym.value not in self._symbols:
            self._symbols[sym.value] = sym
        return sym

    # ------------------------------------------------------------------ #
    # Datos
    # ------------------------------------------------------------------ #
    def securities_dict(self) -> Dict[Symbol, Security]:
        return self._securities

    def securities_contains(self, key: Any) -> bool:
        sym = _as_symbol(key)
        return sym in self._securities

    def history(self, symbol: Any, count: int, resolution: Any = "Daily",
                data_dictionary: Any = None, *a, **k) -> pd.DataFrame:
        """Historico OHLCV de los ultimos `count` periodos hasta el instante actual."""
        return self._engine.history(_as_symbol(symbol), int(count), Resolution.parse(resolution))

    def consolidators(self, symbol: Any, resolution: Any) -> Any:
        return None

    def bars(self, *a, **k) -> pd.DataFrame:
        return pd.DataFrame()

    def get_parameter(self, name: str, *a, **k) -> Any:
        """Valor de un parametro de backtest. Devuelve None si no se supplied."""
        return self._engine.parameters.get(name)

    def get_root_algorithm(self):
        return self

    @property
    def algorithm_name(self) -> str:
        return getattr(self, "_algorithm_name", type(self).__name__)

    # ------------------------------------------------------------------ #
    # Indicadores
    # ------------------------------------------------------------------ #
    def sma(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("sma", symbol, period, resolution)

    def ema(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("ema", symbol, period, resolution)

    def wma(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("wma", symbol, period, resolution)

    def rsi(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("rsi", symbol, period, resolution)

    def atr(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("atr", symbol, period, resolution)

    def macd(self, symbol: Any, period: int = 12, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("macd", symbol, period, resolution)

    def stddev(self, symbol: Any, period: int, resolution: Any = "Daily", *a, **k) -> Indicator:
        return self._indicator("stddev", symbol, period, resolution)

    def _indicator(self, name: str, symbol: Any, period: int, resolution: Any) -> Indicator:
        sym = _as_symbol(symbol)
        res = Resolution.parse(resolution)
        key = f"{name}_{sym.value}_{int(period)}_{res}"
        if key in self._indicators:
            return self._indicators[key]
        ind = Indicator(name, sym, int(period), res, self._engine)
        self._indicators[key] = ind
        return ind

    # ------------------------------------------------------------------ #
    # Ordenes
    # ------------------------------------------------------------------ #
    def market_order(self, symbol: Any, quantity: float, *a, **k) -> Order:
        sym = _as_symbol(symbol)
        qty = float(quantity)
        direction = OrderDirection.BUY if qty >= 0 else OrderDirection.SELL
        return self._engine.submit(sym, abs(qty), direction, "Market")

    def limit_order(self, symbol: Any, quantity: float, limit_price: float, *a, **k) -> Order:
        sym = _as_symbol(symbol)
        qty = float(quantity)
        direction = OrderDirection.BUY if qty >= 0 else OrderDirection.SELL
        return self._engine.submit(sym, abs(qty), direction, "Limit", limit_price=float(limit_price))

    def stop_order(self, symbol: Any, quantity: float, stop_price: float, *a, **k) -> Order:
        return self.limit_order(symbol, quantity, stop_price)

    def market_order_percent(self, symbol: Any, percent: float, *a, **k) -> Order:
        sym = _as_symbol(symbol)
        objetivo = self.portfolio.total_portfolio_value * float(percent)
        precio = self._engine.last_price(sym)
        if precio <= 0:
            return self._engine.submit(sym, 0.0, OrderDirection.BUY, "Market")
        qty = objetivo / precio
        return self.market_order(sym, qty)

    def set_holdings(self, symbol: Any, weight: float, *a, **k) -> Order:
        """Iguala el valor de la posicion al `weight` indicado del equity."""
        sym = _as_symbol(symbol)
        peso = float(weight)
        total = self.portfolio.total_portfolio_value
        precio = self._engine.last_price(sym)
        if precio <= 0:
            return self._engine.submit(sym, 0.0, OrderDirection.BUY, "Market")
        objetivo = total * peso
        actual = self.portfolio[sym].holdings_value
        qty = (objetivo - actual) / precio
        if abs(qty) < 1e-9:
            return self._engine.submit(sym, 0.0, OrderDirection.BUY, "Market")
        return self.market_order(sym, qty)

    def set_holdings_percent(self, symbol: Any, percent: float, *a, **k) -> Order:
        return self.set_holdings(symbol, float(percent))

    def liquidate(self, symbol: Any = None, *a, **k) -> List[Order]:
        """Cierra total la posicion indicada, o todas si no se indica ninguna."""
        if symbol is None:
            return [self.market_order(s, -abs(self.portfolio[s].quantity))
                    for s in list(self._holdings_invested())]
        sym = _as_symbol(symbol)
        qty = self.portfolio[sym].quantity
        if abs(qty) < 1e-12:
            return []
        return [self.market_order(sym, -abs(qty))]

    def liquidate_all(self, *a, **k) -> List[Order]:
        return self.liquidate(None)

    def _holdings_invested(self) -> Iterable[Symbol]:
        return [s for s, h in self.portfolio.securities.items() if h.invested]

    # ------------------------------------------------------------------ #
    # Diagnostico
    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        self._engine.emit_log("INFO", str(message), self._current_time)

    def debug(self, message: str) -> None:
        self._engine.emit_log("DEBUG", str(message), self._current_time)

    def error(self, message: str) -> None:
        self._engine.emit_log("ERROR", str(message), self._current_time)

    def warn(self, message: str) -> None:
        self._engine.emit_log("WARNING", str(message), self._current_time)

    def plot(self, title: str, series: str, value: Any, *a, **k) -> None:
        store = self._plot_data.setdefault(str(title), {})
        arr = store.setdefault(str(series), [])
        arr.append((self._current_time, float(value) if value is not None else float("nan")))

    def plot_time_series(self, *a, **k) -> None:
        pass

    def draw(self, *a, **k) -> None:
        pass

    @property
    def current_slice(self) -> Optional[Slice]:
        return self._engine.current_slice

    def transact_market_order(self, *a, **k):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registro del modulo `AlgorithmImports`
# --------------------------------------------------------------------------- #
_ALGORITHM_EXPORTS = {
    "QCAlgorithm": QCAlgorithm,
    "AlgorithmImports": None,
    "Resolution": Resolution,
    "Slice": Slice,
    "TradeBar": TradeBar,
    "Symbol": Symbol,
    "Security": Security,
    "Portfolio": Portfolio,
    "Holding": Holding,
    "Order": Order,
    "OrderEvent": OrderEvent,
    "OrderStatus": OrderStatus,
    "OrderDirection": OrderDirection,
    "FeeOrder": FeeOrder,
    "Indicator": Indicator,
    "IndicatorPoint": IndicatorPoint,
    "DateRules": DateRules,
    "TimeRules": TimeRules,
    "ScheduleManager": ScheduleManager,
    "InteractiveBrokersFeeModel": InteractiveBrokersFeeModel,
    "FeeModel": InteractiveBrokersFeeModel,
    "ConstantFeeModel": ConstantFeeModel,
    "PercentFeeModel": PercentFeeModel,
    "ZeroFeeModel": ZeroFeeModel,
    "ConstantSlippageModel": ConstantSlippageModel,
    "ZeroSlippageModel": ZeroSlippageModel,
    "MarketFillModel": MarketFillModel,
    "SlippageModel": ConstantSlippageModel,
}


def install_algorithm_imports() -> types.ModuleType:
    """
    Registra `AlgorithmImports` en `sys.modules`.

    Es lo que permite ejecutar los `.py` del usuario sin editar una sola linea:
    su `from AlgorithmImports import *` resuelve contra este modulo y las
    estrategias heredan de la QCAlgorithm de este runtime.
    """
    mod = types.ModuleType("AlgorithmImports")
    for nombre, obj in _ALGORITHM_EXPORTS.items():
        if obj is not None:
            setattr(mod, nombre, obj)
    # Alias de broker habituales en proyectos reales
    mod.InteractiveBrokersBrokerageModel = InteractiveBrokersFeeModel
    mod.FeeModelForOrder = InteractiveBrokersFeeModel
    mod.ZeroSlippage = ZeroSlippageModel
    mod.Equity = Security
    mod.Crypto = Security
    mod.Forex = Security
    mod.Future = Security
    mod.Consolidator = object
    sys.modules["AlgorithmImports"] = mod
    return mod