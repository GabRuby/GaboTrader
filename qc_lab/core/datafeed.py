"""
Generador determinista de datos OHLC simulados (equivalente a un "data feed" de LEAN).

QuantConnect obtiene las velas de un proveedor externo. Para que este laboratorio
sea 100% reproducible y funcione sin conexion, generamos series sinteticas pero
realistas:

  * Un factor de mercado COMUN a todos los activos, con cambios de regimen
    (alcista / bajista / crisis) y volatilidad estocastica AR(1).
  * Un componente idiosyncratico por ticker, calibrado para respetar la
    correlacion declarada con el mercado.
  * Volatilidad y deriva ANUALIZADAS exactamente iguales a las del perfil, de
    modo que las logicas de *volatility targeting* de las estrategias operen
    sobre cifras conocidas y el backtest sea reproducible bit a bit.

Velas disponibles:
  * Diarias -> base para el warm-up, SMA/EMA, `history()` y el benchmark.
  * Minuto  -> Expansion exacta de la vela diaria: el maximo, minimo, apertura y
               cierre de la sesion diaria coinciden con el agregado de sus 390
               minutos (390 = 6,5 h de sesion Americana).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd

MINUTES_PER_SESSION = 390          # 09:30 -> 16:00
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SEED = 20240607
MARKET_OPEN_MINUTE = 570          # 9 h 30 min en minutos desde medianoche

# --------------------------------------------------------------------------- #
# Origen fijo de las series
# --------------------------------------------------------------------------- #
# REGLA CRITICA DE CONSISTENCIA
#
# El precio de un activo debe ser una funcion PURA de (ticker, fecha). Por eso
# TODAS las series se generan siempre desde `EPOCH` hasta la fecha pedida y
# despues se recortan a la ventana solicitada. Si la semilla incluyera `start`
# o `end`, cada ventana reconstruiria una trayectoria de precios DISTINTA y las
# velas diarias y de minuto del mismo activo dejarian de describir el mismo
# mercado: un SMA20 calculado sobre una ventana daria un valor que no corresponde
# con el precio de la vela de minuto, y las estrategias de tendencia nunca se
# activarian. Ademas, `base_price` se interpretaria como el precio al inicio de
# cada ventana en lugar de una fecha de referencia estable.
EPOCH = "2005-01-03"

# Horizonte fijo hasta el que se genera SIEMPRE la serie antes de recortarla.
# Es imprescindible que la longitud no dependa de `end`: los pasos de
# normalizacion (centrar y escalar a la volatilidad objetivo) se calculan sobre el
# vector completo, de modo que cambiar la longitud desplazaria toda la
# trayectoria y el precio de una fecha dada variaria segun la ventana pedida.
HORIZONTE = "2035-12-31"

# Fecha a la que se ancla `base_price` de cada perfil. Elegida cerca del "hoy"
# para que los precios actuales sean realistas y el historico se obtenga
# retrocediendo geometricamente segun el CAGR del activo.
FECHA_ANCLA = "2026-01-02"

# Volatilidad anualizada asignada al factor de mercado antes de aplicar el beta.
SIGMA_MERCADO_ANUAL = 0.16
# Deriva anualizada del factor de mercado (se recentra luego al perfil del activo).
DERIVA_MERCADO_ANUAL = 0.10

# Peso del PATRON de regimen en la deriva del activo.
#
# Antes se dividia por la ventana pedida y el patron se renormalizaba a la deriva
# media, pero eso rompia la consistencia entre ventanas (el mismo dia cambiava de
# precio segun el rango solicitado). Ahora la serie es unica, asi que el patron
# actua de verdad y hay que calibrarlo con cuidado:
#
#   * Con 0.50 la modulacion reachaba +-79% anualizado en SOXL (beta ~3.9) y
#     ahogaba por completo el CAGR objetivo: 159% real frente a 29% previsto.
#   * Con 0.10 la crisis sigue castigando a los ETF apalancados y el bovino sigue
#     pareciendose a un deposito, pero la trayectoria respeta el CAGR del perfil.
PESO_REGIMEN = 0.10

# NOTA sobre el deriva a largo plazo.
#
# Al componer `beta * mercado + idiosincratico`, el término idiosincratico es un
# paseo aleatorio cuyo sesgo medido en una ventana de 3 años tiene una desviacion
# tipica de `vol_anual / sqrt(3)`: para SOXL (74%) unos +-43% ANUALES. Se probo
# un "puente de deriva" acotado que corrigiese esa desviacion acumulada, pero
# media los log-retornos diarios y degradaba la volatilidad realizada, que es la
# propiedad que mas importa para las estrategias de volatility targeting:
#
#     PESO_PUENTE = 0.55   ->  SHV 2,6% -> 1,4%   SPY 16,0% -> 14,4%   AAPL 28% -> 21%
#
# Por eso NO se corrige: la volatilidad y las correlaciones quedan calibradas al
# perfil con exactitud, y la desviacion del CAGR en ventanas cortas se acepta
# como realista (SOXL realmente multiplica su precio varias veces en tres anos).


# --------------------------------------------------------------------------- #
# Perfiles de activo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SymbolProfile:
    """
    Descripcion estadistica de un ticker.

    En lugar de fijar beta y volatilidad idiosyncratica por separado (lo que
    obliga a mantenerlos coherentes a mano), se declara la CORRELACION con el
    mercado y se derivan ambos componentes:

        beta      = vol_anual * corr / sigma_mercado
        vol_idio  = vol_anual * sqrt(1 - corr^2)

    De este modo `sqrt((beta*sigma_m)^2 + vol_idio^2) == vol_anual` siempre.
    """

    ticker: str
    base_price: float      # Precio de partida
    annual_drift: float    # CAGR objetivo
    annual_vol: float      # Volatilidad anualizada objetivo
    corr_market: float     # Correlacion con el factor de mercado

    @property
    def beta(self) -> float:
        return self.annual_vol * self.corr_market / SIGMA_MERCADO_ANUAL

    @property
    def idio_vol(self) -> float:
        return self.annual_vol * float(np.sqrt(max(1.0 - self.corr_market**2, 0.0)))


# Universo cubierto de forma realista. Los ETF apalancados (SOXL/TQQQ) replican
# ~3x el retorno diario del indice, por eso su correlacion con SPY es alta y su
# volatilidad triplica la del mercado.
UNIVERSE: Dict[str, SymbolProfile] = {
    "SPY":  SymbolProfile("SPY",  280.0, 0.105, 0.16, 0.98),
    "SPX":  SymbolProfile("SPX",  3120.0, 0.100, 0.15, 1.00),
    "QQQ":  SymbolProfile("QQQ",  175.0, 0.135, 0.21, 0.90),
    "IWM":  SymbolProfile("IWM",  145.0, 0.085, 0.21, 0.85),
    "SOXL": SymbolProfile("SOXL", 38.0, 0.290, 0.74, 0.85),
    "TQQQ": SymbolProfile("TQQQ", 42.0, 0.310, 0.80, 0.85),
    "UPRO": SymbolProfile("UPRO", 48.0, 0.300, 0.76, 0.84),
    "SPXL": SymbolProfile("SPXL", 52.0, 0.170, 0.38, 0.85),
    "SHV":  SymbolProfile("SHV",   91.0, 0.022, 0.026, 0.00),
    "SHY":  SymbolProfile("SHY",   82.0, 0.021, 0.033, 0.00),
    "BIL":  SymbolProfile("BIL",   91.5, 0.026, 0.025, 0.00),
    "TLT":  SymbolProfile("TLT",   88.0, 0.010, 0.15, -0.55),
    "GLD":  SymbolProfile("GLD",  155.0, 0.055, 0.14, 0.10),
    "TSLA": SymbolProfile("TSLA", 195.0, 0.115, 0.56, 0.50),
    "AAPL": SymbolProfile("AAPL", 165.0, 0.145, 0.28, 0.60),
    "MSFT": SymbolProfile("MSFT", 320.0, 0.130, 0.25, 0.55),
    "NVDA": SymbolProfile("NVDA", 120.0, 0.310, 0.50, 0.60),
    "AMZN": SymbolProfile("AMZN", 160.0, 0.130, 0.30, 0.55),
    "META": SymbolProfile("META", 350.0, 0.135, 0.34, 0.58),
    "AMD":  SymbolProfile("AMD",  120.0, 0.180, 0.46, 0.65),
    "VIX":  SymbolProfile("VIX",   18.0, 0.000, 0.85, -0.35),
    "BTC":  SymbolProfile("BTC", 42000.0, 0.280, 0.72, 0.55),
    "ETH":  SymbolProfile("ETH",  2400.0, 0.250, 0.78, 0.55),
}


def get_profile(ticker: str) -> SymbolProfile:
    """Perfil del ticker o, si es desconocido, uno generico pero estable."""
    t = str(ticker).upper()
    if t in UNIVERSE:
        return UNIVERSE[t]
    crc = zlib.crc32(t.encode("utf-8"))
    rng = np.random.default_rng(DEFAULT_SEED + crc)
    return SymbolProfile(
        ticker=t,
        base_price=float(rng.uniform(20.0, 320.0)),
        annual_drift=float(rng.uniform(0.02, 0.16)),
        annual_vol=float(rng.uniform(0.18, 0.55)),
        corr_market=float(rng.uniform(0.20, 0.70)),
    )


def _stable_seed(*parts: str) -> int:
    """Semilla reproducible (no usa hash(), que Python salta entre procesos)."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return (DEFAULT_SEED + zlib.crc32(raw)) % (2**31 - 1)


# --------------------------------------------------------------------------- #
# Calendario
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Dias habiles (lunes a viernes) entre dos fechas, sin festivos."""
    return pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="B")


# --------------------------------------------------------------------------- #
# Factor de mercado con cambios de regimen
# --------------------------------------------------------------------------- #
def _market_factor(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Log-retornos diarios EXCEDENTES del mercado + deriva por regimen.

    Cadena de Markov de 3 estados que modula la volatilidad, con un factor de
    volatilidad AR(1) suave.

    Devuelve (rets, drift_regimen_anual):
      * `rets`            -> componente de retorno sin deriva (se centra luego).
      * `drift_regimen`   -> deriva anualizada asociada a cada dia, para poder
                             conservar el patron por regimen y a la vez recentrar
                             la deriva media al objetivo del perfil.
    """
    # 0 = alcista, 1 = bajista, 2 = crisis
    drift_estado = np.array([0.175, -0.115, -0.42])
    vol_estado = np.array([0.118, 0.205, 0.46])
    transicion = np.array([
        [0.9965, 0.0033, 0.0002],
        [0.0060, 0.9925, 0.0015],
        [0.0250, 0.0650, 0.9100],
    ])

    estado = 0
    rets = np.empty(n, dtype=float)
    drift_reg = np.empty(n, dtype=float)
    # AR(1) en logaritmos. La innovacion (0.035) fija una desviacion estacionaria
    # de ~0.14: modula la serie sin dominar la volatilidad realizada.
    phi, sigma = 0.97, 0.035
    log_vol = 0.0
    for i in range(n):
        estado = int(rng.choice(3, p=transicion[estado]))
        log_vol = phi * log_vol + sigma * rng.normal()
        vol_mult = float(np.exp(log_vol - (sigma**2) / (2 * (1 - phi**2))))
        rets[i] = vol_estado[estado] * vol_mult / np.sqrt(TRADING_DAYS_PER_YEAR) * rng.normal()
        drift_reg[i] = drift_estado[estado]
    return rets, drift_reg


# --------------------------------------------------------------------------- #
# Velas diarias
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=512)
def daily_bars(ticker: str, start: str, end: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Velas diarias sinteticas y deterministas para un ticker.

    La serie se construye SIEMPRE completa desde `EPOCH` hasta `end` y luego se
    recorta a `[start, end]`. De ese modo `daily_bars("SOXL", a, b)` es siempre
    un subconjunto de `daily_bars("SOXL", EPOCH, b)`: el precio de una fecha
    concreta no depende de la ventana pedida, lo que hace que las velas diarias
    y de minuto sean siempre coherentes entre si.
    """
    perfil = get_profile(ticker)
    vacio = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    dias = trading_days(EPOCH, HORIZONTE)
    if len(dias) == 0:
        return vacio

    n = len(dias)
    # Las semillas NO dependen de la ventana: solo del ticker (y del factor
    # comun de mercado), para que la trayectoria sea unica y estable.
    rng = np.random.default_rng(_stable_seed(ticker, seed, "daily"))

    # CRITICO: el factor de mercado es COMUN a todos los activos, asi que su
    # semilla NO debe depender del ticker. Si lo hiciera, cada simbolo tendria un
    # "mercado" distinto, la correlacion entre activos seria ~0 y los filtros
    # macro (SPY > SMA) de las estrategias dejarian de tener sentido.
    mkt, drift_reg = _market_factor(n, np.random.default_rng(_stable_seed("__MKT__", seed, "daily")))

    # 1) Normalizar el factor a la volatilidad objetivo del mercado.
    mkt = mkt - mkt.mean()
    sd = mkt.std()
    if sd > 0:
        mkt = mkt * (SIGMA_MERCADO_ANUAL / np.sqrt(TRADING_DAYS_PER_YEAR)) / sd

    # 2) Sumar el componente idiosyncratico, ambos en escala DIARIA.
    idio = rng.normal(0.0, 1.0, n)
    idio = idio - idio.mean()
    sd_idio = idio.std()
    if sd_idio > 0:
        idio = idio / sd_idio
    log_ret = perfil.beta * mkt + (perfil.idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)) * idio

    # 3) Escalar el conjunto para que la volatilidad anualizada REALizada coincida
    #    con la del perfil (garantiza que el volatility targeting sea coherente).
    objetivo = perfil.annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    log_ret = log_ret - log_ret.mean()
    sd_real = log_ret.std()
    if sd_real > 0:
        log_ret = log_ret * (objetivo / sd_real)

    # 4) Deriva. Se conserva el PATRON por regimen (dias alcistas positivos, dias
    #    de crisis muy negativos) pero escalado por el BETA del activo: una crisis
    #    castiga mucho mas a un ETF 3x que a un ETF de bonos, y SHV (beta ~0) debe
    #    comportarse como un deposito, no como una accion. Como el patron tiene
    #    media cero, el CAGR resultante sigue siendo el del perfil.
    patron = (drift_reg - drift_reg.mean()) * (perfil.beta * PESO_REGIMEN) / TRADING_DAYS_PER_YEAR
    log_ret = log_ret + patron + perfil.annual_drift / TRADING_DAYS_PER_YEAR

    # 5) Anclar el nivel de precio: `base_price` es el precio en FECHA_ANCLA y el
    #    historico se obtiene retrocediendo segun los log-retornos acumulados.
    acumulado = np.cumsum(log_ret)
    pos_ancla = int(dias.searchsorted(pd.Timestamp(FECHA_ANCLA), side="right")) - 1
    pos_ancla = max(0, min(pos_ancla, n - 1))
    close = perfil.base_price * np.exp(acumulado - acumulado[pos_ancla])
    prev_close = np.concatenate(
        ([perfil.base_price * np.exp(-acumulado[0])], close[:-1])
    )
    # Apertura: gap nocturno que aporta parte del movimiento fuera de sesion.
    vol_diaria = np.abs(log_ret)
    gap = rng.normal(0.0, 1.0, n) * vol_diaria * 0.32 - log_ret * 0.12
    open_ = prev_close * np.exp(gap)

    # Rango intradiario: cola densa alrededor del cierre + mecha aleatoria.
    u = np.abs(rng.normal(0.0, 1.0, n))
    rango_frac = np.clip(vol_diaria * 1.15 + u * vol_diaria * 0.85 + 0.0012, 0.0008, 0.35)
    high = np.maximum(open_, close) * (1.0 + rango_frac * rng.uniform(0.25, 0.75, n))
    low = np.minimum(open_, close) * (1.0 - rango_frac * rng.uniform(0.25, 0.75, n))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    # Volumen: lognormal amplificado por la magnitud del movimiento del dia.
    volumen = np.exp(rng.normal(15.6, 0.42, n)) * (1.0 + 6.0 * np.abs(log_ret))
    volumen = np.round(volumen / 100.0) * 100.0

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volumen},
        index=pd.DatetimeIndex(dias, name="date"),
    )
    df = df.round(4)

    # Recorte final a la ventana pedida. Se hace al final, sobre la serie ya
    # anclada, para que el nivel de precio no dependa de la ventana.
    ini = pd.Timestamp(start)
    fin = pd.Timestamp(end)
    return df.loc[(df.index >= ini) & (df.index <= fin)]


# --------------------------------------------------------------------------- #
# Velas de minuto (expansion exacta de la vela diaria)
# --------------------------------------------------------------------------- #
def _formas_intradiarias(ticker: str, dias: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Camino y reparto de volumen intradiario, uno por dia y deterministas.

    La semilla se deriva del TICKER y de la FECHA, nunca de la ventana pedida, de
    modo que la forma de una vela de minuto solo depende de su dia. Si se usara
    un unico generador para toda la serie, al pedir una ventana distinta se
    consumirian numeros distintos y las horas de un mismo dia cambiarian.
    """
    m = len(dias)
    camino = np.empty((m, MINUTES_PER_SESSION - 1), dtype=float)
    reparto = np.empty((m, MINUTES_PER_SESSION), dtype=float)

    # Forma en U: mas liquidez a la apertura y al cierre.
    j = np.arange(MINUTES_PER_SESSION)
    peso_u = 1.0 + 1.3 * np.exp(-j / 28.0) + 1.3 * np.exp(-(MINUTES_PER_SESSION - 1 - j) / 28.0)

    for i in range(m):
        rng = np.random.default_rng(_stable_seed(ticker, str(dias[i].date()), "intraday"))
        camino[i] = np.cumsum(rng.normal(0.0, 1.0, MINUTES_PER_SESSION - 1))
        g = rng.standard_gamma(0.55, MINUTES_PER_SESSION) * peso_u
        total = g.sum()
        reparto[i] = g / total if total > 0 else 1.0 / MINUTES_PER_SESSION

    return camino, reparto


@lru_cache(maxsize=64)
def minute_bars(ticker: str, start: str, end: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Expande cada vela diaria en 390 velas de minuto de forma EXACTA.

    Construccion (391 nodos -> 390 velas):
      * El primer nodo es el `open` diario y el ultimo el `close` diario.
      * Los 389 nodos INTERIORES se generan como un camino aleatorio normalizado
        a [0, 1] y se escalan al rango [low, high].

    Por construccion, el maximo global de los nodos es exactamente `high`
    (porque los interiores tocan `high` y open/close <= high) y el minimo es
    exactamente `low`. Esa es la propiedad que hace que la vela diaria sea el
    agregado fiel de sus minutos.
    """
    diario = daily_bars(ticker, start, end, seed)
    if diario.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    m = len(diario)

    d_open = diario["open"].to_numpy()
    d_high = diario["high"].to_numpy()
    d_low = diario["low"].to_numpy()
    d_close = diario["close"].to_numpy()

    camino, fracs = _formas_intradiarias(ticker, diario.index)

    # Camino aleatorio normalizado a [0, 1]: min = 0 y max = 1 exactamente.
    camino = camino - camino[:, :1]
    span = camino.max(axis=1) - camino.min(axis=1)
    span[span == 0] = 1.0
    forma = (camino - camino.min(axis=1, keepdims=True)) / span[:, None]

    interior = d_low[:, None] + (d_high - d_low)[:, None] * forma
    # 389 nodos interiores + apertura + cierre = 391 nodos = 390 velas.
    nodos = np.concatenate([d_open[:, None], interior, d_close[:, None]], axis=1)

    n0, n1 = nodos[:, :-1], nodos[:, 1:]
    abrir = np.minimum(n0, n1)
    cerrar = np.maximum(n0, n1)

    # Se fijan apertura del primer minuto y cierre del ultimo. Como open y close estan
    # dentro de [low, high] por construccion, los extremos diarios no se alteran y
    # basta con tomar min/max de los dos bordes de cada vela.
    abrir[:, 0] = d_open
    cerrar[:, -1] = d_close
    maximo = np.maximum(abrir, cerrar)
    minimo = np.minimum(abrir, cerrar)

    # Volumen intradiario con forma en U: mas liquido a la apertura y al cierre.
    vol_min = np.round(diario["volume"].to_numpy()[:, None] * fracs / 100.0) * 100.0

    # Indice temporal. pandas>=3 no difunde Indice + Indice, de ahi la suma
    # exterior con numpy sobre datetime64/timedelta64.
    base = diario.index.to_numpy()
    j = np.arange(MINUTES_PER_SESSION)
    off = pd.to_timedelta(j + MARKET_OPEN_MINUTE, unit="m").to_numpy()
    idx = pd.DatetimeIndex((base[:, None] + off[None, :]).reshape(-1), name="date")

    frame = pd.DataFrame(
        {
            "open": abrir.reshape(-1),
            "high": maximo.reshape(-1),
            "low": minimo.reshape(-1),
            "close": cerrar.reshape(-1),
            "volume": vol_min.reshape(-1),
        },
        index=idx,
    )
    return frame.round(4)


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def bars(ticker: str, resolution: str, start, end, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Velas pedidas. `resolution` acepta "minute"/"min" o "daily"."""
    ini = pd.Timestamp(start).strftime("%Y-%m-%d")
    fin = pd.Timestamp(end).strftime("%Y-%m-%d")
    if str(resolution).lower() in ("minute", "min", "m"):
        return minute_bars(str(ticker).upper(), ini, fin, seed)
    return daily_bars(str(ticker).upper(), ini, fin, seed)


def available_tickers() -> list[str]:
    """Tickers con datos de alta calidad en el universo simulado."""
    return sorted(UNIVERSE.keys())