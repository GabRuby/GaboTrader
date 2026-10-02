#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST LOCAL v3 - Dual Momentum + Volatility Targeting (SOXL / TQQQ / SHV, filtro SPY)

Replica en Python puro la logica de `dual_momentum_core_v3.py` (QuantConnect), sin depender
de ninguna plataforma. Datos gratuitos de Yahoo Finance via yfinance.

INSTALACION
    pip install yfinance pandas numpy matplotlib

USO
    python backtest_local_v3.py                       # corrida A (v3 por defecto)
    python backtest_local_v3.py --ablacion            # corridas A, B, C, D, E en una tabla
    python backtest_local_v3.py --fin 2026-07-02      # mismo corte que tu backtest de QuantConnect
    python backtest_local_v3.py --inicio 2012-01-01 --fin 2019-12-31   # validacion fuera de muestra
    python backtest_local_v3.py --demo                # datos sinteticos (solo para probar que corre)

MODELO DE EJECUCION (aproximado, a proposito conservador)
    - Las senales usan datos hasta el CIERRE del dia anterior (sin look-ahead).
    - Las ordenes se ejecutan a la APERTURA del dia siguiente, con slippage adverso.
    - Comisiones tipo Interactive Brokers: $0.005/accion, minimo $1, maximo 1% del valor,
      mas una tasa regulatoria pequena en ventas. NO es identico al fee model de QuantConnect.
    - Acciones enteras (sin fracciones), igual que en QuantConnect.
    - Precios ajustados por dividendos (auto_adjust): el rendimiento incluye dividendos, pero los
      precios historicos se ven mas bajos que los reales, lo que afecta un poco el redondeo a
      acciones enteras (sobre todo en SHV).

IMPORTANTE: es una herramienta para COMPARAR versiones entre si. No es una promesa de
rentabilidad ni un sustituto de validacion fuera de muestra y paper trading.
"""

import argparse
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ACTIVOS_RIESGO = ["SOXL", "TQQQ"]
REFUGIO = "SHV"
MERCADO = "SPY"
OPERABLES = ACTIVOS_RIESGO + [REFUGIO]
TICKERS = OPERABLES + [MERCADO]

NIVELES = [0.33, 0.66, 1.0]
SEMI_PASO = 0.165


# --------------------------------------------------------------------------- #
# Parametros (mismos nombres que en el codigo de QuantConnect)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    capital: float = 500.0
    vol_target: float = 0.35
    lookback_mom: int = 60
    lookback_vol: int = 20
    peso_maximo: float = 0.95
    peso_minimo: float = 0.05
    banda: float = 0.15
    margen_cambio: float = 0.05
    min_orden_usd: float = 50.0
    eval_semanal: int = 1
    niveles_discretos: int = 1
    hist_nivel: float = 0.05
    colchon_regimen: float = 0.01
    slippage_bps: float = 5.0
    compras_diferidas: int = 0   # 1 = las compras se ejecutan al dia siguiente (simula MarketOnOpen)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def _datos_demo(ini, fin):
    """Datos sinteticos SOLO para probar que el script corre. No representan al mercado."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(ini, fin)
    n = len(idx)
    mercado = rng.normal(0.0004, 0.011, n)
    mercado[int(n * 0.55):int(n * 0.62)] -= 0.0025      # periodo bajista sintetico
    tech = 1.2 * mercado + rng.normal(0.0002, 0.006, n)
    semis = 1.1 * tech + rng.normal(0.0, 0.009, n)
    shv = 0.00008 + rng.normal(0.0, 0.00003, n)
    rets = {"SPY": mercado, "TQQQ": 3 * tech, "SOXL": 3 * semis, "SHV": shv}
    closes = pd.DataFrame(
        {k: 100 * np.cumprod(1 + np.clip(v, -0.5, 0.5)) for k, v in rets.items()}, index=idx
    )[TICKERS]
    gaps = rng.normal(0.0, 0.003, closes.shape)
    opens = closes.shift(1) * (1 + gaps)
    opens.iloc[0] = closes.iloc[0]
    return opens, closes


def descargar(inicio, fin, demo=False):
    ini_desc = pd.Timestamp(inicio) - pd.Timedelta(days=420)   # colchon para SMA200 y lookbacks
    fin_ts = pd.Timestamp(fin)
    if demo:
        return _datos_demo(ini_desc, fin_ts)

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Falta yfinance. Instala con:  pip install yfinance pandas numpy matplotlib")

    raw = yf.download(
        TICKERS,
        start=ini_desc.strftime("%Y-%m-%d"),
        end=(fin_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        sys.exit("yfinance no devolvio datos. Revisa tu conexion o intenta de nuevo en unos minutos.")

    opens, closes = raw["Open"], raw["Close"]
    faltan = [t for t in TICKERS if t not in closes.columns]
    if faltan:
        sys.exit(f"No se pudieron descargar: {faltan}")

    opens, closes = opens[TICKERS], closes[TICKERS]
    opens.index = pd.to_datetime(opens.index).tz_localize(None)
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    ok = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    return opens[ok], closes[ok]


# --------------------------------------------------------------------------- #
# Simulador
# --------------------------------------------------------------------------- #
class Simulador:
    def __init__(self, p, opens, closes, inicio):
        self.p = p
        self.fechas = closes.index
        self.O = {t: opens[t].to_numpy(dtype=float) for t in TICKERS}
        self.C = {t: closes[t].to_numpy(dtype=float) for t in TICKERS}

        spy = closes[MERCADO]
        self.s50 = spy.rolling(50).mean().to_numpy()
        self.s200 = spy.rolling(200).mean().to_numpy()
        self.mom = {
            s: (closes[s] / closes[s].shift(p.lookback_mom) - 1).to_numpy() for s in ACTIVOS_RIESGO
        }
        self.vol = {
            s: (closes[s].pct_change().rolling(p.lookback_vol).std() * np.sqrt(252)).to_numpy()
            for s in ACTIVOS_RIESGO
        }

        self.i0 = int(self.fechas.searchsorted(pd.Timestamp(inicio)))
        if self.i0 < 1 or np.isnan(self.s200[self.i0 - 1]):
            raise ValueError(
                "Historia insuficiente antes de la fecha de inicio (se necesitan ~200 dias de SPY, "
                "y SOXL/TQQQ/SHV deben existir en esa fecha). Usa un --inicio mas reciente."
            )

    # ------------------------- estado ------------------------- #
    def _reset(self):
        self.cash = float(self.p.capital)
        self.shares = {t: 0 for t in OPERABLES}
        self.activo_actual = None
        self.nivel_actual = None
        self.plan_vigente = None
        self.semana_ultima = None
        self.pend_compras = False
        self.buy_plan = None
        self.buy_operar = set()
        self.trades = []

    # ------------------------- costos ------------------------- #
    @staticmethod
    def _comision(qty, precio, venta):
        valor = qty * precio
        fee = min(max(0.005 * qty, 1.0), 0.01 * valor)
        if venta:
            fee += valor * 0.0000278 + min(0.000166 * qty, 8.30)   # tasas regulatorias aprox.
        return fee

    def _ejecutar(self, s, qty, i):
        """qty > 0 compra, qty < 0 venta. Se ejecuta a la apertura del dia i con slippage."""
        if qty == 0:
            return
        venta = qty < 0
        n = abs(int(qty))
        base = self.O[s][i]
        slip = base * self.p.slippage_bps / 10000.0
        precio = base - slip if venta else base + slip

        if venta:
            n = min(n, self.shares[s])
        else:
            while n > 0 and n * precio + self._comision(n, precio, False) > self.cash:
                n -= 1
        if n <= 0:
            return

        fee = self._comision(n, precio, venta)
        if venta:
            self.cash += n * precio - fee
            self.shares[s] -= n
        else:
            self.cash -= n * precio + fee
            self.shares[s] += n

        self.trades.append(
            {
                "fecha": self.fechas[i],
                "activo": s,
                "lado": "VENTA" if venta else "COMPRA",
                "cantidad": n,
                "precio": round(precio, 4),
                "valor": round(n * precio, 2),
                "comision": round(fee, 4),
            }
        )

    # ------------------------- valoracion ------------------------- #
    def _valor_total(self, i):
        return self.cash + sum(self.shares[s] * self.O[s][i] for s in OPERABLES)

    def _valor_pos(self, s, i):
        return self.shares[s] * self.O[s][i]

    # ------------------------- decision ------------------------- #
    def _nivel(self, ratio):
        candidato = min(NIVELES, key=lambda n: abs(n - ratio))
        actual = self.nivel_actual
        if actual is not None and abs(ratio - actual) <= SEMI_PASO + self.p.hist_nivel:
            return actual
        return candidato

    def _calcular_plan(self, prev):
        p = self.p
        spy = self.C[MERCADO][prev]
        regimen_ok = spy > self.s200[prev] and spy > self.s50[prev]

        datos = {}
        for s in ACTIVOS_RIESGO:
            m, v = self.mom[s][prev], self.vol[s][prev]
            if not (np.isnan(m) or np.isnan(v)):
                datos[s] = (m, v)
        if not datos:
            return None

        puntajes = {s: d[0] for s, d in datos.items()}
        mejor = max(puntajes, key=puntajes.get)

        elegido = mejor
        actual = self.activo_actual
        if actual in puntajes and mejor != actual:
            if puntajes[mejor] - puntajes[actual] < p.margen_cambio:
                elegido = actual

        plan = {s: 0.0 for s in OPERABLES}
        if regimen_ok and puntajes[elegido] > 0:
            vol = datos[elegido][1]
            ratio = min(1.0, p.vol_target / vol) if vol > 0 else 1.0
            if p.niveles_discretos:
                ratio = self._nivel(ratio)
                self.nivel_actual = ratio

            peso_riesgo = p.peso_maximo * ratio
            peso_refugio = p.peso_maximo - peso_riesgo
            plan[elegido] = peso_riesgo if peso_riesgo >= p.peso_minimo else 0.0
            plan[REFUGIO] = peso_refugio if peso_refugio >= p.peso_minimo else 0.0
            self.activo_actual = elegido
        else:
            plan[REFUGIO] = p.peso_maximo
            self.activo_actual = None
            self.nivel_actual = None
        return plan

    def _debe_operar(self, s, w_obj, total, i):
        invertido = self.shares[s] > 0
        if w_obj == 0:
            return invertido
        if not invertido:
            return True
        w_act = self._valor_pos(s, i) / total
        return abs(w_obj - w_act) > self.p.banda

    # ------------------------- ejecucion ------------------------- #
    def _vender(self, plan, operar, total, i):
        for s in operar:
            w_obj = plan[s]
            precio = self.O[s][i]
            if precio <= 0:
                continue
            if w_obj == 0 and self.shares[s] > 0:
                self._ejecutar(s, -self.shares[s], i)
                continue
            exceso = self._valor_pos(s, i) - w_obj * total
            if exceso > 0:
                qty = int(exceso / precio)
                if qty > 0 and qty * precio >= self.p.min_orden_usd:
                    self._ejecutar(s, -qty, i)

    def _comprar(self, plan, operar, i):
        total = self._valor_total(i)
        caja = self.cash * 0.98
        for s in ACTIVOS_RIESGO + [REFUGIO]:
            if s not in operar:
                continue
            w_obj = plan[s]
            precio = self.O[s][i]
            if w_obj <= 0 or precio <= 0:
                continue
            falta = w_obj * total - self._valor_pos(s, i)
            if falta <= 0:
                continue
            monto = min(falta, caja)
            qty = int(monto / precio)
            if qty > 0 and qty * precio >= self.p.min_orden_usd:
                antes = self.cash
                self._ejecutar(s, qty, i)
                caja -= antes - self.cash

    # ------------------------- bucle principal ------------------------- #
    def run(self):
        p = self.p
        self._reset()
        fechas_out, equity, expos = [], [], []

        for i in range(self.i0, len(self.fechas)):
            fecha = self.fechas[i]
            prev = i - 1

            if self.pend_compras:
                self._comprar(self.buy_plan, self.buy_operar, i)
                self.pend_compras = False

            total = self._valor_total(i)
            if total > 0:
                spy_prev = self.C[MERCADO][prev]
                urgente = (
                    self.activo_actual is not None
                    and spy_prev < self.s50[prev] * (1 - p.colchon_regimen)
                )
                semana = tuple(fecha.isocalendar())[:2]
                evaluar = (
                    (not p.eval_semanal)
                    or self.plan_vigente is None
                    or semana != self.semana_ultima
                    or urgente
                )

                plan = self.plan_vigente
                if evaluar:
                    nuevo = self._calcular_plan(prev)
                    if nuevo is not None:
                        plan = nuevo
                        self.plan_vigente = nuevo
                        self.semana_ultima = semana

                if plan is not None:
                    operar = {s for s, w in plan.items() if self._debe_operar(s, w, total, i)}
                    self._vender(plan, operar, total, i)
                    if p.compras_diferidas:
                        self.buy_plan, self.buy_operar, self.pend_compras = plan, operar, True
                    else:
                        self._comprar(plan, operar, i)

            # valoracion al cierre
            vals = {s: self.shares[s] * self.C[s][i] for s in OPERABLES}
            eq = self.cash + sum(vals.values())
            assert self.cash > -1e-6, f"Caja negativa el {fecha}: {self.cash}"
            fechas_out.append(fecha)
            equity.append(eq)
            expos.append(
                {
                    "w_SOXL": vals["SOXL"] / eq,
                    "w_TQQQ": vals["TQQQ"] / eq,
                    "w_SHV": vals["SHV"] / eq,
                    "w_caja": self.cash / eq,
                }
            )

        eq_s = pd.Series(equity, index=pd.DatetimeIndex(fechas_out), name="equity")
        ex_df = pd.DataFrame(expos, index=eq_s.index)
        tr_df = pd.DataFrame(self.trades)
        return eq_s, ex_df, tr_df


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def metricas(eq, capital, rf=0.0):
    r = eq.pct_change()
    r.iloc[0] = eq.iloc[0] / capital - 1
    anios = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    ret_total = eq.iloc[-1] / capital - 1
    cagr = (eq.iloc[-1] / capital) ** (1 / anios) - 1
    vol = r.std() * np.sqrt(252)
    exceso = r - rf / 252
    sharpe = exceso.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(exceso, 0) ** 2)) * np.sqrt(252)
    sortino = (r.mean() * 252 - rf) / downside if downside > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "retorno_total": ret_total,
        "cagr": cagr,
        "vol_anual": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": maxdd,
        "calmar": calmar,
        "equity_final": eq.iloc[-1],
    }


def retornos_anuales(eq, capital):
    fin_anio = eq.groupby(eq.index.year).last()
    previo = fin_anio.shift(1)
    previo.iloc[0] = capital
    return fin_anio / previo - 1


def estadisticas_corrida(p, eq, ex, tr, rf):
    m = metricas(eq, p.capital, rf)
    fees = float(tr["comision"].sum()) if len(tr) else 0.0
    ganancia_neta = eq.iloc[-1] - p.capital
    bruta = ganancia_neta + fees
    n = len(tr)
    volumen = float(tr["valor"].sum()) if n else 0.0
    m.update(
        {
            "fees": fees,
            "fees_sobre_bruta": fees / bruta if bruta > 0 else np.nan,
            "fees_sobre_capital": fees / p.capital,
            "fills": n,
            "orden_media": volumen / n if n else 0.0,
            "volumen_sobre_capital": volumen / p.capital,
            "w_riesgo": float((ex["w_SOXL"] + ex["w_TQQQ"]).mean()),
            "w_SHV": float(ex["w_SHV"].mean()),
            "w_caja": float(ex["w_caja"].mean()),
            "w_SOXL": float(ex["w_SOXL"].mean()),
            "w_TQQQ": float(ex["w_TQQQ"].mean()),
            "dias_en_riesgo": float(((ex["w_SOXL"] + ex["w_TQQQ"]) > 0.05).mean()),
        }
    )
    return m


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def pct(x, d=1):
    return "n/d" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/d" if pd.isna(x) else f"{x:.{d}f}"


def imprimir_reporte(nombre, p, m, eq, closes, opens, i0, rf):
    print("=" * 78)
    print(f"RESULTADOS - {nombre}")
    print(f"Periodo: {eq.index[0].date()} a {eq.index[-1].date()} | Capital inicial: ${p.capital:,.2f}")
    print("=" * 78)
    print(f"Equity final ............ ${m['equity_final']:,.2f}")
    print(f"Retorno total ........... {pct(m['retorno_total'])}   (CAGR {pct(m['cagr'])})")
    print(f"Volatilidad anual ....... {pct(m['vol_anual'])}")
    print(f"Sharpe (rf={rf:.1%}) ...... {num(m['sharpe'])}    Sortino: {num(m['sortino'])}")
    print(f"Max drawdown ............ {pct(m['max_dd'])}   Calmar: {num(m['calmar'])}")
    print("-" * 78)
    print(f"Comisiones .............. ${m['fees']:,.2f}  ({pct(m['fees_sobre_capital'], 0)} del capital, "
          f"{pct(m['fees_sobre_bruta'], 0)} de la ganancia bruta)")
    print(f"Fills ................... {m['fills']}   | Orden media: ${m['orden_media']:,.2f} "
          f"| Volumen/capital: {m['volumen_sobre_capital']:.0f}x")
    print(f"Exposicion media ........ riesgo {pct(m['w_riesgo'])} (SOXL {pct(m['w_SOXL'])}, "
          f"TQQQ {pct(m['w_TQQQ'])}) | SHV {pct(m['w_SHV'])} | caja {pct(m['w_caja'])}")
    print(f"Dias con riesgo > 5% .... {pct(m['dias_en_riesgo'])}")

    # Benchmarks buy & hold (fraccionales, para referencia)
    print("-" * 78)
    filas = []
    bench = {}
    for t in [MERCADO, "TQQQ"]:
        serie = p.capital * closes[t].iloc[i0:] / opens[t].iloc[i0]
        bench[t] = serie
        mb = metricas(serie, p.capital, rf)
        filas.append(
            {
                "Serie": f"Buy&Hold {t}",
                "Retorno": pct(mb["retorno_total"]),
                "CAGR": pct(mb["cagr"]),
                "Sharpe": num(mb["sharpe"]),
                "MaxDD": pct(mb["max_dd"]),
            }
        )
    filas.insert(
        0,
        {
            "Serie": nombre,
            "Retorno": pct(m["retorno_total"]),
            "CAGR": pct(m["cagr"]),
            "Sharpe": num(m["sharpe"]),
            "MaxDD": pct(m["max_dd"]),
        },
    )
    print(pd.DataFrame(filas).to_string(index=False))

    # Retornos por anio
    print("-" * 78)
    anual = pd.DataFrame(
        {
            nombre: retornos_anuales(eq, p.capital),
            "SPY B&H": retornos_anuales(bench[MERCADO], p.capital),
            "TQQQ B&H": retornos_anuales(bench["TQQQ"], p.capital),
        }
    )
    print("Retorno por anio:")
    print(anual.map(lambda x: pct(x)).to_string())
    print("=" * 78)
    return bench


def graficar(ruta, nombre, eq, ex, bench):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no esta instalado: se omite la grafica)")
        return
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.5]})
    ax[0].plot(eq.index, eq.values, label=nombre, lw=1.8)
    ax[0].plot(bench[MERCADO].index, bench[MERCADO].values, label="SPY B&H", lw=1.0)
    ax[0].plot(bench["TQQQ"].index, bench["TQQQ"].values, label="TQQQ B&H", lw=0.8, alpha=0.7)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Equity (log)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    dd = eq / eq.cummax() - 1
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="tab:red", alpha=0.4)
    ax[1].set_ylabel("Drawdown %")
    ax[1].grid(alpha=0.3)
    ax[2].stackplot(
        ex.index,
        ex["w_SOXL"] * 100, ex["w_TQQQ"] * 100, ex["w_SHV"] * 100, ex["w_caja"] * 100,
        labels=["SOXL", "TQQQ", "SHV", "Caja"],
    )
    ax[2].set_ylabel("Exposicion %")
    ax[2].legend(loc="upper left", ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"Grafica guardada en: {ruta}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def construir_parser():
    ap = argparse.ArgumentParser(description="Backtest local v3 (Dual Momentum + Vol Targeting)")
    ap.add_argument("--inicio", default="2020-01-01")
    ap.add_argument("--fin", default="2026-09-30")
    ap.add_argument("--rf", type=float, default=0.0, help="tasa libre de riesgo anual para Sharpe/Sortino")
    ap.add_argument("--salida", default="resultados_local", help="carpeta para CSV y grafica")
    ap.add_argument("--ablacion", action="store_true", help="corre A, B, C, D, E y compara")
    ap.add_argument("--demo", action="store_true", help="datos sinteticos (solo para probar el script)")
    ap.add_argument("--sin-grafica", action="store_true")
    d = Params()
    ap.add_argument("--capital", type=float, default=d.capital)
    ap.add_argument("--vol-target", type=float, default=d.vol_target)
    ap.add_argument("--lookback-mom", type=int, default=d.lookback_mom)
    ap.add_argument("--lookback-vol", type=int, default=d.lookback_vol)
    ap.add_argument("--peso-maximo", type=float, default=d.peso_maximo)
    ap.add_argument("--peso-minimo", type=float, default=d.peso_minimo)
    ap.add_argument("--banda", type=float, default=d.banda)
    ap.add_argument("--margen-cambio", type=float, default=d.margen_cambio)
    ap.add_argument("--min-orden-usd", type=float, default=d.min_orden_usd)
    ap.add_argument("--eval-semanal", type=int, default=d.eval_semanal)
    ap.add_argument("--niveles-discretos", type=int, default=d.niveles_discretos)
    ap.add_argument("--hist-nivel", type=float, default=d.hist_nivel)
    ap.add_argument("--colchon-regimen", type=float, default=d.colchon_regimen)
    ap.add_argument("--slippage-bps", type=float, default=d.slippage_bps)
    ap.add_argument("--compras-diferidas", type=int, default=d.compras_diferidas)
    return ap


def params_desde_args(a):
    return Params(
        capital=a.capital, vol_target=a.vol_target, lookback_mom=a.lookback_mom,
        lookback_vol=a.lookback_vol, peso_maximo=a.peso_maximo, peso_minimo=a.peso_minimo,
        banda=a.banda, margen_cambio=a.margen_cambio, min_orden_usd=a.min_orden_usd,
        eval_semanal=a.eval_semanal, niveles_discretos=a.niveles_discretos,
        hist_nivel=a.hist_nivel, colchon_regimen=a.colchon_regimen,
        slippage_bps=a.slippage_bps, compras_diferidas=a.compras_diferidas,
    )


def main():
    a = construir_parser().parse_args()
    base = params_desde_args(a)
    os.makedirs(a.salida, exist_ok=True)

    print(f"Descargando datos {'(DEMO sintetico)' if a.demo else '(yfinance)'} ...")
    opens, closes = descargar(a.inicio, a.fin, demo=a.demo)
    print(f"Datos: {closes.index[0].date()} a {closes.index[-1].date()} ({len(closes)} dias)")
    if a.demo:
        print("*** MODO DEMO: datos sinteticos, los numeros NO significan nada ***")

    if a.ablacion:
        corridas = {
            "A  v3 completo": base,
            "B  sin eval semanal": replace(base, eval_semanal=0),
            "C  sin niveles": replace(base, niveles_discretos=0),
            "D  tipo v2": replace(base, eval_semanal=0, niveles_discretos=0, banda=0.10,
                                  min_orden_usd=25.0, margen_cambio=0.02),
            "E  A + compras dia sig.": replace(base, compras_diferidas=1),
        }
    else:
        corridas = {"v3 (corrida A)": base}

    filas = []
    primera = None
    for nombre, p in corridas.items():
        sim = Simulador(p, opens, closes, a.inicio)
        eq, ex, tr = sim.run()
        m = estadisticas_corrida(p, eq, ex, tr, a.rf)
        filas.append(
            {
                "Corrida": nombre,
                "Return": pct(m["retorno_total"]),
                "CAGR": pct(m["cagr"]),
                "Sharpe": num(m["sharpe"]),
                "Sortino": num(m["sortino"]),
                "MaxDD": pct(m["max_dd"]),
                "Fees $": f"{m['fees']:.0f}",
                "Fees/bruta": pct(m["fees_sobre_bruta"], 0),
                "Fills": m["fills"],
                "Orden $": f"{m['orden_media']:.0f}",
                "Vol/cap": f"{m['volumen_sobre_capital']:.0f}x",
                "Riesgo": pct(m["w_riesgo"], 0),
                "Caja": pct(m["w_caja"], 0),
            }
        )
        if primera is None:
            primera = (nombre, p, m, eq, ex, tr, sim)

    nombre, p, m, eq, ex, tr, sim = primera
    bench = imprimir_reporte(nombre, p, m, eq, closes, opens, sim.i0, a.rf)

    if a.ablacion:
        print("\nCOMPARACION DE CORRIDAS (ablacion)")
        print(pd.DataFrame(filas).to_string(index=False))
        print("\nLectura rapida:")
        print("  B vs A -> cuanto aporta evaluar semanalmente")
        print("  C vs A -> cuanto aportan los niveles de exposicion")
        print("  D      -> debe parecerse a v2 (control)")
        print("  E vs A -> sensibilidad a que las compras se ejecuten un dia despues")

    # Archivos de salida (corrida A)
    tag = "demo" if a.demo else "yf"
    salida_eq = pd.concat(
        [eq, ex, bench[MERCADO].rename("SPY_BH"), bench["TQQQ"].rename("TQQQ_BH")], axis=1
    )
    salida_eq.to_csv(os.path.join(a.salida, f"equity_{tag}.csv"), index_label="fecha")
    if len(tr):
        tr.to_csv(os.path.join(a.salida, f"trades_{tag}.csv"), index=False)
    print(f"CSV guardados en: {a.salida}/")
    if not a.sin_grafica:
        graficar(os.path.join(a.salida, f"grafica_{tag}.png"), nombre, eq, ex, bench)


if __name__ == "__main__":
    main()