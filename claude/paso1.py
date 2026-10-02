# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class DualMomentumCoreV2(QCAlgorithm):
    """
    PASO 1 - NUCLEO: Dual Momentum + Volatility Targeting (v2)

    Cambios vs. la version original:
      - Banda de rebalanceo: solo opera si el peso se desvia mas de `banda`.
      - Histeresis de cambio de activo: SOXL <-> TQQQ solo si la ventaja supera `margen_cambio`.
      - Ventas ANTES que compras (dos eventos separados) para no quedarse sin buying power.
      - Regimen: SPY > SMA200 (estructural) y SPY > SMA50 (salida rapida).
      - Lookback de momentum mas largo (60 dias) y vol independiente (20 dias).
      - Tamano minimo de orden para que la comision minima de IB no se coma el capital.
      - Slippage modelado, benchmark SPY y resumen de costos al final.
      - Todos los parametros se pueden sobreescribir con get_parameter (util para el Paso 4).
    """

    # ------------------------------------------------------------------ #
    # Utilidad: leer parametros del backtest con valor por defecto
    # ------------------------------------------------------------------ #
    def _param(self, nombre, defecto):
        valor = self.get_parameter(nombre)
        return type(defecto)(valor) if valor else defecto

    def initialize(self):
        # Periodo unificado (incluye el mercado bajista de 2022)
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2026, 9, 30)
        self.set_cash(self._param("capital", 500))

        # ---------------- Universo ---------------- #
        self.activos_riesgo = ["SOXL", "TQQQ"]
        self.refugio = "SHV"  # T-bills de corto plazo

        self.simbolos_riesgo = []
        for ticker in self.activos_riesgo:
            self.simbolos_riesgo.append(self._agregar_equity(ticker))
        self.simbolo_refugio = self._agregar_equity(self.refugio)

        # Filtro macro
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.sma50 = self.sma(self.spy, 50, Resolution.DAILY)
        self.sma200 = self.sma(self.spy, 200, Resolution.DAILY)
        self.set_benchmark(self.spy)

        # ---------------- Parametros cuantitativos ---------------- #
        self.vol_target = self._param("vol_target", 0.35)            # vol anualizada objetivo
        self.lookback_mom = self._param("lookback_mom", 60)          # dias para momentum
        self.lookback_vol = self._param("lookback_vol", 20)          # dias para volatilidad
        self.peso_maximo = self._param("peso_maximo", 0.95)          # 5% de colchon en caja
        self.peso_minimo = self._param("peso_minimo", 0.05)          # por debajo => 0
        self.banda = self._param("banda", 0.10)                      # tolerancia de rebalanceo
        self.margen_cambio = self._param("margen_cambio", 0.02)      # histeresis SOXL<->TQQQ
        self.min_orden_usd = self._param("min_orden_usd", 25.0)      # orden minima en USD

        # ---------------- Estado ---------------- #
        self.activo_actual = None   # activo de riesgo elegido en la ultima evaluacion
        self.plan = None            # {symbol: peso objetivo}
        self.operar = set()         # simbolos que deben operarse hoy
        self.num_fills = 0

        self.set_warm_up(210, Resolution.DAILY)

        # Fase 1 (ventas) y Fase 2 (compras) separadas por 5 minutos
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.fase_ventas,
        )
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 35),
            self.fase_compras,
        )

    def _agregar_equity(self, ticker):
        eq = self.add_equity(ticker, Resolution.DAILY)
        eq.set_fee_model(InteractiveBrokersFeeModel())
        eq.set_slippage_model(ConstantSlippageModel(0.0005))  # 5 bps, conservador para ETFs 3x
        return eq.symbol

    # ------------------------------------------------------------------ #
    # Metricas
    # ------------------------------------------------------------------ #
    def _metricas(self, simbolo):
        """Devuelve (momentum, volatilidad anualizada) o None si faltan datos."""
        n = max(self.lookback_mom, self.lookback_vol) + 1
        hist = self.history(simbolo, n, Resolution.DAILY)
        if hist.empty or len(hist) < n:
            return None

        cierres = hist["close"].values
        momentum = cierres[-1] / cierres[-(self.lookback_mom + 1)] - 1

        c_vol = cierres[-(self.lookback_vol + 1):]
        retornos = np.diff(c_vol) / c_vol[:-1]
        vol = float(np.std(retornos, ddof=1) * np.sqrt(252))
        return momentum, vol

    # ------------------------------------------------------------------ #
    # Decision: calcula el plan de pesos objetivo
    # ------------------------------------------------------------------ #
    def calcular_plan(self):
        # 1. Regimen de mercado
        precio_spy = self.securities[self.spy].price
        regimen_ok = (
            precio_spy > self.sma200.current.value
            and precio_spy > self.sma50.current.value
        )

        # 2. Momentum relativo
        datos = {}
        for s in self.simbolos_riesgo:
            m = self._metricas(s)
            if m is not None:
                datos[s] = m
        if not datos:
            return None

        puntajes = {s: d[0] for s, d in datos.items()}
        mejor = max(puntajes, key=puntajes.get)

        # 3. Histeresis: no cambiar de activo por diferencias pequenas
        elegido = mejor
        actual = self.activo_actual
        if actual in puntajes and mejor != actual:
            if puntajes[mejor] - puntajes[actual] < self.margen_cambio:
                elegido = actual

        # 4. Pesos objetivo
        plan = {s: 0.0 for s in self.simbolos_riesgo}
        plan[self.simbolo_refugio] = 0.0

        if regimen_ok and puntajes[elegido] > 0:
            vol = datos[elegido][1]
            escalador = min(1.0, self.vol_target / vol) if vol > 0 else 1.0
            peso_riesgo = self.peso_maximo * escalador
            peso_refugio = self.peso_maximo - peso_riesgo

            plan[elegido] = peso_riesgo if peso_riesgo >= self.peso_minimo else 0.0
            plan[self.simbolo_refugio] = peso_refugio if peso_refugio >= self.peso_minimo else 0.0
            self.activo_actual = elegido
        else:
            plan[self.simbolo_refugio] = self.peso_maximo
            self.activo_actual = None

        return plan

    def _debe_operar(self, simbolo, w_obj, total):
        """Banda de rebalanceo: entra/sale siempre, ajusta solo si la desviacion supera la banda."""
        posicion = self.portfolio[simbolo]
        if w_obj == 0:
            return posicion.invested
        if not posicion.invested:
            return True
        w_act = posicion.holdings_value / total
        return abs(w_obj - w_act) > self.banda

    # ------------------------------------------------------------------ #
    # Fase 1: decidir y VENDER
    # ------------------------------------------------------------------ #
    def fase_ventas(self):
        self.plan = None
        self.operar = set()

        if self.is_warming_up or not (self.sma50.is_ready and self.sma200.is_ready):
            return

        plan = self.calcular_plan()
        if plan is None:
            return

        total = self.portfolio.total_portfolio_value
        if total <= 0:
            return

        self.plan = plan
        for s, w_obj in plan.items():
            if self._debe_operar(s, w_obj, total):
                self.operar.add(s)

        for s in self.operar:
            w_obj = plan[s]
            posicion = self.portfolio[s]
            precio = self.securities[s].price
            if precio <= 0:
                continue

            if w_obj == 0 and posicion.invested:
                self.liquidate(s)
                self.log(f"[VENTA TOTAL] {s.value}")
                continue

            exceso = posicion.holdings_value - w_obj * total
            if exceso > 0:
                qty = int(exceso / precio)
                if qty > 0 and qty * precio >= self.min_orden_usd:
                    self.market_order(s, -qty)
                    self.log(f"[REDUCE] {s.value} -{qty} (peso obj {w_obj:.2f})")

    # ------------------------------------------------------------------ #
    # Fase 2: COMPRAR (riesgo primero, luego refugio)
    # ------------------------------------------------------------------ #
    def fase_compras(self):
        if self.plan is None or not self.operar:
            return

        total = self.portfolio.total_portfolio_value
        caja = self.portfolio.cash * 0.98  # margen para comisiones/slippage

        orden = self.simbolos_riesgo + [self.simbolo_refugio]
        for s in orden:
            if s not in self.operar:
                continue
            w_obj = self.plan[s]
            precio = self.securities[s].price
            if w_obj <= 0 or precio <= 0:
                continue

            falta = w_obj * total - self.portfolio[s].holdings_value
            if falta <= 0:
                continue

            monto = min(falta, caja)
            qty = int(monto / precio)
            if qty > 0 and qty * precio >= self.min_orden_usd:
                self.market_order(s, qty)
                caja -= qty * precio
                self.log(f"[COMPRA] {s.value} +{qty} (peso obj {w_obj:.2f})")

        self.plan = None
        self.operar = set()

    # ------------------------------------------------------------------ #
    # Seguimiento de costos
    # ------------------------------------------------------------------ #
    def on_order_event(self, order_event: OrderEvent):
        if order_event.status == OrderStatus.FILLED:
            self.num_fills += 1

    def on_end_of_algorithm(self):
        self.log(
            f"RESUMEN | Equity final: ${self.portfolio.total_portfolio_value:,.2f} | "
            f"Fills: {self.num_fills} | Comisiones: ${self.portfolio.total_fees:,.2f}"
        )