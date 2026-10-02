# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class DualMomentumCoreV3(QCAlgorithm):
    """
    PASO 1b - NUCLEO v3: misma logica que v2, pero enfocada en REDUCIR TURNOVER.

    Diagnostico de v2: comisiones = ~70% del capital inicial y ~42% de la ganancia bruta.

    Cambios vs. v2:
      1. Evaluacion SEMANAL (primer dia habil de cada semana) en vez de diaria.
         Excepcion: salida urgente si SPY cae mas de `colchon_regimen` bajo su SMA50.
      2. Exposicion por NIVELES (0.33 / 0.66 / 1.0) en vez de escalar de forma continua,
         con histeresis para no saltar entre niveles.
      3. Banda de rebalanceo mas ancha (15%) y orden minima mas alta ($50).
      4. Histeresis de cambio SOXL <-> TQQQ mas alta (5 puntos de momentum).
      5. Reintento de fills: los dias sin evaluacion se sincroniza contra el ULTIMO plan
         (sin recalcular momentum), asi una compra parcial no queda colgada una semana.
      6. Diagnostico: log de caja/pesos tras cada fase, log de cada fill, grafica de caja %
         y resumen final de costos (comision por fill, orden promedio, fees / capital).

    Parametros para pruebas A/B (Paso 4):
      eval_semanal=0      -> evalua a diario (como v2)
      niveles_discretos=0 -> escala de volatilidad continua (como v2)
      banda, min_orden_usd, margen_cambio, vol_target, lookback_mom, lookback_vol, capital
    """

    # ------------------------------------------------------------------ #
    # Utilidad: leer parametros del backtest con valor por defecto
    # ------------------------------------------------------------------ #
    def _param(self, nombre, defecto):
        valor = self.get_parameter(nombre)
        return type(defecto)(valor) if valor else defecto

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2026, 9, 30)
        self.set_cash(self._param("capital", 500))
        self.capital_inicial = float(self.portfolio.cash)

        # ---------------- Universo ---------------- #
        self.activos_riesgo = ["SOXL", "TQQQ"]
        self.refugio = "SHV"

        self.simbolos_riesgo = [self._agregar_equity(t) for t in self.activos_riesgo]
        self.simbolo_refugio = self._agregar_equity(self.refugio)
        self.todos = self.simbolos_riesgo + [self.simbolo_refugio]

        # Filtro macro
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.sma50 = self.sma(self.spy, 50, Resolution.DAILY)
        self.sma200 = self.sma(self.spy, 200, Resolution.DAILY)
        self.set_benchmark(self.spy)

        # ---------------- Parametros cuantitativos ---------------- #
        self.vol_target = self._param("vol_target", 0.35)
        self.lookback_mom = self._param("lookback_mom", 60)
        self.lookback_vol = self._param("lookback_vol", 20)
        self.peso_maximo = self._param("peso_maximo", 0.95)
        self.peso_minimo = self._param("peso_minimo", 0.05)

        # ---------------- Parametros anti-turnover (nuevos en v3) ---------------- #
        self.banda = self._param("banda", 0.15)
        self.margen_cambio = self._param("margen_cambio", 0.05)
        self.min_orden_usd = self._param("min_orden_usd", 50.0)
        self.eval_semanal = self._param("eval_semanal", 1)
        self.niveles_discretos = self._param("niveles_discretos", 1)
        self.hist_nivel = self._param("hist_nivel", 0.05)
        self.colchon_regimen = self._param("colchon_regimen", 0.01)

        # Niveles de exposicion (fraccion del peso maximo). Piso de 0.33: nunca menos de eso en riesgo.
        self.niveles = [0.33, 0.66, 1.0]
        self.semi_paso = 0.165  # mitad del espacio entre niveles

        # ---------------- Estado ---------------- #
        self.activo_actual = None
        self.nivel_actual = None
        self.plan = None            # plan del dia (para la fase de compras)
        self.plan_vigente = None    # ultimo plan calculado (se reutiliza los dias sin evaluacion)
        self.operar = set()
        self.semana_ultima = None

        # Contadores de diagnostico
        self.num_fills = 0
        self.volumen_usd = 0.0

        self.set_warm_up(210, Resolution.DAILY)

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
        eq.set_slippage_model(ConstantSlippageModel(0.0005))  # 5 bps
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

    def _nivel(self, ratio):
        """Cuantiza el escalador de volatilidad al nivel mas cercano, con histeresis."""
        candidato = min(self.niveles, key=lambda n: abs(n - ratio))
        actual = self.nivel_actual
        if actual is not None and abs(ratio - actual) <= self.semi_paso + self.hist_nivel:
            return actual
        return candidato

    # ------------------------------------------------------------------ #
    # Decision: plan de pesos objetivo
    # ------------------------------------------------------------------ #
    def calcular_plan(self):
        precio_spy = self.securities[self.spy].price
        regimen_ok = (
            precio_spy > self.sma200.current.value
            and precio_spy > self.sma50.current.value
        )

        datos = {}
        for s in self.simbolos_riesgo:
            m = self._metricas(s)
            if m is not None:
                datos[s] = m
        if not datos:
            return None

        puntajes = {s: d[0] for s, d in datos.items()}
        mejor = max(puntajes, key=puntajes.get)

        # Histeresis de cambio de activo
        elegido = mejor
        actual = self.activo_actual
        if actual in puntajes and mejor != actual:
            if puntajes[mejor] - puntajes[actual] < self.margen_cambio:
                elegido = actual

        plan = {s: 0.0 for s in self.todos}

        if regimen_ok and puntajes[elegido] > 0:
            vol = datos[elegido][1]
            ratio = min(1.0, self.vol_target / vol) if vol > 0 else 1.0
            if self.niveles_discretos:
                ratio = self._nivel(ratio)
                self.nivel_actual = ratio

            peso_riesgo = self.peso_maximo * ratio
            peso_refugio = self.peso_maximo - peso_riesgo

            plan[elegido] = peso_riesgo if peso_riesgo >= self.peso_minimo else 0.0
            plan[self.simbolo_refugio] = peso_refugio if peso_refugio >= self.peso_minimo else 0.0
            self.activo_actual = elegido
        else:
            plan[self.simbolo_refugio] = self.peso_maximo
            self.activo_actual = None
            self.nivel_actual = None

        return plan

    def _salida_urgente(self, precio_spy):
        """Salida fuera del calendario semanal: SPY rompe su SMA50 por mas del colchon."""
        return (
            self.activo_actual is not None
            and precio_spy < self.sma50.current.value * (1 - self.colchon_regimen)
        )

    def _debe_operar(self, simbolo, w_obj, total):
        """Banda de rebalanceo: entra/sale siempre; ajusta solo si la desviacion supera la banda."""
        posicion = self.portfolio[simbolo]
        if w_obj == 0:
            return posicion.invested
        if not posicion.invested:
            return True
        w_act = posicion.holdings_value / total
        return abs(w_obj - w_act) > self.banda

    # ------------------------------------------------------------------ #
    # Fase 1: decidir (o reutilizar plan) y VENDER
    # ------------------------------------------------------------------ #
    def fase_ventas(self):
        self.plan = None
        self.operar = set()

        if self.is_warming_up or not (self.sma50.is_ready and self.sma200.is_ready):
            return

        total = self.portfolio.total_portfolio_value
        if total <= 0:
            return

        semana_hoy = tuple(self.time.isocalendar())[:2]
        precio_spy = self.securities[self.spy].price
        urgente = self._salida_urgente(precio_spy)

        evaluar = (
            not self.eval_semanal
            or self.plan_vigente is None
            or semana_hoy != self.semana_ultima
            or urgente
        )

        if evaluar:
            plan = self.calcular_plan()
            if plan is None:
                return
            self.plan_vigente = plan
            self.semana_ultima = semana_hoy
            motivo = "URGENTE" if urgente else "EVALUACION"
            plan_txt = {s.value: round(w, 2) for s, w in plan.items() if w > 0}
            self.log(f"[{motivo}] plan={plan_txt} | nivel={self.nivel_actual}")
        else:
            plan = self.plan_vigente  # dias intermedios: solo sincroniza contra el plan vigente

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
                continue

            exceso = posicion.holdings_value - w_obj * total
            if exceso > 0:
                qty = int(exceso / precio)
                if qty > 0 and qty * precio >= self.min_orden_usd:
                    self.market_order(s, -qty)

    # ------------------------------------------------------------------ #
    # Fase 2: COMPRAR (riesgo primero, luego refugio)
    # ------------------------------------------------------------------ #
    def fase_compras(self):
        if self.plan is None or not self.operar:
            return

        total = self.portfolio.total_portfolio_value
        caja = self.portfolio.cash * 0.98

        for s in self.simbolos_riesgo + [self.simbolo_refugio]:
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

        self._log_estado("TRAS-COMPRAS")
        self.plan = None
        self.operar = set()

    # ------------------------------------------------------------------ #
    # Diagnostico
    # ------------------------------------------------------------------ #
    def _log_estado(self, etiqueta):
        total = self.portfolio.total_portfolio_value
        if total <= 0:
            return
        pesos = {
            s.value: round(self.portfolio[s].holdings_value / total, 3)
            for s in self.todos
            if self.portfolio[s].invested
        }
        self.log(
            f"[{etiqueta}] equity=${total:,.2f} | caja={self.portfolio.cash / total:.1%} | pesos={pesos}"
        )

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return
        total = self.portfolio.total_portfolio_value
        if total > 0:
            self.plot("Diagnostico", "Caja %", float(self.portfolio.cash / total * 100))
            self.plot("Diagnostico", "Invertido %", float(self.portfolio.total_holdings_value / total * 100))

    def on_order_event(self, order_event: OrderEvent):
        if order_event.status == OrderStatus.FILLED:
            self.num_fills += 1
            monto = abs(order_event.fill_quantity * order_event.fill_price)
            self.volumen_usd += monto
            self.log(
                f"[FILL] {order_event.symbol.value} {order_event.fill_quantity:+.0f} @ "
                f"{order_event.fill_price:.2f} | fee ${order_event.order_fee.value.amount:.2f}"
            )

    def on_end_of_algorithm(self):
        fees = float(self.portfolio.total_fees)
        equity = float(self.portfolio.total_portfolio_value)
        fills = max(self.num_fills, 1)
        self.log(
            f"RESUMEN | Equity final: ${equity:,.2f} | Fills: {self.num_fills} | "
            f"Comisiones: ${fees:,.2f} ({fees / self.capital_inicial:.0%} del capital inicial) | "
            f"Comision media/fill: ${fees / fills:.2f} | Orden media: ${self.volumen_usd / fills:,.2f} | "
            f"Volumen / capital: {self.volumen_usd / self.capital_inicial:.0f}x"
        )