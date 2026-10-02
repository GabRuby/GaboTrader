# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class DualMomentumVolTargeting(QCAlgorithm):

    def initialize(self):
        # Mismas variables base del backtest anterior para comparación directa
        self.set_start_date(2025, 9, 30)
        self.set_end_date(2026, 9, 30)
        self.set_cash(500) # Capital inicial: $500 USD

        # Universos de Activos
        self.activos_riesgo = ["SOXL", "TQQQ"]
        self.refugio = "SHV" # ETF de Tesoros a Corto Plazo (Rendimiento libre de riesgo)
        
        self.simbolos_riesgo = []
        for ticker in self.activos_riesgo:
            eq = self.add_equity(ticker, Resolution.DAILY)
            eq.set_fee_model(InteractiveBrokersFeeModel())
            self.simbolos_riesgo.append(eq.symbol)

        eq_refugio = self.add_equity(self.refugio, Resolution.DAILY)
        eq_refugio.set_fee_model(InteractiveBrokersFeeModel())
        self.simbolo_refugio = eq_refugio.symbol

        # Filtro Macro (SPY)
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.sma_spy = self.sma(self.spy, 50, Resolution.DAILY)

        # Parámetros Cuantitativos
        self.vol_target = 0.35        # Volatilidad objetivo anualizada (35%)
        self.lookback_momentum = 20   # Ventana para medir momentum (20 días hábiles)
        self.lookback_vol = 20        # Ventana para medir volatilidad histórica

        self.set_warm_up(60, Resolution.DAILY)

        # Ejecución Diaria al Aperturar
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.evaluar_rebalanceo
        )

    def evaluar_rebalanceo(self):
        if self.is_warming_up or not self.sma_spy.is_ready:
            return

        precio_spy = self.securities[self.spy].price
        tendencia_macro_ok = precio_spy > self.sma_spy.current.value

        # 1. MEDIR MOMENTUM RELATIVO (SOXL vs TQQQ)
        puntajes_momentum = {}
        for s in self.simbolos_riesgo:
            hist = self.history(s, self.lookback_momentum + 5, Resolution.DAILY)
            if len(hist) < self.lookback_momentum:
                continue
            hist = hist.reset_index(level=0, drop=True)
            retorno_mom = (hist['close'].iloc[-1] - hist['close'].iloc[-self.lookback_momentum]) / hist['close'].iloc[-self.lookback_momentum]
            puntajes_momentum[s] = retorno_mom

        if not puntajes_momentum:
            return

        # Seleccionar el activo con mayor momentum
        mejor_activo = max(puntajes_momentum, key=puntajes_momentum.get)
        mejor_retorno = puntajes_momentum[mejor_activo]

        # 2. REGRA DE ENTRADA Y CONTROL DE VOLATILIDAD
        if tendencia_macro_ok and mejor_retorno > 0:
            # Calcular Volatilidad Histórica Anualizada del activo ganador
            hist_vol = self.history(mejor_activo, self.lookback_vol + 5, Resolution.DAILY).reset_index(level=0, drop=True)
            retornos_diarios = hist_vol['close'].pct_change().dropna()
            vol_actual = retornos_diarios.std() * np.sqrt(252)

            # Ajustar exposición inversamente a la volatilidad
            escalador_vol = min(1.0, self.vol_target / vol_actual) if vol_actual > 0 else 1.0

            peso_riesgo = 0.95 * escalador_vol
            peso_refugio = 0.95 - peso_riesgo # El remanente pasa a bonos ultra cortos

            self.rebalancear_portafolio(mejor_activo, peso_riesgo, self.simbolo_refugio, peso_refugio)
        else:
            # Mercado Bajista o Momentum Negativo: Refugio 100% en SHV
            self.rebalancear_portafolio(None, 0.0, self.simbolo_refugio, 0.95)

    def rebalancear_portafolio(self, activo_riesgo, peso_riesgo, activo_refugio, peso_refugio):
        # Liquidar el activo de riesgo que ya no encabeza el momentum
        for s in self.simbolos_riesgo:
            if s != activo_riesgo and self.portfolio[s].invested:
                self.liquidate(s)

        # Asignación eficiente de capital
        if activo_riesgo and peso_riesgo > 0.05:
            self.set_holdings(activo_riesgo, peso_riesgo)

        if activo_refugio and peso_refugio > 0.05:
            self.set_holdings(activo_refugio, peso_refugio)
        elif self.portfolio[activo_refugio].invested and peso_refugio <= 0.05:
            self.liquidate(activo_refugio)