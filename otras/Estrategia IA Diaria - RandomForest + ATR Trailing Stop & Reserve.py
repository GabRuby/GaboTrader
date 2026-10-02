# region imports
from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# endregion

class IAOptimizadaDiariaDefinitiva(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 9, 30)
        self.set_end_date(2026, 9, 30)
        self.set_cash(500) # Capital inicial en USD

        # Activos 3x apalancados
        self.activos = ["SOXL", "TQQQ"]
        self.simbolos = []

        for ticker in self.activos:
            eq = self.add_equity(ticker, Resolution.DAILY)
            eq.set_fee_model(InteractiveBrokersFeeModel())
            self.simbolos.append(eq.symbol)

        # Filtro de Tendencia Macro
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.sma_spy = self.sma(self.spy, 50, Resolution.DAILY)

        # Modelo de IA
        self.modelo_ia = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.entrenado = False

        self.set_warm_up(60, Resolution.DAILY)
        self.posicion_actual = None
        self.precio_entrada = 0
        self.stop_loss_precio = 0

        # Reglas de Reserva e Hitos Progresivos
        self.hito_1_alcanzado = False
        self.hito_2_alcanzado = False
        self.reserva_fija = 0.0

        # Ejecución diaria 30 min después del cierre de apertura
        self.schedule.on(self.date_rules.every_day(), self.time_rules.after_market_open("SPY", 30), self.procesar_mercado)

    def procesar_mercado(self):
        if self.is_warming_up or not self.sma_spy.is_ready:
            return

        equity_actual = self.portfolio.total_portfolio_value

        # 1. PROTECCIÓN Y RESERVA PROGRESIVA
        if equity_actual >= 750.0 and not self.hito_1_alcanzado:
            self.hito_1_alcanzado = True
            self.reserva_fija = equity_actual * 0.35
            self.log(f"[HITO 1 - $750] Equity: ${equity_actual:.2f}. Reserva asegurada: ${self.reserva_fija:.2f}")

        if equity_actual >= 900.0 and not self.hito_2_alcanzado:
            self.hito_2_alcanzado = True
            self.reserva_fija = equity_actual * 0.50
            self.log(f"[HITO 2 - $900] Equity: ${equity_actual:.2f}. Reserva incrementada a 50%: ${self.reserva_fija:.2f}")

        capital_operable = equity_actual - self.reserva_fija if self.reserva_fija > 0 else equity_actual

        # Re-entrenamiento mensual
        if not self.entrenado or self.time.day == 1:
            self.entrenar_modelo_ia()

        # 2. GESTIÓN DE POSICIÓN ABIERTA CON TRAILING STOP MEJORADO
        if self.portfolio.invested:
            s = self.posicion_actual
            if s is None or not self.securities.contains_key(s):
                return

            precio_actual = self.securities[s].price
            history = self.history(s, 14, Resolution.DAILY)
            if history.empty: return
            atr_val = (history['high'] - history['low']).mean()

            # Mover a Breakeven en +1.2 ATR
            if precio_actual >= self.precio_entrada + (1.2 * atr_val):
                nuevo_stop = max(self.stop_loss_precio, self.precio_entrada)
                if nuevo_stop > self.stop_loss_precio:
                    self.stop_loss_precio = nuevo_stop

            # Trailing Stop Ajustado en ganancias (+2.0 ATR)
            if precio_actual >= self.precio_entrada + (2.0 * atr_val):
                stop_dinamico = precio_actual - (1.2 * atr_val)
                if stop_dinamico > self.stop_loss_precio:
                    self.stop_loss_precio = stop_dinamico

            # Cierre por Stop Loss / Trailing Stop
            if precio_actual <= self.stop_loss_precio:
                self.liquidate(s)
                self.log(f"[{s.value}] SALIDA CONSERVADORA a ${precio_actual:.2f}")
                self.posicion_actual = None

            # Take Profit en +3.5 ATR
            elif precio_actual >= self.precio_entrada + (3.5 * atr_val):
                self.liquidate(s)
                self.log(f"[{s.value}] TAKE PROFIT ALCANZADO a ${precio_actual:.2f}")
                self.posicion_actual = None
            return

        # 3. FILTRO MACRO Y ENTRADAS IA
        precio_spy = self.securities[self.spy].price
        if precio_spy < self.sma_spy.current.value:
            return

        if capital_operable < 50:
            return

        mejores_candidatos = []

        for s in self.simbolos:
            history = self.history(s, 40, Resolution.DAILY)
            if len(history) < 25: continue

            features_actuales = self.extraer_caracteristicas(history)
            if features_actuales is None: continue

            probabilidades = self.modelo_ia.predict_proba([features_actuales])[0]
            prob_subida = probabilidades[1] if len(probabilidades) > 1 else 0

            if prob_subida >= 0.60:
                precio = self.securities[s].price
                atr_val = (history['high'] - history['low']).mean()
                mejores_candidatos.append((s, precio, atr_val, prob_subida))

        if mejores_candidatos:
            mejores_candidatos.sort(key=lambda x: x[3], reverse=True)
            s_elegido, precio_p, atr_p, prob = mejores_candidatos[0]

            monto_a_invertir = capital_operable * 0.95
            cantidad = int(monto_a_invertir / precio_p)

            if cantidad > 0:
                self.market_order(s_elegido, cantidad)
                self.posicion_actual = s_elegido
                self.precio_entrada = precio_p
                self.stop_loss_precio = precio_p - (1.5 * atr_p)
                self.log(f"[ENTRADA IA] {s_elegido.value} | Invertido: ${monto_a_invertir:.2f} USD | Prob: {prob*100:.1f}%")

    def entrenar_modelo_ia(self):
        X, y = [], []

        for s in self.simbolos:
            hist = self.history(s, 150, Resolution.DAILY)
            if len(hist) < 60: continue

            hist = hist.reset_index(level=0, drop=True)
            hist['ret_1d'] = hist['close'].pct_change(1)
            hist['ret_5d'] = hist['close'].pct_change(5)
            hist['volatilidad'] = (hist['high'] - hist['low']) / hist['close']
            hist['target'] = (hist['close'].shift(-3) > hist['close']).astype(int)
            hist = hist.dropna()

            for i in range(len(hist) - 3):
                row = hist.iloc[i]
                X.append([row['ret_1d'], row['ret_5d'], row['volatilidad']])
                y.append(row['target'])

        if len(X) > 30:
            self.modelo_ia.fit(X, y)
            self.entrenado = True

    def extraer_caracteristicas(self, history):
        hist = history.reset_index(level=0, drop=True)
        ret_1d = (hist['close'].iloc[-1] - hist['close'].iloc[-2]) / hist['close'].iloc[-2]
        ret_5d = (hist['close'].iloc[-1] - hist['close'].iloc[-6]) / hist['close'].iloc[-6]
        volatilidad = (hist['high'].iloc[-1] - hist['low'].iloc[-1]) / hist['close'].iloc[-1]
        return [ret_1d, ret_5d, volatilidad]