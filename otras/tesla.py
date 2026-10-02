# region imports
from AlgorithmImports import *
# endregion

class EscaleraDinamicaBolaDeNieve(QCAlgorithm):

    def initialize(self):
        # 1. Periodo del backtest
        self.set_start_date(2025, 9, 30)
        self.set_end_date(2026, 9, 30)

        # 2. Capital inicial
        self.set_cash(100000)

        # 3. Configuración del activo TSLA en resolución de Minuto
        self.equity = self.add_equity("TSLA", Resolution.MINUTE)
        self.symbol = self.equity.symbol

        # Modelo de comisiones de broker real
        self.equity.set_fee_model(InteractiveBrokersFeeModel())

        # 4. Indicador de Tendencia (SMA 200) y precarga de datos (Warm-Up)
        self.sma_200 = self.sma(self.symbol, 200, Resolution.DAILY)
        self.set_warm_up(200, Resolution.DAILY)

        # Variables de control
        self.ultimo_precio_compra = 0
        self.conteo_escalones = 0
        self.max_escalones = 5

    def on_data(self, data: Slice):
        # Durante el periodo de precarga (Warm-Up), no ejecutamos órdenes
        if self.is_warming_up:
            return

        if not self.sma_200.is_ready or not data.contains_key(self.symbol) or data[self.symbol] is None:
            return

        precio_actual = data[self.symbol].close
        tendencia_alcista = precio_actual > self.sma_200.current.value

        # PARÁMETROS DINÁMICOS SEGÚN LA TENDENCIA
        if tendencia_alcista:
            pct_capital_escalon = 0.30  # Escalón más grande en tendencia alcista
            distancia_escalon = 0.012
            objetivo_ganancia = 0.025
        else:
            pct_capital_escalon = 0.10  # Escalón más prudente en tendencia bajista
            distancia_escalon = 0.025
            objetivo_ganancia = 0.015

        # LÓGICA DE COMPRA (PRIMER ESCALÓN)
        if not self.portfolio.invested:
            capital_disponible = self.portfolio.cash
            monto = capital_disponible * pct_capital_escalon
            
            if monto > 0:
                cantidad = int(monto / precio_actual)
                if cantidad > 0:
                    self.market_order(self.symbol, cantidad)
                    self.ultimo_precio_compra = precio_actual
                    self.conteo_escalones = 1

        # LÓGICA DE ESCALONADO Y VENTA
        else:
            precio_promedio = self.portfolio[self.symbol].average_price

            # ESCALÓN EN BAJA
            if (precio_actual <= self.ultimo_precio_compra * (1 - distancia_escalon)) and (self.conteo_escalones < self.max_escalones):
                capital_disponible = self.portfolio.cash
                monto = capital_disponible * pct_capital_escalon

                if capital_disponible >= monto and monto > 0:
                    cantidad = int(monto / precio_actual)
                    if cantidad > 0:
                        self.market_order(self.symbol, cantidad)
                        self.ultimo_precio_compra = precio_actual
                        self.conteo_escalones += 1

            # TOMA DE GANANCIAS
            elif precio_actual >= precio_promedio * (1 + objetivo_ganancia):
                self.liquidate(self.symbol)
                self.ultimo_precio_compra = 0
                self.conteo_escalones = 0