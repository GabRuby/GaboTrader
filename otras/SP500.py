# region imports
from AlgorithmImports import *
# endregion

class EscaleraSP500BolaDeNieve3Anos(QCAlgorithm):

    def initialize(self):
        # 1. PERÍODO A 3 AÑOS (Septiembre 2023 - Septiembre 2026)
        self.set_start_date(2023, 9, 30)
        self.set_end_date(2026, 9, 30)

        # 2. Capital inicial
        self.set_cash(100000)

        # 3. ETF S&P 500 (SPY)
        self.equity = self.add_equity("SPY", Resolution.MINUTE)
        self.symbol = self.equity.symbol

        # Modelo de comisiones de broker real
        self.equity.set_fee_model(InteractiveBrokersFeeModel())

        # 4. Indicador de Tendencia (SMA 200 diaria) + Warm-Up
        self.sma_200 = self.sma(self.symbol, 200, Resolution.DAILY)
        self.set_warm_up(200, Resolution.DAILY)

        # Variables de control
        self.ultimo_precio_compra = 0
        self.conteo_escalones = 0
        self.max_escalones = 5

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return

        if not self.sma_200.is_ready or not data.contains_key(self.symbol) or data[self.symbol] is None:
            return

        precio_actual = data[self.symbol].close
        tendencia_alcista = precio_actual > self.sma_200.current.value

        # PARÁMETROS ADAPTADOS PARA EL S&P 500
        if tendencia_alcista:
            pct_capital_escalon = 0.30  # Reinvierte el 30% del capital acumulado por escalón
            distancia_escalon = 0.008   # Comprar cada caída del 0.8%
            objetivo_ganancia = 0.015   # Salir con +1.5% de beneficio
        else:
            pct_capital_escalon = 0.10  # En tendencia bajista reduce al 10%
            distancia_escalon = 0.015   # Caída del 1.5%
            objetivo_ganancia = 0.010   # Salir rápido con +1.0%

        # COMPRA INICIAL (EFECTO BOLA DE NIEVE: Reinvierte el capital total disponible)
        if not self.portfolio.invested:
            capital_disponible = self.portfolio.cash
            monto = capital_disponible * pct_capital_escalon
            
            if monto > 0:
                cantidad = int(monto / precio_actual)
                if cantidad > 0:
                    self.market_order(self.symbol, cantidad)
                    self.ultimo_precio_compra = precio_actual
                    self.conteo_escalones = 1
                    # Registramos el capital al momento de comprar para medir el crecimiento de la bola
                    self.log(f"--- NUEVO CICLO BOLA DE NIEVE --- Capital disponible: ${capital_disponible:,.2f} | Acciones compradas: {cantidad}")

        # LÓGICA DE ESCALONES Y VENTAS
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

            # TOMA DE GANANCIAS (CIERRE Y LIBERACIÓN DE CAPITAL CON BOLA DE NIEVE)
            elif precio_actual >= precio_promedio * (1 + objetivo_ganancia):
                self.liquidate(self.symbol)
                self.ultimo_precio_compra = 0
                self.conteo_escalones = 0
                self.log(f"¡VENTA BOLA DE NIEVE EXITOSA! Capital resultante en caja: ${self.portfolio.cash:,.2f}")