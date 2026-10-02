# region imports
from AlgorithmImports import *
# endregion

class EscaleraSOXLConStopLoss(QCAlgorithm):

    def initialize(self):
        # Periodo de prueba
        self.set_start_date(2025, 9, 30)
        self.set_end_date(2026, 9, 30)
        self.set_cash(100000)

        # Activo: SOXL (Semiconductores 3x)
        self.equity = self.add_equity("SOXL", Resolution.MINUTE)
        self.symbol = self.equity.symbol
        self.equity.set_fee_model(InteractiveBrokersFeeModel())

        # Media Movil Rapida (SMA 20)
        self.sma_20 = self.sma(self.symbol, 20, Resolution.DAILY)
        self.set_warm_up(20, Resolution.DAILY)

        # Variables de control
        self.ultimo_precio_compra = 0
        self.conteo_escalones = 0
        self.max_escalones = 4

    def on_data(self, data: Slice):
        if self.is_warming_up or not self.sma_20.is_ready or not data.contains_key(self.symbol) or data[self.symbol] is None:
            return

        precio_actual = data[self.symbol].close
        tendencia_alcista = precio_actual > self.sma_20.current.value

        # PARAMETROS AGRESIVOS + PROTECCION
        pct_capital_escalon = 0.40  # 40% del capital disponible por escalon
        distancia_escalon = 0.012   # Caida del 1.2% para comprar siguiente escalon
        objetivo_ganancia = 0.020   # Toma de ganancia con +2.0%
        pct_stop_loss = 0.040       # STOP LOSS: Cortar perdida si cae 4.0% del precio promedio

        # COMPRA INICIAL (SOLO SI HAY TENDENCIA ALCISTA)
        if not self.portfolio.invested:
            if tendencia_alcista:
                capital_disponible = self.portfolio.cash
                monto = capital_disponible * pct_capital_escalon
                
                if monto > 0:
                    cantidad = int(monto / precio_actual)
                    if cantidad > 0:
                        self.market_order(self.symbol, cantidad)
                        self.ultimo_precio_compra = precio_actual
                        self.conteo_escalones = 1
                        self.log(f"--- ENTRADA ALCISTA --- Compradas {cantidad} acciones a ${precio_actual:.2f}")

        # LOGICA DE GESTION DE POSICION ABIERTA
        else:
            precio_promedio = self.portfolio[self.symbol].average_price

            # 1. REGLA DE STOP LOSS (Corte de perdida)
            if precio_actual <= precio_promedio * (1 - pct_stop_loss):
                self.liquidate(self.symbol)
                self.ultimo_precio_compra = 0
                self.conteo_escalones = 0
                self.log(f"[STOP LOSS ACTIVADO] Posicion cerrada por proteccion a ${precio_actual:.2f}. Capital en caja: ${self.portfolio.cash:,.2f}")

            # 2. SALIDA DE EMERGENCIA POR CAMBIO DE TENDENCIA (BAJISTA)
            elif not tendencia_alcista:
                self.liquidate(self.symbol)
                self.ultimo_precio_compra = 0
                self.conteo_escalones = 0
                self.log(f"[SALIDA POR TENDENCIA BAJISTA] Posicion cerrada a ${precio_actual:.2f}")

            # 3. ESCALON EN BAJA (SI NO HA TOCADO STOP LOSS Y SE MANTIENE ALCISTA)
            elif (precio_actual <= self.ultimo_precio_compra * (1 - distancia_escalon)) and (self.conteo_escalones < self.max_escalones):
                capital_disponible = self.portfolio.cash
                monto = capital_disponible * pct_capital_escalon

                if capital_disponible >= monto and monto > 0:
                    cantidad = int(monto / precio_actual)
                    if cantidad > 0:
                        self.market_order(self.symbol, cantidad)
                        self.ultimo_precio_compra = precio_actual
                        self.conteo_escalones += 1
                        self.log(f"Nuevo escalon {self.conteo_escalones} comprado a ${precio_actual:.2f}")

            # 4. TOMA DE GANANCIAS (BOLA DE NIEVE)
            elif precio_actual >= precio_promedio * (1 + objetivo_ganancia):
                self.liquidate(self.symbol)
                self.ultimo_precio_compra = 0
                self.conteo_escalones = 0
                self.log(f"[TOMA DE GANANCIAS] Venta exitosa a ${precio_actual:.2f}. Capital resultante: ${self.portfolio.cash:,.2f}")