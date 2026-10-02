# Estrategia comparable de demostración
# Expone la lógica como run_strategy(data, fast_ma, slow_ma) para que QuantLab
# pueda contrastarla contra la Estrategia Principal en la vista comparativa.
import numpy as np
import pandas as pd


def run_strategy(data, fast_ma=8, slow_ma=21):
    fast, slow = int(fast_ma), int(slow_ma)
    data["SMA_Fast"] = data["Close"].rolling(window=fast).mean()
    data["SMA_Slow"] = data["Close"].rolling(window=slow).mean()

    spread = (data["SMA_Fast"] - data["SMA_Slow"]) / data["SMA_Slow"]
    data["Signal"] = np.where(spread > 0.002, 1, 0)

    # Filtra de volatilidad: solo opera cuando el movimiento reciente es ordenado.
    vol = data["Close"].pct_change().rolling(window=slow).std()
    data.loc[vol > vol.quantile(0.8), "Signal"] = 0

    return data
