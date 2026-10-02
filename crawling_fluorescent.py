# Estrategia Crawling Fluorescent
import pandas as pd
import numpy as np

def run_strategy(data, fast_ma=12, slow_ma=26):
    # Lógica de momento / MACD básico
    data['SMA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['Signal'] = np.where(data['SMA_Fast'] > data['SMA_Slow'], 1, 0)
    return data