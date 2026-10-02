# Estrategia Grid Trading v2
import pandas as pd
import numpy as np

def run_strategy(data, fast_ma=10, slow_ma=50):
    # Lógica de grid trading
    data['SMA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['SMA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['Signal'] = np.where(data['SMA_Fast'] > data['SMA_Slow'], 1, 0)
    return data
