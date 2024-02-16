import pandas as pd
import numpy as np


class RSI:

    def __init__(self, data: pd.DataFrame, round: bool = True, period: int = 14):
        self.data = data
        self.round = round
        self.period = period

    
    def tradingview_rsi(self):
        # Generate the delta 
        delta = self.data["close"].diff()
        
        #Zero out all losses and find average gain
        up = delta.copy()
        up[up < 0] = 0  
        up = pd.Series.ewm(up, alpha=1/self.period).mean()

        #Zero out all gains and find average loss 
        delta[delta > 0] = 0
        delta *= -1
        down = pd.Series.ewm(delta, alpha=1/self.period).mean()

        rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))

        return np.round(rsi, 2) if self.round else rsi