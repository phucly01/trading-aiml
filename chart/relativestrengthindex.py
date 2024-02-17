import pandas as pd
import numpy as np


class RSI:

    def __init__(self, data: pd.DataFrame, period: int = 14, smoothK: int = 3, smoothD : int = 3):
        self.data = data 
        self.period = period
        self.smoothK = smoothK
        self.smoothD = smoothD

    
    def tradingview_rsi(self, round: bool = True):
        
        #Equation:  rsi = 100 - (100 / (1 + average_gain/average_loss))
        #More info: https://www.investopedia.com/terms/r/rsi.asp
        
        # Generate the delta 
        delta = self.data["close"].diff()
        
        #Zero out all losses and find average gain
        gain = delta.copy()
        gain[gain < 0] = 0  
        gain = pd.Series.ewm(gain, alpha=1/self.period).mean()

        #Zero out all gains and find average loss 
        delta[delta > 0] = 0
        delta *= -1
        loss = pd.Series.ewm(delta, alpha=1/self.period).mean()

        rsi = np.where(gain == 0, 0, np.where(loss == 0, 100, 100 - (100 / (1 + gain / loss))))

        return np.round(rsi, 2) if round else rsi
    

    def tradingview_stochastic_rsi(self):     
        # Calculate RSI
        rsi = self.tradingview_rsi(False)

        # Calculate StochRSI
        rsi = pd.Series(rsi)
        roll = rsi.rolling(self.period)
        rollmin = roll.min()
        stochrsi  = (rsi - rollmin) / (roll.max() - rollmin)
        stochrsi_K = stochrsi.rolling(self.smoothK).mean()
        stochrsi_D = stochrsi_K.rolling(self.smoothD).mean()

        return round(rsi, 2), round(stochrsi_K * 100, 2), round(stochrsi_D * 100, 2)