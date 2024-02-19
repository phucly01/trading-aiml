import pandas as pd
import numpy as np
import numba as nb


class RSI:

    def __init__(self, data: pd.DataFrame):
        self.data = data  

    def rsi(self, period: int = 14, round: bool = True):
        close = self.data['close'] 
        """
        First formula:
            rsi1 = 100 - (100 / (1 + avggain/avgloss))
            First formula uses data from 0 to period-1
        """
        delta = np.append([0], np.diff(close)) #Delta will be one less element, so insert one to make up
        avggain = np.sum(delta[0:period].clip(min=0))/period
        avgloss = np.sum(delta[0:period].clip(max=0))/period
        
        data = np.empty(delta.shape[0])
        data.fill(np.nan)

        #Define lambda macro just for code simplification
        gain_func = lambda val: val if val > 0 else 0
        loss_func = lambda val: -val if val < 0 else 0

        """
        Second formula:
            rsi2 = 100 - (100 / (1 + (lastgain * (period-1) + currgain) / (lastloss * (period-1) + currloss)))
            Second formula uses data from period to end
        """
        n = period-1
        data[n] = 100 - (100 / (1 + avggain/avgloss)) if avgloss > 0 else 100
        i = period #0 based index
        for val in delta[period:]:   
            avggain = (avggain * n + gain_func(val))/period
            avgloss = (avgloss * n + loss_func(val))/period
            data[i] = 100 - (100 / (1 + avggain/avgloss)) if avgloss > 0 else 100
            i += 1

        return data
    
    

    def tradingview_stochastic_rsi(self, period: int=14, smoothK: int=3, smoothD: int=3): 

        if self.data is None or self.data.empty:
            return None
            
        # Calculate RSI
        rsi = self.rsi(period, False)

        # Calculate StochRSI
        rsi = pd.Series(rsi)
        roll = rsi.rolling(period)
        rollmin = roll.min()
        stochrsi  = (rsi - rollmin) / (roll.max() - rollmin)
        stochrsi_K = stochrsi.rolling(smoothK).mean()
        stochrsi_D = stochrsi_K.rolling(smoothD).mean()

        return round(rsi, 2), round(stochrsi_K * 100, 2), round(stochrsi_D * 100, 2)