
import pandas as pd
import mplfinance as mplf
import matplotlib.pyplot as plot

class HeikinAshi:

    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.data = df

    
    def ha(self):
        """
        Formula:
        ha-close = (open + high + low + close)/4
        ha-open = (prevhaclose + prevhaopen)/2
        ha-high = max(high, ha-open, ha-close)
        ha-low = min(low, ha-open, ha-close)
        """

        open = self.data['open']
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        ha_close = []
        ha_open  = []
        ha_high  = []
        ha_low   = []
        ha_close.append((open[0] + close[0] + high[0] + low[0])/4)
        ha_open.append((open[0] + close[0])/2 ) #There are no previous ha_open and ha_close for first one
        ha_high.append(max(high[0], ha_close, ha_open))
        ha_low.append(min(low[0], ha_close, ha_open))

        for i in range(1, len(self.data)):
            ha_close.append((open[i] + close[i] + high[i] + low[i])/4)
            ha_open.append((ha_open[i-1] + ha_close[i-1])/ 2)
            ha_high.append(max(high[i], ha_close[i], ha_open[i]))
            ha_low.append(min(low[i], ha_close[i], ha_open[i]))

        ha_data = {'date': self.data['date'], 'close': ha_close, 'open': ha_open, 'high': ha_high, 'low': ha_low}
        df = pd.DataFrame(ha_data)
        df.index = pd.DatetimeIndex(df['date'])
        return df
    

    def plot(self, df: pd.DataFrame, title, imagename): 

        # Plot candlestick.
        # Add volume.
        # Add moving averages: 3,6,9.
        # Save graph to *.png.
        mplf.plot(df, type='candle', style='charles',
            title=title,
            ylabel='Price ($)',
            ylabel_lower='Shares \nTraded',
            #volume=True, 
            mav=(3,6,9))
            # ,savefig=imagename)
