
from re import A
from matplotlib import legend, pyplot, ticker, dates
import pandas as pd
import finplot as plot
import numpy as np
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector

import plotly.graph_objects as ply 


class Stock:
    
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.data.index = pd.DatetimeIndex(self.data['date'])
        
    
    def heikin_ashi(self):
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
        volume = self.data['volume']
        ha_close = []
        ha_open  = []
        ha_high  = []
        ha_low   = []
        ha_close.append((open.iloc[0] + close.iloc[0] + high.iloc[0] + low.iloc[0])/4)
        ha_open.append((open.iloc[0] + close.iloc[0])/2 ) #There are no previous ha_open and ha_close for first one
        ha_high.append(max(high.iloc[0], ha_close, ha_open))
        ha_low.append(min(low.iloc[0], ha_close, ha_open))

        for i in range(1, len(self.data)):
            ha_close.append((open.iloc[i] + close.iloc[i] + high.iloc[i] + low.iloc[i])/4)
            ha_open.append((ha_open[i-1] + ha_close[i-1])/ 2)
            ha_high.append(max(high.iloc[i], ha_close[i], ha_open[i]))
            ha_low.append(min(low.iloc[i], ha_close[i], ha_open[i]))

        ha_data = {'date': self.data['date'], 'close': ha_close, 'open': ha_open, 'high': ha_high, 'low': ha_low, 'volume':volume}
        df = pd.DataFrame(ha_data)
        df.index = pd.DatetimeIndex(df['date'])
        return df
    
    
    def macd(self, fast_length:int=12, slow_length:int=26, signal_smooth:int=9, show:bool=False):
        fast_length     = self.data['close'].ewm(span=fast_length, adjust=False).mean()
        slow_length     = self.data['close'].ewm(span=slow_length, adjust=False).mean()
        macd      = fast_length - slow_length
        signal    = macd.ewm(span=signal_smooth, adjust=False).mean()
        histogram = macd - signal
        
        results = {
                'fast_length': fast_length,
                'slow_length': slow_length,
                'macd': macd,
                'histogram':histogram,
                'signal':signal
        }
        
        if show:
            fb_green = dict(y1=macd.values,y2=signal.values,where=signal<macd,color="green",alpha=0.6,interpolate=True)
            fb_red   = dict(y1=macd.values,y2=signal.values,where=signal>macd,color="red",alpha=0.6,interpolate=True)
            fb_green['panel'] = 1
            fb_red['panel'] = 1
            fb       = [fb_green,fb_red]
            import mplfinance as mpf
            apds = [mpf.make_addplot(fast_length,color='blue'),
                    mpf.make_addplot(slow_length,color='yellow'),
                    mpf.make_addplot(histogram,type='bar',width=0.7,panel=1,
                                    color='gray',alpha=1,secondary_y=True),
                    mpf.make_addplot(macd,panel=1,color='purple',secondary_y=False),
                    mpf.make_addplot(signal,panel=1,color='black',secondary_y=False)#,fill_between=fb),
                ]

            s = mpf.make_mpf_style(base_mpf_style='classic',rc={'figure.facecolor':'lightgray'})

            mpf.plot(self.data,type='candle',addplot=apds,figscale=1.6,figratio=(6,5),title='MACD',
                    style=s,volume=True,volume_panel=2,panel_ratios=(3,4,1),fill_between=fb)#,show_nontrading=True)
        else:
            return pd.DataFrame(results)
    
     
       
    
    def rsi(self, data, period: int = 14): 
        """
        First formula:
            rsi1 = 100 - (100 / (1 + avggain/avgloss))
            First formula uses data from 0 to period-1
        """
        delta = np.append([], np.diff(data)) #Delta will be one less element, insert 0 so the data lines up with the index
        avggain = np.sum(delta[0:period].clip(min=0))/period
        avgloss = abs(np.sum(delta[0:period].clip(max=0)))/period
        
        data = np.empty(len(data))
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
        data[period] = 100 - (100 / (1 + avggain/avgloss))
        i = period+1 #0 based index
        for val in delta[period:]:   
            avggain = (avggain * n + gain_func(val))/period
            avgloss = (avgloss * n + loss_func(val))/period
            data[i] = 100 - (100 / (1 + avggain/avgloss))
            i += 1

        return data
    
     
    def contains(self, x1, y1, width, height, x, y) ->bool:
        x2 = x1 + width
        y2 = y1 + height
        
        leftx = min(x1, x2)
        boty = min(y1, y2)
        rightx = max(x1, x2)
        topy = max(y1, y2)
        return leftx <= x and x <= rightx and boty <= y and y <= topy
        

    def hover(self, event):
        for anotation in self.annotions:
            if event.inaxes is not None:
                if event.inaxes == anotation.axes:
                    for container in event.inaxes.axes.patches:
                        x = container.get_x()
                        y = container.get_y()
                        h = container.get_height()
                        if self.contains(x, y, container.get_width(), h, event.xdata, event.ydata):
                            anotation.xy = (x, y)
                            anotation.set_text("{}[{}]".format(y, h))
                            anotation.get_bbox_patch().set_alpha(0.4)
                            anotation.set_visible(True)
                            return
            else: 
                anotation.set_visible(False)


    def zoom(self, layout, xrange):
        in_view = self.inputs[0].loc[self.fig.layout.xaxis.range[0]:self.fig.layout.xaxis.range[1]]
        print("{},{}".format(in_view.high.min(), in_view.high.max()))
        self.fig.layout.yaxis.range = [in_view.high.min() - 10, in_view.high.max() + 10]
    
    def plot(self, df:pd.DataFrame, ha:pd.DataFrame, macd: pd.DataFrame, title:str="Chart", show:bool=True):
        
        self.inputs = [df, ha, macd]
        
        data=[
                ply.Candlestick(
                    x=df.index,
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                    increasing_line_color='#adff2f',
                    decreasing_line_color='#ff4500'
                ),
                ply.Candlestick(
                    x=ha.index,
                    open=ha.open,
                    high=ha.high,
                    low=ha.low,
                    close=ha.close,
                    increasing_line_color='#013220',
                    decreasing_line_color='#8b0000'
                )
            ] 
        
        layout = ply.Layout(
            xaxis=dict(
                autorange=False,
                rangeslider=dict(
                    visible = False
                ),
                type='date',
                range=[0,1000] 
            ),
            yaxis=dict(
                title='Ticks',
                titlefont=dict(
                    family='Arial, sans-serif',
                    size=18,
                    color='lightgrey'
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    family='Old Standard TT, serif',
                    size=14,
                    color='black'
                ),
                exponentformat='e',
                showexponent='all',
                tickmode='linear',
                tick0=0.500,
                dtick=.0001,
                autorange=False,
                range=[100,200]
                
            ),
            #annotations = getChartBarNumbers(),
            height = 800
            
        )
        self.fig = ply.FigureWidget(data=data, layout=layout)
        
        self.fig.layout.on_change(self.zoom, 'xaxis.range')


        if show:
            self.fig.show()
        
        return self.fig

    
    def plots(self, df: pd.DataFrame, heikinashi: pd.DataFrame=None, macd:  pd.DataFrame=None, title: str="Chart"): 

        # Plot candlestick.
        # Add volume.
        # Add moving averages: 3,6,9.
        # Save graph to *.png.    
        ax1, ax2, ax3 = pyplot.make.create_plot(title=title, rows=3)
        
        df[['open','close','high','low']].plot(ax=ax1, kind="candle") #Normal candle stick
        
        if heikinashi is not None:
            heikinashi.drop('date', axis=1, inplace=True)
            heikinashi.plot(ax=ax2, kind="candle")
            
        if macd is not None:
            macd[['macd']].plot(ax=ax2, legend='MACD')
            macd[['signal']].plot(ax=ax2, legend='Signal')
#              # plot macd with standard colors first
# macd = df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean()
# signal = macd.ewm(span=9).mean()
# df['macd_diff'] = macd - signal
# fplt.volume_ocv(df[['Date','Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
# fplt.plot(macd, ax=ax2, legend='MACD')
# fplt.plot(signal, ax=ax2, legend='Signal')  

            
        plot.show()
               
        # mpf.plot(df, type=type, style='charles',
        #     title=title,
        #     ylabel='Price ($)',
        #     ylabel_lower='Shares \nTraded',
        #     #volume=True, 
        #     # mav=(3,6,9)
        #     # ,savefig=imagename)
        # )
        
    
    # def onselect(self, xmin, xmax):
    #     indmin, indmax = np.searchsorted(x, (xmin, xmax))
    #     indmax = min(len(x) - 1, indmax)
    #     region_x = x[indmin:indmax]

    #     if len(region_x) >= 2:
    #         # plot the original diagram but with a different x range
    #         self.main_plots.clear()
    #         mpf.plot(df,
    #                 ax=ax2,
    #                 type="candle"
    #             )
    #         ax2.set_xlim(region_x[0], region_x[-1])
    #         ax2.set_title('Zoomed')
    #         fig.canvas.draw_idle()
        
    def plot_candles(self, price: pd.DataFrame, heikinashi: pd.DataFrame, macd: pd.DataFrame, show:bool=False):
        fig, axes = pyplot.subplots(3, 1, 
                                    facecolor='white',  #Background
                                    sharex=True         #Share the same x axis
                                    )
        self.fig = fig
        
        for ax in axes:
            ax.set_facecolor('white')
        pyplot.xticks(rotation=45, ha='right')
         
        
        # self.annotions = [ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
        #             bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
        #             arrowprops=dict(arrowstyle="->"))]
        # self.annotions[0].set_visible(False)
       # fig.canvas.mpl_connect("motion_notify_event", self.hover)

    
        # Normal candle based on actual price:
        if price is not None:
            #define width of candlestick elements
            price_body_width = .5
            price_wick_width = .1

            #define up and down prices
            up = price[price.close>=price.open]
            down = price[price.close<price.open]

            #plot up prices
            bar1_a = axes[0].bar(up.index, up.close - up.open, price_body_width, bottom=up.open, color='#adff2f', edgecolor='#adff2f', zorder=3)
            axes[0].bar(up.index, up.high - up.close, price_wick_width, bottom=up.close, color='#adff2f', zorder=3, align='edge')
            axes[0].bar(up.index, up.low - up.open, price_wick_width, bottom=up.open, color='#adff2f', zorder=3, align='edge')

            #plot down prices
            bar1_d = axes[0].bar(down.index, down.close - down.open, price_body_width, bottom=down.open, color='#ff4500', edgecolor='#ff4500', zorder=3)
            axes[0].bar(down.index, down.high - down.open, price_wick_width, bottom=down.open, color='#ff4500', zorder=3, align='edge')
            axes[0].bar(down.index, down.low - down.close, price_wick_width, bottom=down.close, color='#ff4500', zorder=3, align='edge') 
                
        #Heikin Ashi candle, smaller shape
        if heikinashi is not None:
            #define width of candlestick elements
            price_body_width = .25
            price_wick_width = .05

            #define up and down prices
            up = heikinashi[heikinashi.close>=heikinashi.open]
            down = heikinashi[heikinashi.close<heikinashi.open]

            #plot up prices
            bar2_a = axes[0].bar(up.index, up.close - up.open, price_body_width, bottom=up.open, color='#013220', edgecolor='#013220', zorder=3)
            axes[0].bar(up.index, up.high - up.close, price_wick_width, bottom=up.close, color='#013220', zorder=3)
            axes[0].bar(up.index, up.low - up.open, price_wick_width, bottom=up.open, color='#013220', zorder=3)

            #plot down prices
            bar2_d = axes[0].bar(down.index, down.close - down.open, price_body_width, bottom=down.open, color='#8b0000', edgecolor='#8b0000', zorder=3)
            axes[0].bar(down.index, down.high - down.open, price_wick_width, bottom=down.open, color='#8b0000', zorder=3)
            axes[0].bar(down.index, down.low - down.close, price_wick_width, bottom=down.close, color='#8b0000', zorder=3)
            
        
        axes[0].legend(handles=[bar1_a, bar2_a, bar1_d, bar2_d], 
                       labels=['','',"Normal Candle","Heikin Ashi"], 
                       handlelength=1, 
                       columnspacing=-0.85, #Horizontal spacing
                       numpoints=1, 
                       labelspacing=1,  #Vertical spacing
                       ncol=2)
        
        
        if macd is not None:
            axes[1].plot(macd.index, macd.macd, color='#ffa500')
            axes[1].plot(macd.index, macd.signal, color='#0000ff')
            axes[1].legend(labels=['MACD', 'Signal'])
            
            
        pyplot.tight_layout() 
        #axes[1].set_xlim(xmin=pd.to_datetime('2024-02-01'))
        
        pyplot.autoscale(enable=True, axis='y', tight=True) 
            
        

        SpanSelector(
            axes[2],
            self.onselect,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True
        )
        
            
        if show:
            pyplot.show()
            
        return fig