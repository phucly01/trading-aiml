#from db.cassandra_db import CassandraDb
from db.db import DB
from data.downloader import Downloader
from data.sourceparser import SourceParser
from data.config import Configuration
from db.database import Database
import ast
import pandas as pd
import numpy as np
from chart.relativestrengthindex import RSI
import io
import json
from datetime import date
import matplotlib.pyplot as plot
from chart.stock_chart import Stock
from aiml.tensorflowml import TensorFlow, TensorFlowMLAdam
import aiml.rockikz_x4nth055
from aiml.rockikz_x4nth055 import Rockikz_x4nth055

# from libs.garden.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as FigureCanvas
# from kivy.app import App 
# from kivy.lang import Builder 


def init_database(sources, db):
    if not db.is_keyspace_exist(Configuration.db_name):
        if not db.create_database(Configuration.db_name): 
            print("Error: cannot create the database {}".format(Configuration.db_name))
            return None
    print("database {} exists".format(Configuration.db_name))

    for source in sources: 
        for symbol in source['symbols']:
            table = "{}.{}".format(Configuration.db_name, symbol)
            if not db.is_table_exist(table):
                if not db.create_table(table, ['date date', 'open float', 'close float', 'volume int', 'high float', 'low float']):
                    print("Error: cannot create table {}".format(table))
                    continue

            print("table {} exists".format(symbol))


def download_data(sources, db : Database):
    dfs = []
    for source in sp.sources:
        dl = Downloader(source['name'], source['url'], None) #source['key'])
        for i in range(0,len(source['symbols'])):
            symbol = source['symbols'][i]
            startdate = source['start_dates'][i]        
            data = dl.download(symbol, startdate)
            rows = data.json()['data']['tradesTable']['rows']
            df = pd.DataFrame(row for row in rows)
            df['open'] = df['open'].replace({r'\$':''}, regex=True)
            df['close'] = df['close'].replace({r'\$':''}, regex=True)
            df['high'] = df['high'].replace({r'\$':''}, regex=True)
            df['low'] = df['low'].replace({r'\$':''}, regex=True)
            df['volume'] = df['volume'].replace({r',':''}, regex=True)
            #Convert the data types so they are compatible with the database table's
            datatype = {
                    'date': 'datetime64[ns]',
                    'close': float,
                    'open': float,
                    'high': float,
                    'low': float,
                    'volume': int
            }
            try:
                df = df.astype(datatype) #Change data type
            except Exception as err:
                msg = str(err)
                if "'volume'" in msg:
                    column = []
                    for val in df['volume']:
                        try:
                            column.append(int(val))
                        except Exception as e:
                            column.append(0)
                    df['volume'] = column
            
                df = df.astype(datatype) #Change data type, again
                                            
                
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d') # Change the date format
            print(df.dtypes)
            df.sort_values(by=['date'], ascending=True, inplace=True)
            dfs.append(df)
            if db is not None:
                tablename = "{}.{}".format(Configuration.db_name, symbol)
                # Cannot do the following because the types in pandas and cassandra are incompatible
                # columns = ["{} {}".format(col, df.dtypes[col]) for col in df.columns]
                # db.create_table(tablename, columns)
                db.writeDataFrame(tablename, df)
            
    return dfs



    
    
    
    

        

# db = CassandraDb()

# db.connect('192.168.1.22', 9042)

# rows = db.read("desc keyspaces")

# for row in rows:
#     print(str(row))

# from kivy.uix.gridlayout import GridLayout

# class Window(App): 
  
#     def build(self, figure): 
#         self.frame_layout = GridLayout(cols=2)
        
#         self.form_layout = GridLayout(rows=10)
        
#         self.frame_layout.add_widget(self.form_layout, width=200)
        
#         plot_canvas = FigureCanvas(figure)

#         self.frame_layout.add_widget(plot_canvas)
#         return self.frame_layout
  
  



if __name__ == '__main__':
    cfg = Configuration('config/config.cfg')

    #db = Database(cfg.get_section('DATABASE'))
    #db.connect()

    # Each source in sp.sources has the following structure:
    # - name: the name of the soure, which maps to the filename in data folder and also to the database name
    # - url: the url where the data is downloaded
    # - symbols: the list of symbols whose data is downloaded, each symbol is a table name under the database
    # - start_dates: the list of start date each correspond to each symbol in symbols list
    sp = SourceParser(cfg.get_section('DATASOURCE')['SourceFile'])#, db)

    #init_database(sp.sources, db)

    dfs = download_data(sp.sources, db=None)

    for df in dfs:
        print(df["date"].dtypes.name)
        rsi = RSI(df)

        rsival2 = rsi.rsi()
        stochrsi = rsi.tradingview_stochastic_rsi()
        # df2 = pd.DataFrame({'date': df['date'], 'rsi2':rsival2})
        # df3 = pd.DataFrame({'date': df['date'], 'tvrsi':stochrsi[0]})

        # ax = df.plot(kind='line', x='date', y='close', color='red')
        # df2.plot(kind='line', ax=ax, x='date', y='rsi2', color='green')
        # df3.plot(kind='line', x='date', y='tvrsi', color='blue')
        # plot.show()
        

        # s = Stock(df)
        # hadf = s.heikin_ashi()
        # macd = s.macd(fast_length=8, slow_length=21, signal_smooth=5, show=False)
        # figure = s.plot(df, hadf, macd, "Test", True)
        
        #s.plot(df=df, heikinashi=hadf, macd=macd, title="OCEA")
        # s.plot(df=df, heikinashi=hadf, title="OCEA")
        
        # rsi1 = s.rsi(df['close'])  
        # s.plot(rsi1, "OCEA", "line") 
        
        
        # tf = TensorFlowML(df.ticker, df, "date", ["close", "open", "high", "low"])
        
        tf = TensorFlowMLAdam("HYLN", df, "date", "close")
        tf.train()
        tf.test()
        tf.predict()
        tf.plot(30)
        
        # rx = Rockikz_x4nth055()
        # rx.train(df)
        # rx.show_prediction()
    
    # Window(figure).run()