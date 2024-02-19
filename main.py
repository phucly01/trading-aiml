from db.cassandra_db import CassandraDb
from db.db import DB
from data.downloader import Downloader
from data.sourceparser import SourceParser
from data.config import Configuration
from db.database import Database
import ast
import pandas as pd
from chart.relativestrengthindex import RSI
import io
import json
from datetime import date
import matplotlib.pyplot as plot
from chart.heikinashi import HeikinAshi


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
                    'close': float,
                    'open': float,
                    'high': float,
                    'low': float,
                    'volume':int
            }
            df = df.astype(datatype) #Change data type
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d') # Change the date format
            df.sort_values(by=['date'], ascending=True, inplace=True)
            dfs.append(df)
            if db is not None:
                tablename = "{}.{}".format(Configuration.db_name, symbol)
                # Cannot do the following because the types in pandas and cassandra are incompatible
                # columns = ["{} {}".format(col, df.dtypes[col]) for col in df.columns]
                # db.create_table(tablename, columns)
                db.writeDataFrame(tablename, df)
            
    return dfs


cfg = Configuration('config/config.cfg')

db = Database(cfg.get_section('DATABASE'))
#db.connect()

# Each source in sp.sources has the following structure:
# - name: the name of the soure, which maps to the filename in data folder and also to the database name
# - url: the url where the data is downloaded
# - symbols: the list of symbols whose data is downloaded, each symbol is a table name under the database
# - start_dates: the list of start date each correspond to each symbol in symbols list
sp = SourceParser(cfg.get_section('DATASOURCE')['SourceFile'], db)

#init_database(sp.sources, db)

dfs = download_data(sp.sources, db=None)

for df in dfs:
    rsi = RSI(df)

    rsival2 = rsi.rsi()

    stochrsi = rsi.tradingview_stochastic_rsi()

    ha = HeikinAshi(df)

    hadf = ha.ha()
    ha.plot(hadf, "Test", "image.png")

    # df2 = pd.DataFrame({'date': df['date'], 'rsi2':rsival2})
    # #df3 = pd.DataFrame({'date': df['date'], 'rsi':stochrsi})

    # ax = df.plot(kind='line', x='date', y='close', color='red')
    # df2.plot(kind='line', ax=ax, x='date', y='rsi2', color='green')
    # #df3.plot(kind='line', x='date', y='rsi', color='blue')
    plot.show()
    

        

# db = CassandraDb()

# db.connect('192.168.1.22', 9042)

# rows = db.read("desc keyspaces")

# for row in rows:
#     print(str(row))



