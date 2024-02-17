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
                if not db.create_table(table, ['date date', 'open int', 'close int', 'volume bigint', 'high int', 'low int']):
                    print("Error: cannot create table {}".format(table))
                    continue

            print("table {} exists".format(symbol))


def download_data(sources, db : Database):
    for source in sp.sources:
        dl = Downloader(source['name'], source['url'], None) #source['key'])
        for i in range(0,len(source['symbols'])):
            symbol = source['symbols'][i]
            date = source['start_dates'][i]        
            data = dl.download(symbol, date)
            pf = pd.read_csv(io.StringIO(data))
            if db is not None:
                pf.to_sql()
                enddate=date
                for row in data:
                    db.insert(symbol, row)
                    if enddate < row['date']:
                        enddate = row['date']
                source['start_dates'][i] = enddate
            if df is not None:
                return pd.read_csv()
            
        


cfg = Configuration('config/config.cfg')

db = Database(cfg.get_section('DATABASE'))
db.connect()

# Each source in sp.sources has the following structure:
# - name: the name of the soure, which maps to the filename in data folder and also to the database name
# - url: the url where the data is downloaded
# - symbols: the list of symbols whose data is downloaded, each symbol is a table name under the database
# - start_dates: the list of start date each correspond to each symbol in symbols list
sp = SourceParser(cfg.get_section('DATASOURCE')['SourceFile'], db)

init_database(sp.sources, db)

df = pd.DataFrame()

download_data(sp.sources, db, df)

rsi = RSI(df, 14, 3, 3)

rsival = rsi.tradingview_rsi(True)

stochrsi = rsi.tradingview_stochastic_rsi()

print(rsival)

print(".....")

print(stochrsi)

        

# db = CassandraDb()

# db.connect('192.168.1.22', 9042)

# rows = db.read("desc keyspaces")

# for row in rows:
#     print(str(row))



