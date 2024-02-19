
from pydoc import locate
from db.db import DB
import string
import ast
import pandas as pd

#This class is THE class to be used in the code.  
#All other classes in the db folder are the specific definition for individual
#db type and is instantiated by this class using the information in the config.cfg
class Database(DB):
    db = None
    cfg = None

    ## Instantiate an db object specific to a particular db type.  
    ## 
    def __init__(self, db_config_section) -> None:
        self.cfg = db_config_section
        dbtype = db_config_section['DBType']
        db_file_name = dbtype+"_db"
        db_class_name = string.capwords(dbtype)+"Db"
        class_name = locate("db.{}.{}".format(db_file_name, db_class_name))
        if class_name != None:
            self.db = class_name()


    def connect(self, ip=None, port=None):
        if ip is None:
            ip = ast.literal_eval(self.cfg['DBAddresses']) #Convert string to list
        if port is None:
            port = self.cfg['DBPort']

        return self.db.connect(ip, port)
    
    def is_keyspace_exist(self, name):
        return self.db.is_keyspace_exist(name)
    
    def is_table_exist(self, name):
        return self.db.is_table_exist(name)
    
    def read(self, query):
        return self.db.read(query)

    def create_table(self, name, columns, key_counts=1):
        if not len(columns):
            print("Error: Call to Database.create_table with no columns")
            return False
        return self.db.create_table(name, columns, key_counts)
        

    def create_database(self, name, node_count=1):
        iplist = ast.literal_eval(self.cfg['DBAddresses']) #Convert string to list
        nodes = len(iplist) 
        node_count = nodes if nodes > node_count else node_count
        return self.db.create_database(name, node_count=node_count)



    def insert(self, table, row):
        return self.db.insert(table, row)


    def writeDataFrame(self, table, df: pd.DataFrame):
        return self.db.writeDataFrame(table, df)
    
    def readDataFrame(self, table):
        return self.db.readDataFrame(table)