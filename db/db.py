
import abc; #abstract base classes
import pandas as pd

class DB(metaclass=abc.ABCMeta):
        
    @abc.abstractmethod
    def connect(self, ip, port):
        return
    
    @abc.abstractmethod
    def is_keyspace_exist(self, name):
        pass

    @abc.abstractmethod
    def is_table_exist(self, name):
        pass

    @abc.abstractmethod
    def read(self, query):
        pass

    @abc.abstractmethod
    def create_table(self, name, columns, key_counts=1):
        pass

    @abc.abstractmethod
    def create_database(self, name, node_count=1):
        pass


    @abc.abstractmethod
    def insert(self, table, row):
        pass

    @abc.abstractmethod
    def writeDataFrame(self, table, df: pd.DataFrame):
        pass
    
    @abc.abstractmethod
    def readDataFrame(self, table):
        pass
