
# from cassandra.cluster import Cluster
# from db.db import DB
# import pandas as pd
# import re



# class CassandraDb(DB):
#     session = None     

#     def __init__(self):
#         pass

#     def pd_row_factory(self, columns, rows):
#         return pd.DataFrame(rows, columns=columns)

#     def connect(self, ip, port):
#         cluster = Cluster(ip, port=port)
#         self.session = cluster.connect()
#         self.session.row_factory = self.pd_row_factory
    
#     def create_table(self, name, columns, key_counts=1):
#         try:
#             key_counts = 1 #Cassandra db has only one primary key
#             query = 'CREATE TABLE IF NOT EXISTS {} ('.format(name)
            
#             for column in columns:
#                 if key_counts:
#                     key_counts -= 1
#                     query += column +' PRIMARY KEY, '
#                 else:
#                     query += column + ', '
#             query = re.sub(', $', ')', query) 
#             print("Creating {}".format(query))
#             val = self.session.execute(query)
#             print(" returns {}".format(val))
#             return True
#         except Exception as err:
#             print(err)
#         return False

#     def create_database(self, name, node_count=1):
#         try:
#             val = self.session.execute("CREATE KEYSPACE IF NOT EXISTS {} WITH replication = {{'class':'SimpleStrategy', 'replication_factor' : {}}}".format(name, node_count))
#             return True
#         except Exception as err:
#             print(err)
#         return False

#     def is_keyspace_exist(self, name):
#         try:
#             val = self.session.execute("DESCRIBE KEYSPACE {}".format(name))
#             print(val)
#             return val is not None
#         except Exception as err:
#             print(err)
#         return False
    
#     def is_table_exist(self, name):
#         try:
#             val = self.session.execute("DESCRIBE TABLE {}".format(name)) #Requires keyspace_name.table_name
#             print(val)
#             return val is not None
#         except Exception as err:
#             print(err)
#         return False

#     def read(self, query):
#         return self.session.execute(query)

#     def insert(self, table, row):
#         query = "INSERT INTO {} (".format(table)
#         values="VALUES("
#         for key, value in row:
#             query += key+ ','
#             values += value+','
#         query = re.sub(',$', ')', query) + re.sub(',$', ')', values) + " IF NOT EXISTS"
#         return self.session.execute(query)


#     def writeDataFrame(self, table, df: pd.DataFrame):
#         columns = list(df.columns.values)
#         query = "INSERT INTO {} ({}) VALUES({}) IF NOT EXISTS".format(table, ','.join(columns), ','.join([val.replace(val, "?") for val in columns]))
#         preparedquery = self.session.prepare(query)
#         try:
#             for row in df.loc:
#                 values = [row[col] for col in columns]
#                 self.session.execute(preparedquery, values)
#         except Exception as err:
#             print(err)


#     def readDataFrame(self, table):
#         oldfactory = self.session.row_factory
#         self.session.row_factory = self.pd_row_factory
#         resultset = self.session.execute("SELECT * FROM {}".format(table))
#         return resultset._current_rows