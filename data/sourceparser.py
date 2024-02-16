import yaml
import json
from urllib.parse import urlparse
from data.config import Configuration

class SourceParser:

    #Structure of sources and database:
    # Each source will be one database, each stock symbol will be one table.
    sources = []

    def __init__(self, path_to_yaml_file, db_obj) -> None:
        
        with open(path_to_yaml_file, 'r') as stream: 
            try:
                loader = yaml.safe_load(stream)

                for row in loader:
                    source = row[list(row.keys())[0]]
                    url = source['url']
                    domain = str(urlparse(url).hostname).replace(".", "")
                    source['name'] = domain
                    try:
                        last_read_table = db_obj.read('SELECT * from {}.{}'.format(source['name'], Configuration.db_table_last_read))
                    except Exception as err:
                        last_read_table = None
                    start_dates = [] 
                    for symbol in source['symbols']:
                        start_dates.append(None if last_read_table is None or last_read_table[symbol] is None else last_read_table[symbol])
                    
                    source['start_dates'] = start_dates
                    self.sources.append(source)
                
            except yaml.YAMLError as exc:
                print(exc)