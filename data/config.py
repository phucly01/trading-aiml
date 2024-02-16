import configparser


class Configuration:
    db_table_last_read = 'last_read_date'

    config = None

    def __init__(self, path_to_config_file):
        self.config = configparser.ConfigParser()
        self.config.read(filenames=path_to_config_file)

    
    def get_section(self, section_name):
        return self.config[section_name]
        