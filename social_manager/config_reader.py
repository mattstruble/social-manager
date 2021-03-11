import configparser
from copy import deepcopy


class ConfigReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(file_path)

        self.config = {}
        if len(self.config_parser.sections()) > 1:
            for section in self.config_parser.sections():
                self.config[section] = {}
                for key, val in self.config_parser.items(section):
                    self.config[section][key] = deepcopy(val)
        else:
            for section in self.config_parser.sections():
                for key, val in self.config_parser.items(section):
                    self.config[key] = deepcopy(val)


    def __getitem__(self, item):
        return self.config[item]

    def get_config(self, section=None):
        if section:
            return self.config[section]

        return self.config

    def __repr__(self):
        return "{}: {}".format(self.file_path, self.config)