
import configparser


def load_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def read_constraint(filename, section):
    config = configparser.ConfigParser()
    config.read(filename)
    constraints = {}
    if section in config:
        for key, value in config.items(section):
            constraints[int(key)] = value
    return constraints
def read_config_boolean(filename, section, key):
    config = load_config(filename)
    return config.getboolean(section, key)

def read_config_int(filename, section, key):
    config = load_config(filename)
    return config.getint(section, key)



