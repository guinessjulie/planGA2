
import configparser

def read_constraint(filename, section):
    config = configparser.ConfigParser()
    config.read(filename)
    constraints = {}
    if section in config:
        for key, value in config.items(section):
            constraints[int(key)] = value
    return constraints
