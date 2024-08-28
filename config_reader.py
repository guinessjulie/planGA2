
import configparser
import ast

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

def read_str_constraints(filename, section):
    config = configparser.ConfigParser()
    config.read(filename)
    constraints = {}
    if section in config:
        for key, value in config.items(section):
            constraints[key] = value
    return constraints

def read_str_constraints_from(config, section): # info read from configparser object
    constraints = {}
    if section in config:
        for key, value in config.items(section):
            constraints[key] = value
    return constraints

def read_ken_int_constraints_from(config, section): # info read key as int
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



def read_ini_file(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def write_ini_file(config, file_name):
    with open(file_name, 'w') as configfile:
        config.write(configfile)

def get_room_names():
    config = read_ini_file('config.ini')
    room_names = {}
    if config.has_section('RoomNames'):
        for room_number, room_name in config.items('RoomNames'):
            room_names[int(room_number)] = room_name
        return room_names

def check_adjacent_requirement():
    ini_filename = 'constraints.ini'
    section = 'AdjacencyRequirements'
    edges_str = read_str_constraints(ini_filename,section )

    adjacency_list = ast.literal_eval(edges_str['adjacent'])  # TO LIST
    return adjacency_list


def read_room_names():
    config = read_ini_file('config.ini')
    section = 'RoomNames'

    # Room ID와 Room Name 매핑
    return {int(key): value for key, value in config[section].items()}
