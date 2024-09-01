import configparser
from config_reader import read_key_int_constraints_from, read_str_constraints_from
import ast
class Req:
    def __init__(self, config_file = 'constraints.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.size_req = self._parse_size_req()
        self.adj_list = self.get_adj_req()
        self.orientation = self.load_orientation_req()
        self.fitness_weight = self.load_weight_req()
        print(self.fitness_weight)
    def _parse_size_req(self):
        size_req = {}
        for room_id in self.config['SizeRequirements']:
            min_area, max_area = map(int, self.config['SizeRequirements'][room_id].split(','))
            size_req[int(room_id)] = (min_area, max_area)
        return size_req
    def get_area_range(self, room_id):
        return self.size_req.get(room_id, (None, None))

    def get_adj_req(self):
        section = 'AdjacencyRequirements'
        edges_str = read_str_constraints_from(self.config, section)
        return ast.literal_eval(edges_str['adjacent'])  # TO LIST

    def load_orientation_req(self):
        section = 'OrientationRequirements'
        return read_key_int_constraints_from(self.config, section)

    def load_weight_req(self):
        section = 'FitnessWeight'
        return read_str_constraints_from(self.config, section)




