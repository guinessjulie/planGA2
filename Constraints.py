import configparser
class Req:
    def __init__(self, config_file = 'constraints.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.size_req = self._parse_size_req()
    def _parse_size_req(self):
        size_req = {}
        for room_id in self.config['SizeRequirements']:
            min_area, max_area = map(int, self.config['SizeRequirements'][room_id].split(','))
            size_req[int(room_id)] = (min_area, max_area)
        return size_req
    def get_area_range(self, room_id):
        return self.size_req.get(room_id, (None, None))