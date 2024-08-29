from config_reader import load_config
class Options:
    def __init__(self, config_file = 'config.ini'):
        config = load_config(config_file)
        # 옵션 값 읽기, 설정 파일에 값이 없으면 fallback 값을 사용
        self.display = config.getboolean('RunningOptions', 'display_place_room_process', fallback=False)
        self.save = config.getboolean('RunningOptions', 'save_place_room_process', fallback=False)
        self.min_size_alloc  = config.getboolean('RunningOption', 'min_size_optimized_allocation', fallback = True)
