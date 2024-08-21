
import numpy as np
from config_reader import read_config_int, check_adjacent_requirement
from collections import defaultdict,deque
from Constraints import Req
# todo test room_polygons: shape requirement fitness function examine and ensure to make sensible
# todo fitness에서 self.room_polygon[room_id].area/perimeter/simplicity etc 접근 가능
# underway fit.area etc와 polygon의 metrics를 비교

class Fitness:
    def __init__(self, floorplan, num_rooms, room_polygons):
        self.floorplan = floorplan
        self.num_rooms = num_rooms
        self.room_polygons = room_polygons # underway room_polygon instance는 corners, area, perimeter, min_Length, simplicity를 가지고 있다.
# underway to use room_polygons property
# info room_polygon 구조 {room_id:RoomPolygon(property=corner,area,perimeter,min_length, simplicity
        self.boundary_length = None
        self.room_adjacency_list = None
        self.num_cells = {} # todo 이 모든 dictionaryu들을 모두 저장할 필요가 없다  평균값만 저장하자
        self.boundary_lengths = {}
        self.room_areas = {} # info 1. cell을 count해서 구한 area와 RoomPolygon 인스턴스에서 가져온  area는 구하는 방법은 다르지만 결과는 같다. 2.todo 하지만 RoomPolygon에서 Edge를 이동할 것이기 때문에 여기서 구한 area를 지우고 RoomPolygon에서 구한 area를 쓰는 게 좋겠다. 일단 두자.
        self.pa_ratios = {}
        self.reqs = Req()
        self.measure_room_metrics() # todo create_cell_adjacency_list 를 이 함수 속에 넣고 self.cell_adjacencies는
        self.adj_satisfaction = self.calc_adj_satisfaction()
        self.size_satisfaction = self.calc_size_satisfaction()
        self.complexity = self.calc_complexity()
        self.rectangularity = self.calc_rectangularity()
        self.regularity = self.calc_regularity()


    def calc_simplicity(self):
        return np.average([(room.simplicity) for room_id, room in self.room_polygons.items()]) # underway 동작확인
    def calc_areas(self):
        return {room_id: room.area for room_id, room in self.room_polygons.items()} # underway 동작확인

    def calc_rectangularity(self):
        return np.average([room.rectangularity for room_id, room in self.room_polygons.items()]) # todo change .values and remove room_id

    def calc_regularity(self):
        return np.average([room.regularity for room in self.room_polygons.values()])

    def calc_complexity(self):
        return np.average([room.complexity for room in self.room_polygons.values()])

    def room_min_length(self):
        return np.average([(room.min_length) for room_id, room in self.room_polygons.items()]) # underway 동작확인

    def get_basic_properties(self): # underway 원래 여기에서 기본 연산을 할 작정이었음
        pass

    def room_shape_efficiency(self):
        print(f'room_polygons = {self.room_polygon}')


    def calc_adj_satisfaction(self):
        adj_requirement = check_adjacent_requirement()
        adj_list = self.create_room_adjacency_list()
        score = 0
        total_requirements = len(adj_requirement)

        for room1, room2 in adj_requirement:
            if room2 in adj_list[room1]:
                score += 1

        return score / total_requirements

    def calc_size_satisfaction(self):
        total_score = 0
        total_req_rooms = len(self.reqs.size_req)
        cell_side_length_mm = read_config_int('constraints.ini', 'Metrics', 'scale')
        size_req = self.reqs.size_req
        for room_id, area_in_cells in self.room_areas.items():
            min_area, max_area = self.reqs.get_area_range(room_id)

            if min_area is None or max_area is None:
                continue

            # constraints.ini 파일에는 Size Requirements가 m 단위로 되어 있다.
            area_in_m2 = area_in_cells * (1/(cell_side_length_mm**2)) # 셀 하나의 크기는 1m²

            # 피트니스 점수 계산
            if min_area <= area_in_m2 <= max_area:
                score = 1  # 완전히 범위 안에 들어올 경우
            elif area_in_m2 < min_area:
                score = area_in_m2 / min_area  # 최소 면적에 비례한 점수
            else:
                score = max_area / area_in_m2  # 최대 면적에 비례한 점수

            total_score += score

        # 총 피트니스 점수는 모든 방의 평균 점수로 계산
        size_satisfaction = total_score / total_req_rooms
        return size_satisfaction
    def create_room_adjacency_list(self):
        """
        floorplan을 순회하지 않고 각 방의 셀을 기준으로 인접 리스트를 만드는 함수.
        BFS(너비 우선 탐색)을 사용하여 이미 탐색된 셀은 건너뜁니다.
        """
        rows, cols = self.floorplan.shape
        adj_list = defaultdict(set)
        visited = np.full(self.floorplan.shape, False)  # 방문한 셀을 추적
        processed_rooms = set()  # 이미 인접 리스트가 만들어진 방 번호를 추적

        def bfs(r, c):
            queue = deque([(r, c)])
            room = self.floorplan[r, c]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우

            while queue:
                x, y = queue.popleft()
                visited[x, y] = True

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                        if self.floorplan[nx, ny] == room:
                            queue.append((nx, ny))
                        elif self.floorplan[nx, ny] != -1:
                            adj_list[room].add(self.floorplan[nx, ny])
                            adj_list[self.floorplan[nx, ny]].add(room)

        for r in range(rows):
            for c in range(cols):
                room = self.floorplan[r, c]
                if room != -1 and not visited[r, c] and room not in processed_rooms:
                    bfs(r, c)
                    processed_rooms.add(room)

        return adj_list

    def cell_adjacent_graph_for_rooms(self, grid):
        rows, cols = grid.shape
        adjGraphs = {}  # 숫자별로 그래프를 저장할 딕셔너리

        # 모든 좌표에 대해 그래프 생성
        for x in range(rows):
            for y in range(cols):
                room_type = grid[x, y]
                if room_type == -1:  # -1은 방이 아니므로 무시
                    continue

                # 현재 방의 그래프가 없으면 초기화
                if room_type not in adjGraphs:
                    adjGraphs[room_type] = {}

                curCell = (x, y)
                adjs = self.adjacent_four_way(curCell, rows, cols)
                child = [adj for adj in adjs if grid[adj[0], adj[1]] == room_type]
                adjGraphs[room_type][curCell] = child

        return adjGraphs

    def adjacent_four_way(self, loc, rows, cols):
        x, y = loc
        potential_moves = [
            (x - 1, y),  # 위쪽
            (x + 1, y),  # 아래쪽
            (x, y - 1),  # 왼쪽
            (x, y + 1)  # 오른쪽
        ]

        # 유효한 좌표만 반환
        valid_moves = [
            (nx, ny) for nx, ny in potential_moves
            if 0 <= nx < rows and 0 <= ny < cols
        ]

        return valid_moves


    # todo 0818 to convert to polygon
    def measure_room_metrics(self):

        room_cell_adjacencies = self.cell_adjacent_graph_for_rooms(self.floorplan)
        cell_side_length_mm = read_config_int('constraints.ini', 'Metrics', 'scale')
        aspect_ratios = []
        for room_num, adj_list in room_cell_adjacencies.items():
            cur_room_adj = room_cell_adjacencies[room_num]
            cur_num_cell = len([(loc) for loc, adj in cur_room_adj.items()])
            self.num_cells[room_num] = cur_num_cell
            cur_sum_adjs = sum(len(v) for v in cur_room_adj.values())
            cur_unit_length = cur_num_cell * 4 - cur_sum_adjs
            cur_boundary_length = cur_unit_length * cell_side_length_mm
            self.boundary_lengths[room_num] = cur_boundary_length * cell_side_length_mm
            cell_area = cell_side_length_mm ** 2
            cur_room_area = cur_num_cell * cell_area
            self.room_areas[room_num] = cur_room_area
            cur_pa_ratio = 16*cur_room_area / cur_boundary_length**2
            self.pa_ratios[room_num] = cur_pa_ratio

            # todo pa_ratio test
        self.pa_ratio = np.average([pa_ratio for room, pa_ratio in self.pa_ratios.items()])

            # todo get directional_side from get_south_ratio etc
        return





    def calculate_area_by_color(grid):
        # 색상별 면적(셀의 수)을 저장할 딕셔너리
        area_by_color = {}

        # 그리드를 순회하며 각 색상별로 셀의 수를 계산
        for row in grid:
            for cell in row:
                if cell not in area_by_color:
                    area_by_color[cell] = 1
                else:
                    area_by_color[cell] += 1

        # -1 (비활성 셀)은 제외하고 반환
        if -1 in area_by_color:
            del area_by_color[-1]

        return area_by_color

    def calculate_shape(self, floorplan):
        pass


def main():

    grid = np.array([
        [2, 2, 2, -1, -1],
        [4, 1, 1, -1, -1],
        [4, 1, 3, -1, -1],
        [4, 4, 3, 3, 3],
        [4, 4, 3, 3, 3]
    ])

    # shapes = calculate_shapes(grid)  # TODO 2. to calculate the #edge #vertices of the polygon shape, refer to fitness.py of last year code
    # areas = calculate_area_by_color(grid)
    #
    #
    # print(f'shape={shapes}')
    # print(f'area={areas}')
    # for color, info in shapes.items():
    #     if color != -1:  # Ignore inactive cells
    #         print(f"Color {color}:  모서리수 = {info['vertices']}, 면개수 = {info['edges']}")
    #

# if __name__ == '__main__':
    # main()