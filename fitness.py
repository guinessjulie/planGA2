
import numpy as np
from config_reader import read_config_int, check_adjacent_requirement
from collections import defaultdict,deque
from reqs import Req
# todo test room_polygons: shape requirement fitness function examine and ensure to make sensible
# todo fitness에서 self.room_polygon[room_id].area/perimeter/simplicity etc 접근 가능
# underway fit.area etc와 polygon의 metrics를 비교
# todo strategy for create_population
#   1. 좋은 seed를 구한다. run_iteration 반복에서 각 seed별 평균, 최대, 세가지 목표를 반영한 파레토를 구해서 기준에 어긋나면 버린다.
#   2.계속해서  floorplans를  생성한다.

class Fitness:
    def __init__(self, floorplan, num_rooms, room_polygons, reqs):
        self.floorplan = floorplan
        self.num_rooms = num_rooms
        self.room_polygons = room_polygons # room_polygon instance는 corners, area, perimeter, min_Length, simplicity를 가지고 있다.
        self.reqs = reqs
        self.adj_satisfaction = self.calc_adj_satisfaction()
        self.size_satisfaction = self.calc_size_satisfaction()
        self.simplicity = self.calc_simplicity()
        self.rectangularity = self.calc_rectangularity()
        self.regularity = self.calc_regularity()
        self.pa_ratio = self.calc_pa_ratio()
        self.orientation_satisfaction = self.calc_orientation_satisfaction()

    @property # todo see proposal from GPT => saved in obsidian '논문-Fitness 계산' 참조
    def fitness(self):
        epsilon = 1e-9
        adj_weight, ori_weight, size_weight = int(self.reqs.fitness_weight['adjacency']), int(self.reqs.fitness_weight['orientation']), int(self.reqs.fitness_weight['size'])
        adj =  self.adj_satisfaction if self.adj_satisfaction !=0 else epsilon
        size =  self.size_satisfaction if self.size_satisfaction !=0 else epsilon
        ori =  self.orientation_satisfaction if self.orientation_satisfaction !=0 else epsilon
        total_fitness =( adj_weight + ori_weight + size_weight )/ ((adj_weight /adj) + (ori_weight/size) + (size_weight/ori))
        return total_fitness

    def calc_areas(self):
        return {room_id: room.area for room_id, room in self.room_polygons.items()} # underway 동작확인
    def calc_simplicity(self):
        return np.average([(room.simplicity) for room_id, room in self.room_polygons.items()]) # underway 동작확인

    def calc_rectangularity(self):
        return np.average([room.rectangularity for room_id, room in self.room_polygons.items()]) # todo change .values and remove room_id

    def calc_regularity(self):
        return np.average([room.regularity for room in self.room_polygons.values()])

    def calc_pa_ratio(self):
        return np.average([room.pa_ratio for room in self.room_polygons.values()])

    def room_min_length(self):
        return np.average([(room.min_length) for room_id, room in self.room_polygons.items()]) # underway 동작확인


    def calc_adj_satisfaction(self): # info fitness에서 직접 계산
        adj_requirement = self.reqs.get_adj_req()
        adj_list = self.create_room_adjacency_list()
        score = 0
        total_requirements = len(adj_requirement)

        for room1, room2 in adj_requirement:
            if room2 in adj_list[room1]:
                score += 1

        return score / total_requirements

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
    def calc_size_satisfaction(self):
        total_score = 0
        total_req_rooms = len(self.reqs.size_req)
        cell_side_length_mm = read_config_int('config.ini', 'Metrics', 'scale')
        size_req = self.reqs.size_req
        room_areas = {key : room.area for key, room in self.room_polygons.items()}
        for room_id, area_in_mm2 in room_areas.items():
            min_area, max_area = self.reqs.get_area_range(room_id)

            if min_area is None or max_area is None:
                continue

            # constraints.ini 파일에는 Size Requirements가 m 단위로 되어 있다.
            # 내부에서는 mm2 로 표시된다.
            area_in_m2 = area_in_mm2 * (1/(cell_side_length_mm**2)) # 셀 하나의 크기는 1m²

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

    def calc_orientation_satisfaction(self):

        total_satisfaction = 0

        if not self.reqs.orientation:
            return 1.0  # 방향 요구사항이 없는 경우 1.0을 반환

        for room_id, room_polygon in self.room_polygons.items():
            required_direction = self.reqs.orientation.get(room_id, None)
            if required_direction:
                actual_directions = self.calculate_boundary_directions(room_id)
                if required_direction in actual_directions:
                    total_satisfaction += 1  # 완전 일치 시 1점

        return total_satisfaction / len(self.reqs.orientation) if self.reqs.orientation else 1.0

    def calculate_boundary_directions(self, room_id):
        """
        방의 셀을 기반으로 경계에 인접한 셀의 방향을 구하는 함수.
        """
        rows, cols = self.floorplan.shape
        boundary_directions = set()

        # 방 ID에 해당하는 셀들의 집합 가져오기
        room_cells = np.argwhere(self.floorplan == room_id)

        for cell in room_cells:
            r, c = cell

            # 경계에 위치한 셀의 방향 추가
            if r == 0:
                boundary_directions.add('north')
            if r == rows - 1:
                boundary_directions.add('south')
            if c == 0:
                boundary_directions.add('west')
            if c == cols - 1:
                boundary_directions.add('east')

            # Invalid cell (boundary)와 접하는 방향을 확인
            neighbors = self.get_neighbors(r, c)
            for direction, (nr, nc) in neighbors.items():
                if not (0 <= nr < rows and 0 <= nc < cols) or self.floorplan[nr, nc] == -1:
                    boundary_directions.add(direction)

        return boundary_directions

    def get_neighbors(self, r, c):
        """
        셀의 상하좌우 이웃을 반환하는 함수.
        """
        return {
            'north': (r - 1, c),
            'south': (r + 1, c),
            'west': (r, c - 1),
            'east': (r, c + 1)
        }


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