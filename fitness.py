
import numpy as np
from config_reader import read_config_int
from collections import defaultdict


class GraphBuilder:
    def buildUndirectedGraph(self, grid):
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
class Fitness:
    def __init__(self, floorplan):
        self.floorplan = floorplan
        self.complexity = None
        self.boundary_length = None
        self.adj_graph = None
        self.num_cells = {}
        self.boundary_lengths = {}
        self.areas = {}
        self.create_adjacency_list()
        self.calculate_length()

    def create_adjacency_list(self):
        # 그래프 빌더 인스턴스 생성 및 그래프 빌드
        graph_builder = GraphBuilder()
        self.adj_graph = graph_builder.buildUndirectedGraph(self.floorplan)

    # todo calc_metrics로 바꾸고 여기서 여러가지 다 계산
    def calculate_length(self):

        cell_side_length_mm = read_config_int('constraints.ini', 'Metrics', 'scale')
        for room_num, adj_list in self.adj_graph.items():
            cur_room_adj = self.adj_graph[room_num]
            cur_num_cell = len([(loc) for loc, adj in cur_room_adj.items()])
            self.num_cells[room_num] = cur_num_cell
            cur_sum_adjs = sum(len(v) for v in cur_room_adj.values())
            cur_boundary_length = cur_num_cell * 4 - cur_sum_adjs
            self.boundary_lengths[room_num] = cur_boundary_length * cell_side_length_mm
            cell_area = cell_side_length_mm ** 2
            cur_room_area = cur_num_cell * cell_area
            self.areas[room_num] = cur_room_area
            # todo pa_ratio
            # todo aspaect_ratio
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

    shapes = calculate_shapes(grid)  # TODO 2. to calculate the #edge #vertices of the polygon shape, refer to fitness.py of last year code
    areas = calculate_area_by_color(grid)


    print(f'shape={shapes}')
    print(f'area={areas}')
    for color, info in shapes.items():
        if color != -1:  # Ignore inactive cells
            print(f"Color {color}:  모서리수 = {info['vertices']}, 면개수 = {info['edges']}")


# if __name__ == '__main__':
    # main()