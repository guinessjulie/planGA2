from GridDrawer import GridDrawer
from GridPolygon import GridPolygon
from pareto import is_dominated
import random
import numpy as np
from scipy.ndimage import label
from scipy.spatial import distance
from queue import Queue
from fitness import Fitness
from simplify import count_cascading_cells
from plan import find_parallel_adjacent_cells, assign_cells_to_adjacent_room, is_room_split
from options import Options
from reqs import Req
from convert import min_max_scaling
from scipy.ndimage import label

# todo 0904
#  crossover에서 방 선택후 그 방만 다른 부모의 해와 교환하도록 현재 로직
#  이 로직은 해당 방의 크기를 늘리고, 다른 방을 쪼개놓을 수 있음
#  로직을 바꿔서 두 방의 교차영역만을 그 방으로 만들고
#  나머지 빈 공간은 인접 룸 중 하나를 선택하여 모두 칠한다.

# underway 0904 두 floorplan이 같은지 검사
class GeneticAlgorithm:
    def __init__(self, population):
        # info: population taken from floorplans_dict {seed:[(fl, ft), (fl, ft], (fl, ft) ... , seed2: [(fl, ft), (fl, ft), ...]} => take only values() part => into list
        #  so population structure:  [(fl, ft),(fl, ft),...(fl, ft)], ...[(fl, ft),(fl, ft),...(fl, ft)]
        self.population = population
        self.options = Options()  # todo num_rooms 만 가져오자

    def run(self, num_generations, population_size, mutation_rate):
        reqs = Req()
        options = Options()
        num_rooms=options.num_rooms
        for generation in range(num_generations):
            new_population = self.create_new_generation(population_size, mutation_rate, reqs, num_rooms)
            self.population.extend(new_population)
            # self.population = self.compute_pareto_front()[:population_size]
            # best_solution = max(self.population, key=lambda sol: sol[1].adj_satisfaction) # underway 0904 need error resolving
            # print(f'Generation {generation}: Best adj_satisfaction = {best_solution[1].adj_satisfaction}') # underway 0904

    # todo population과 self.population  차이

    def select_parents(self, num_parents):
        # info  self.population은 [fp1, fp2, ...] 형태의 리스트이며, 각 fp는 (np.array, fitness) 형태의 튜플
        #  [fp[1] for fp in self.population]는 전체 튜플을 가져옴. 이는 self.population이 중첩된 리스트 구조를 가지기 때문
        all_floorplans_fitness = [flt for seed_sub in self.population for flt in seed_sub]
        fitness_scores = [flt[1].fitness for flt in all_floorplans_fitness]
        # todo info comment out after testing crossover, 파퓰레이션이 적을 때에는 똑같은 해끼리 부모가 되는 경우가 많아서 일단 이 연산을 배제하고 나중에 다시 붙여서 테스트한다.
        # fitness_scores = min_max_scaling(fitness_scores) # todo fitness 객체를 모두 가지고 있지 말고, 대표값만 가지고 있는 것이 좋겠다.
        parents = random.choices(all_floorplans_fitness, weights=fitness_scores, k=num_parents)
        average_fitness = np.average([flt[1].fitness for flt in all_floorplans_fitness])  # todo to display in the label
        parent_average_fitness = np.average([flt[1].fitness for flt in parents])  # todo to display in the label
        print(f'average fitness value \n all_floorplans = {average_fitness} \n parents = {parent_average_fitness}')
        return parents

        # 교차 전략
        return child_floorplan

    def mutate(self, floorplan, mutation_rate):
        room_id = random.choice(range(1, self.options.num_rooms + 1))
        if random.random() < mutation_rate:
            return assign_cells_to_adjacent_room(floorplan, room_id) # todo see if it works
        return floorplan



    def create_new_generation(self, population_size, mutation_rate, reqs, num_rooms=8):
        new_population = []
        parents = self.select_parents(population_size)
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            GridDrawer.color_cells_by_value(parent1[0], './', 'parent1', True, True)
            GridDrawer.color_cells_by_value(parent2[0], './', 'parent2', True, True)

            child1, exchanged_room1 = self.crossover(parent1, parent2,reqs, num_rooms)
            child2, exchanged_room2 = self.crossover(parent2, parent1,reqs, num_rooms)
            child1_text = f'child1: exchanged_room = {exchanged_room1}, fitness = {child1[1].fitness}'
            child2_text = f'child2:  exchanged_room = {exchanged_room2}, fitness = {child2[1].fitness}'
            GridDrawer.color_cells_by_value(child1[0], './', child1_text, True, True)
            GridDrawer.color_cells_by_value(child2[0], './', child2_text, True, True)
            child1 = self.mutate(child1[0], mutation_rate)
            child2 = self.mutate(child2[0], mutation_rate)
            new_population.append(child1)
            new_population.append(child2)

        return new_population

    # info Crossover
    # todo child가 invalid 하면 통과
    def crossover(self, floorplan_fit1, floorplan_fit2, reqs, num_rooms = 8):
        """
        두 플로어플랜을 교차하여 자식 플로어플랜을 생성한다.

        Parameters:
        - floorplan1: tuple of (2D numpy array, fitness) 첫 번째 부모 플로어플랜
        - floorplan2: tuple of (2D numpy array, fitness) 두 번째 부모 플로어플랜

        Returns:
        - child_floorplan: 2D numpy array, 생성된 자식 플로어플랜
        """
        parent1, parent2 = floorplan_fit1[0], floorplan_fit2[0]
        parent1_fit, parent2_fit = floorplan_fit1[1], floorplan_fit2[1]

        overlapped_rooms = overlapped_room_list(parent1, parent2, num_rooms)
        while overlapped_rooms:
            selected_room = overlapped_rooms.pop()
            child1 = self.combine_floorplans(parent1, parent2, selected_room)  # overlapped가 큰 것은 없어지는 방이 생김

            # 두 덩어리로 나뉘어진 방이 있는지 확인
            if is_room_split(child1, selected_room):
                print(f'Room {selected_room} is split after crossover, skipping this configuration.')
                continue  # 덩어리가 나뉘어졌으면, 다음 방 선택으로 넘어감

            grid_polygon1 = GridPolygon(child1)
            child1_fitness = Fitness(child1, num_rooms, grid_polygon1.room_polygons, reqs)

            if child1_fitness.fitness < parent1_fit.fitness:
                return (child1, child1_fitness), selected_room
            print(f'child1 {child1_fitness.fitness} <= parent1 {parent1_fit.fitness}, skipping this configuration.')

        return floorplan_fit1, None

    # 이렇게 하면 현재 parent1과 같은 걸 만들어냄
    def combine_floorplans(self, parent1, parent2, selected_room):
        # combined = np.where(parent2 == selected_room, parent2, parent1)  # parent1에서 선택한 방을 유지, 나머지는 parent2에서
        # 교차 영역 설정
        """
        두 부모의 교차된 영역만 선택된 방으로 만들고,
        나머지 빈 공간은 인접 룸 중 하나로 채운다.
        """
        # 교차 영역(intersection) 설정: 두 부모에서 방 번호가 동일한 교차 영역만 선택된 방으로 설정 => 이 경우 child는
        intersection = (parent1 == selected_room) & (parent2 == selected_room)
        child = np.where(intersection, selected_room, parent1)

        # 유니온 영역 설정: 교차되지 않은 유니온 영역을 이웃 방 번호로 채움
        union = (parent1 == selected_room) | (parent2 == selected_room)
        union_minus_intersection = union & ~intersection

        child = self.fill_union_area(child, union_minus_intersection, parent1, parent2)
        isequal = np.array_equal(child, parent1)
        print(f'child, parent1 is equal?{isequal}')

        # child = self.fill_empty_cells(child)
        # merged = self.find_shortest_path_and_merge(child, selected_room)
        # return merged
        return child

    def fill_union_area(self, child, union_area, parent1, parent2):
        """
        유니온 영역을 인접한 방 중 하나로 채운다.
        """
        rows, cols = child.shape
        for i in range(rows):
            for j in range(cols):
                if union_area[i, j] and child[i, j] == 0:  # 유니온 영역 중 빈 셀 찾기
                    neighbors = self.get_neighbors(child, i, j)
                    if neighbors:
                        child[i, j] = neighbors[0]  # 첫 번째 이웃으로 채우기
        return child


    # 빈 셀(0)을 인접한 셀의 방 번호로 채우기
    def fill_empty_cells(self, floorplan):
        filled_floorplan = floorplan.copy()
        rows, cols = filled_floorplan.shape
        for i in range(rows):
            for j in range(cols):
                if filled_floorplan[i, j] == 0:  # 빈 셀인 경우
                    neighbors = self.get_neighbors(filled_floorplan, i, j)
                    if neighbors:
                        filled_floorplan[i, j] = random.choice(neighbors)  # 인접한 방 번호 중 하나로 채움
        return filled_floorplan


    # 두 덩어리 사이의 최단 경로를 찾아서 병합하는 함수 # todo  두 덩어리인지 확인하기 위해
    def find_shortest_path_and_merge(self, floorplan, room_number):
        binary_floorplan = (floorplan == room_number).astype(int)
        labeled_array, num_features = label(binary_floorplan)

        if num_features <= 1:
            return floorplan

        # 각 덩어리의 좌표들 가져오기
        component_coords = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]

        # 가장 큰 덩어리 선택
        largest_component = max(component_coords, key=len)
        largest_x_min, largest_x_max = np.min(largest_component[:, 0]), np.max(largest_component[:, 0])
        largest_y_min, largest_y_max = np.min(largest_component[:, 1]), np.max(largest_component[:, 1])

        # 작은 덩어리들을 큰 덩어리에 병합
        for small_component in component_coords:
            if np.array_equal(small_component, largest_component):
                continue

            small_x_min, small_x_max = np.min(small_component[:, 0]), np.max(small_component[:, 0])
            small_y_min, small_y_max = np.min(small_component[:, 1]), np.max(small_component[:, 1])

            # X축과 Y축의 최소 거리 계산
            if small_x_max < largest_x_min:
                x_dist = largest_x_min - small_x_max
                closest_x_large = largest_x_min
                closest_x_small = small_x_max
            elif largest_x_max < small_x_min:
                x_dist = small_x_min - largest_x_max
                closest_x_large = largest_x_max
                closest_x_small = small_x_min
            else:
                x_dist = 0
                closest_x_large = closest_x_small = max(small_x_min, largest_x_min)

            if small_y_max < largest_y_min:
                y_dist = largest_y_min - small_y_max
                closest_y_large = largest_y_min
                closest_y_small = small_y_max
            elif largest_y_max < small_y_min:
                y_dist = small_y_min - largest_y_max
                closest_y_large = largest_y_max
                closest_y_small = small_y_min
            else:
                y_dist = 0
                closest_y_large = closest_y_small = max(small_y_min, largest_y_min)

            # 두 덩어리 사이의 경로 채우기
            path = self.draw_path((closest_x_small, closest_y_small), (closest_x_large, closest_y_large))
            for (x, y) in path:
                floorplan[x, y] = room_number

        return floorplan

    # 인접한 셀들의 방 번호를 가져오는 함수
    def get_neighbors(self, floorplan, i, j):
        neighbors = []
        if i > 0 and floorplan[i - 1, j] != 0:
            neighbors.append(floorplan[i - 1, j])
        if i < floorplan.shape[0] - 1 and floorplan[i + 1, j] != 0:
            neighbors.append(floorplan[i + 1, j])
        if j > 0 and floorplan[i, j - 1] != 0:
            neighbors.append(floorplan[i, j - 1])
        if j < floorplan.shape[1] - 1 and floorplan[i, j + 1] != 0:
            neighbors.append(floorplan[i, j + 1])
        return neighbors


    # 두 좌표 사이의 직선 경로를 그리는 함수 (동서남북 방향으로만 이동)
    def draw_path(self, start, end):
        path = []
        x1, y1 = start
        x2, y2 = end

        # x 방향으로 먼저 이동
        while x1 != x2:
            if x1 < x2:
                x1 += 1
            elif x1 > x2:
                x1 -= 1
            path.append((x1, y1))

        # 그 다음 y 방향으로 이동
        while y1 != y2:
            if y1 < y2:
                y1 += 1
            elif y1 > y2:
                y1 -= 1
            path.append((x1, y1))

        return path

    # info end of crossover functon
    def compute_pareto_front(self):
        pareto_front = []
        for i, solution_a in enumerate(self.population):
            dominated = False
            for j, solution_b in enumerate(self.population):
                if i != j and is_dominated(solution_a, solution_b):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(solution_a)
        return pareto_front

    # info for crossover dont't delete it might need for some operation
    # 겹치는 좌표 구하기
def find_common_coordinates(parent1, parent2):
    common_coords = {}
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if parent1[i, j] == parent2[i, j] and parent1[i, j] != -1:  # 같은 방이고 -1이 아닐 때
                room = parent1[i, j]
                if room not in common_coords:
                    common_coords[room] = []
                common_coords[room].append((i, j))
    return common_coords

# 겹치는 좌표를 2차원 배열에 표시
def mark_common_coordinates(parent1, common_coords) :
    result_array = np.zeros_like(parent1)  # 동일한 크기의 0으로 채워진 배열 생성
    for room, coords in common_coords.items():
        for (i, j) in coords:
            result_array[i, j] = room  # 겹치는 좌표에 방 값을 표시
    return result_array

def select_most_overlapped_room(room1, room2, num_rooms = 8):
    common_coords = find_common_coordinates(room1, room2) # 각 방별 두 플랜간의 공통 영역
    common_fl = mark_common_coordinates(room1, common_coords) #2D Array로 변환
    # GridDrawer.color_cells_by_value(common_fl, './', 'common_fl', True, True, 8) # todo to delete for debug
    room_areas = calc_room_cell_areas(common_fl, num_rooms)
    return  max(room_areas, key=room_areas.get), min(room_areas, key=room_areas.get)

def overlapped_room_list(room1, room2, num_rooms = 8):
    common_coords = find_common_coordinates(room1, room2) # 각 방별 두 플랜간의 공통 영역
    common_fl = mark_common_coordinates(room1, common_coords) #2D Array로 변환
    room_areas = calc_room_cell_areas(common_fl, num_rooms)
    sorted_rooms_id = [room for room, area in sorted(room_areas.items(), key=lambda item: item[1], reverse=True) if area > 0]

    return sorted_rooms_id
def calc_room_cell_areas(room, num_rooms): # info cell 단위로 면적 구함
    return  {room_id: np.sum(room == room_id) for room_id in range(1, num_rooms + 1)}
