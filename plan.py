import random
import sys

import networkx as nx
from scipy.sparse import lil_matrix
import trivial_utils
import numpy as np
import plan_utils
from GridDrawer import GridDrawer
import itertools
from collections import defaultdict
import random
from measure import categorize_boundary_cells
from config_reader import read_constraint, read_config_boolean, check_adjacent_requirement, read_config_int
from cells_utils import is_valid_cell, is_inside
from plan_utils import dict_value_to_coordinates
from options import Options
from reqs import Req
from scipy.ndimage import label

# todo 1: 사이즈 작은 사이즈일 수록 빈 인접셀 적다. 이를 이용해서 초기화에 활용
# todo 2: 인접 리스트를 만들어서 해당 조건 만족하도록
# todo 6: Graph의 구조가 같은지를 체크할 수 있도록 GraphGrid에 equality function들을 만들었다. 이를 활용해서 인접성 리스트를 optimize 하자
# todo 3: 10개를  Seed에서 하는데 확인. 이 10개의 fitness가 유사하다. 그러므로 당분간? 10개씩 하지 말고? 다른 seed로 해보자.
# todo 4: 분리
# info : adj_requirements를 추가했으나 성과가 없다. graph를 비교해서 방을 바꾸는 방법으로 가는 것도 방법
# todo 0828: 일단 seed를 출력하고 fitness를 보자
# todo 0905 option 컨트롤을 다 중앙으로 옮겨야 돼 여기저기 뒤죽박죽

def locate_initial_cell(empty_grid, k):
    options = Options()
    ini_filename = 'constraints.ini'
    adjacency_list = check_adjacent_requirement()
    orientation_requirements = read_constraint(ini_filename, 'OrientationRequirements')

    initialized_grid, initial_cells = place_seed(to_np_array(empty_grid), k,
                                                 adjacency_list)  # info now seed does not have room number
    display_process(initialized_grid, k, options, "Seed0")
    initialized_grid, remaining_positions, assigned_seed_by = assign_room_id(initialized_grid, initial_cells,
                                                           orientation_requirements,
                                                           adjacency_list) # info seed now has the room_id
    display_process(initialized_grid, k, options, "Seed2")
    return initialized_grid, initial_cells, assigned_seed_by

# info 방 개수만큼 랜덤하게  seed 셀을 선택해서 255로 채움
def place_seed(grid_arr, k, adjacency_graph=None):
    # Randomly place k rooms within the floorplan
    rooms_placed = 0
    cells_coords = set()
    rooming_grid = grid_arr.copy()
    m, n = grid_arr.shape

    while rooms_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if rooming_grid[row, col] == 0:  # Ensure the cell is within the floorplan and unroomed
            rooms_placed += 1
            rooming_grid[row, col] = 255  # 방을 표시하기 위해 값을 1로 설정
            cells_coords.add((row, col))  # 셀의 위치를 저장

    return rooming_grid, cells_coords



# 랜덤하게 방향요구사항 인접 요구사항을 선택하여 seed에 room id 할당
def assign_room_id(grid, cell_positions, orientation_requirements, adjacency_requirements):
    assigned_seed_by=set()
    # 1. 방향 요구 사항에 따라 방 배치
    new_grid = grid.copy()
    if random.choice([True, False]):
        new_grid = assign_by_orientation(grid, cell_positions, orientation_requirements)
        assigned_seed_by.add('orientation')
    # 2. 인접성 요구 사항에 따라 방 배치
    if random.choice([True, False]):
        new_grid = assign_by_adjacency(new_grid, cell_positions, adjacency_requirements)
        assigned_seed_by.add('adjacency')

    # 3. 빈 방셀을 모두 무작위로 assign
    new_grid = assign_undecided_roomid_for_initial_cells(new_grid, cell_positions)

    return new_grid, cell_positions, assigned_seed_by


# info 1: by orientation
def assign_by_orientation(grid, cell_positions, orientation_requirements):
    new_grid = np.copy(grid)
    scores = get_orientation_scores(cell_positions)

    for room_num, direction in orientation_requirements.items():
        pos = scores[direction].pop()
        delete_cell_from(scores, pos)
        new_grid[pos] = room_num

    return new_grid

def get_orientation_scores(cell_positions):
    north = sorted(cell_positions, key=lambda pos: pos[0], reverse=True)
    west = sorted(cell_positions, key=lambda pos: pos[1], reverse=True)

    scores = {
        'north': north,
        'south': north[::-1],
        'west': west,
        'east': west[::-1]
    }
    return scores


def delete_cell_from(scores, cell):
    for direction, cell_list in scores.items():
        if cell in cell_list:
            scores[direction].remove(cell)
    return scores


# info 2: by adjacency
def assign_by_adjacency(grid, cell_positions, adjacency_requirements):
    new_grid = np.copy(grid)

    assigned_room_positions = get_assigned_cell_positions(grid)
    distances = calculate_sorted_distances(cell_positions)
    adj_dict = defaultdict(list)

    for a, b in adjacency_requirements:
        adj_dict[a].append(b)
        adj_dict[b].append(a)

    for room_id, adj_list in adj_dict.items():
        if set(assigned_room_positions.keys()) == set(range(1, len(cell_positions) + 1)):
            break

        if not set(adj_list).issubset(assigned_room_positions):
            unassigned_pos = set(distances.keys()) - set(assigned_room_positions.values())
            if unassigned_pos:
                room_pos = unassigned_pos.pop()
                unassigned_pos_distances = [
                    room_dist for room_dist in distances[room_pos]
                    if room_dist[0] not in assigned_room_positions.values()
                ]

                for adj_id in adj_list:
                    if adj_id not in assigned_room_positions and unassigned_pos_distances:
                        adj_pos, _ = unassigned_pos_distances.pop()
                        new_grid[adj_pos] = adj_id
                        assigned_room_positions[adj_id] = adj_pos

        elif room_id in assigned_room_positions:
            room_pos = assigned_room_positions[room_id]
            unassigned_pos_distances = [
                room_dist for room_dist in distances[room_pos]
                if room_dist[0] not in assigned_room_positions.values()
            ]
            for adj_id in adj_list:
                if adj_id not in assigned_room_positions and unassigned_pos_distances:
                    adj_pos, _ = unassigned_pos_distances.pop()
                    new_grid[adj_pos] = adj_id
                    assigned_room_positions[adj_id] = adj_pos

    return new_grid


def get_assigned_cell_positions(grid):
    return {grid[tuple(cell)]: tuple(cell) for cell in np.argwhere((grid > 0) & (grid != 255))}  # room_id : position


# 각 셀 사이의 거리를 구한 후
def calculate_sorted_distances(cell_positions):
    distances = {}
    for pos1 in cell_positions:
        distances[pos1] = sorted(
            [(pos2, abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])) for pos2 in cell_positions if pos1 != pos2],
            key=lambda x: x[1],
            reverse=True)
    return distances


# info 3 아직 방 배정이 안된 초기셀이 있으면 방배정을 한다.
def assign_undecided_roomid_for_initial_cells(grid, cell_positions):
    assigned_room_positions = get_assigned_cell_positions(grid)
    unassigned_pos_set = set(cell_positions) - set(
        assigned_room_positions.values())  # underway 0904 to see  when to set cell_positions
    all_room_ids = set(range(1, len(cell_positions) + 1))
    unassigned_id_set = all_room_ids - set(assigned_room_positions.keys())

    for pos, room_id in zip(unassigned_pos_set, unassigned_id_set):
        grid[pos] = room_id
    return grid


############################################################
# Utility Function
############################################################
def display_process(initialized_grid, k, options, prefix, postfix=None):
    if options is None:
        options = Options()
    if options.display is False and options.save is False:
        return
    # initialized_grid, initial_cells = place_k_rooms_on_grid(to_np_array(empty_grid), k) # todo place_seed에서 그래프 만족시키는 seed 새로 만듦
    path = trivial_utils.create_folder_by_datetime()
    full_path = trivial_utils.create_filename(path, prefix, postfix = str(postfix), filename='', ext='png')
    GridDrawer.color_cells_by_value(initialized_grid, full_path, display=options.display, save=options.save,
                                    num_rooms=k)




def find_parallel_adjacent_cells(floorplan, room_id):
    """
    주어진 방 번호에 대해 수평 또는 수직 방향으로 일직선상에 있는 셀들의 좌표를 찾고,
    이 셀들이 다른 방과 인접해 있는지 확인하여 결과를 반환한다.

    Parameters:
    - floorplan: 2D numpy array, 플로어플랜의 배열 (각 셀은 방 번호를 가짐)
    - room_id: int, 찾고자 하는 방의 번호

    Returns:
    - result: list of tuples, 각 튜플은 (셀 좌표 리스트, 방향)으로 구성
    """
    result = []
    rows, cols = floorplan.shape

    # 수평 방향으로 일직선상에 있는 셀들 확인
    for i in range(rows):
        row_coords = [(i, j) for j in range(cols) if floorplan[i, j] == room_id]  # 수평 방향으로 일직선상에 있는 셀들
        if len(row_coords) >= 2:
            # 이 셀들이 상하 방향으로 다른 방과 인접해 있는지 확인
            for x, y in row_coords:
                if x > 0 and floorplan[x - 1, y] != room_id and floorplan[x - 1, y] != 0:
                    result.append((row_coords, 'up'))
                    break  # 중복 추가 방지
                if x < rows - 1 and floorplan[x + 1, y] != room_id and floorplan[x + 1, y] != 0:
                    result.append((row_coords, 'down'))
                    break  # 중복 추가 방지

    # 수직 방향으로 일직선상에 있는 셀들 확인
    for j in range(cols):
        col_coords = [(i, j) for i in range(rows) if floorplan[i, j] == room_id]  # 수직 방향으로 일직선상에 있는 셀들
        if len(col_coords) >= 2:
            # 이 셀들이 좌우 방향으로 다른 방과 인접해 있는지 확인
            for x, y in col_coords:
                if y > 0 and floorplan[x, y - 1] != room_id and floorplan[x, y - 1] != 0:
                    result.append((col_coords, 'left'))
                    break  # 중복 추가 방지
                if y < cols - 1 and floorplan[x, y + 1] != room_id and floorplan[x, y + 1] != 0:
                    result.append((col_coords, 'right'))
                    break  # 중복 추가 방지

    return result


def assign_cells_to_adjacent_room(floorplan, room_id):
    """
    주어진 방의 셀들을 인접한 다른 방의 번호로 할당한다.
    이때, 인접한 다른 방이 있는 방향 중 한 방향만 임의로 선택하여 할당한다.

    Parameters:
    - floorplan: 2D numpy array, 플로어플랜의 배열 (각 셀은 방 번호를 가짐)
    - room_id: int, 할당할 셀들의 방 번호

    Returns:
    - modified_floorplan: 할당된 후의 플로어플랜 배열
    """
    adjacent_cells = find_parallel_adjacent_cells(floorplan, room_id)

    if adjacent_cells:
        # 인접 방향 중 하나를 임의로 선택
        cells, direction = random.choice(adjacent_cells)

        for (x, y) in cells:
            if direction == 'up' and x > 0 and floorplan[x - 1, y] != 0:
                floorplan[x, y] = floorplan[x - 1, y]
            elif direction == 'down' and x < floorplan.shape[0] - 1 and floorplan[x + 1, y] != 0:
                floorplan[x, y] = floorplan[x + 1, y]
            elif direction == 'left' and y > 0 and floorplan[x, y - 1] != 0:
                floorplan[x, y] = floorplan[x, y - 1]
            elif direction == 'right' and y < floorplan.shape[1] - 1 and floorplan[x, y + 1] != 0:
                floorplan[x, y] = floorplan[x, y + 1]

    return floorplan

###########################################################
# info functions that is not used but maybe useful later
##########################################################
# 주어진 adjacency_list로 nx Graph를 빌드
def create_req_graph(adjacency_list):
    adjacency_graph = nx.Graph()
    adjacency_graph.add_edges_from(adjacency_list)
    if not nx.check_planarity(adjacency_graph):
        return False
    return adjacency_graph


def calculate_manhattan_distances(cell_positions):
    distances = []

    # 리스트로 변환하여 인덱스로 접근할 수 있도록 함
    cell_positions_list = list(cell_positions)

    for i in range(len(cell_positions_list)):
        for j in range(i + 1, len(cell_positions_list)):
            pos1 = cell_positions_list[i]
            pos2 = cell_positions_list[j]
            manhattan_distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            distances.append(((pos1, pos2), manhattan_distance))

    return distances


def calculate_relative_scores(cell_positions):
    scores = {pos: {'north': 0, 'south': 0, 'west': 0, 'east': 0} for pos in cell_positions}

    # 북-남 방향 정렬 (북쪽이 먼저 오도록 정렬)
    sorted_by_north_south = sorted(cell_positions, key=lambda x: x[0])
    # 동-서 방향 정렬 (서쪽이 먼저 오도록 정렬)
    sorted_by_east_west = sorted(cell_positions, key=lambda x: x[1])

    # 북쪽과 남쪽 점수 계산
    for i, pos in enumerate(sorted_by_north_south):
        scores[pos]['north'] = i  # 북쪽 점수는 인덱스가 작을수록 높음
        scores[pos]['south'] = len(sorted_by_north_south) - 1 - i  # 남쪽 점수는 반대 방향으로

    # 서쪽과 동쪽 점수 계산
    for i, pos in enumerate(sorted_by_east_west):
        scores[pos]['west'] = i  # 서쪽 점수는 인덱스가 작을수록 높음
        scores[pos]['east'] = len(sorted_by_east_west) - 1 - i  # 동쪽 점수는 반대 방향으로

    return scores

#######################################################
####  info allocate room
#####################################################

def create_floorplan(initialized_grid, k, options, reqs=None):
    if reqs is None:
        reqs = Req()
    grid_copy = initialized_grid.copy()
    display_process(grid_copy, k=k, options=options, prefix='Init0')  # info just save and display on pycharm

    if options.min_size_alloc is True:
        floorplan = allocate_room_with_size(grid_copy, options.display, save=options.save, num_rooms=k, reqs=reqs)
    else:
        floorplan = allocate_rooms(grid_copy, display=options.display, save=options.save, num_rooms=k)

    return floorplan



# info adj와 orientation Requirement는 place_seed에서 처리, size는 allocate_room_with_size 에서 처리
#  structure App.create_floorplan_from_seed > Floorplan.iterate_optimal_floorplans
#  > plan.create_floorplan > allocate_rooms_with_size
def allocate_room_with_size(floorplan, display=False, save=True, num_rooms=8, reqs=None):
    print(f'allocate_room_with_size')
    def choose_new_adjacent_cell(floorplan, cell):
        row, col = cell
        rows, cols = floorplan.shape
        if floorplan[row, col] > 0:
            valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                             if 0 <= row + dy < rows and 0 <= col + dx < cols and floorplan[row + dy, col + dx] == 0]
            if valid_offsets:
                dr = random.choice(valid_offsets)
                return (row + dr[0], col + dr[1]), dr
        return None, (0, 0)

    def update_active_cells(floorplan, cell, active_cells):
        if not has_empty_neighbor(floorplan, cell[0], cell[1]):
            if cell in active_cells:
                active_cells.remove(cell)
        else:
            active_cells.add(cell)

        for adj_cell in all_active_neighbors(cell, floorplan):
            if not len(get_unassigned_neighbor_set(adj_cell, floorplan)) > 0:
                if adj_cell in active_cells:
                    active_cells.remove(adj_cell)

    def is_inside(floorplan, coord):
        """Check if the given coordinate is inside the bounds of the floorplan."""
        rows, cols = floorplan.shape
        return 0 <= coord[0] < rows and 0 <= coord[1] < cols

    active_cells = process_valid_cells(floorplan)
    current_step = 0

    # 방의 현재 면적을 추적하기 위한 딕셔너리
    room_areas = {room_id: np.sum(floorplan == room_id) for room_id in range(1, num_rooms + 1)}
    max_area_exceeded = set()  # 최대 면적을 초과한 방의 번호를 저장할 집합
    expandable_rooms = set(range(1, num_rooms + 1))  # 확장 가능한 방을 추적하기 위한 집합
    num_iter = 0
    while active_cells and expandable_rooms:
        if len(expandable_rooms) == 1:
            room_id = expandable_rooms.pop()
            if not is_room_split(floorplan, room_id):  # 한 덩어리인데, assign할 방이 하나 남아있다는 소리
                fill_empty_cell_with_value(floorplan, room_id)  # 나머지 모든 방을 다 assign하고 floorplan을 리턴
                return floorplan
        num_iter += 1
        if len(expandable_rooms) == 1:
            room_id = expandable_rooms.pop()
            if not is_room_split(floorplan, room_id): # 한 덩어리인데, assign할 방이 하나 남아있다는 소리
                fill_empty_cell_with_value(floorplan, room_id) # 나머지 모든 방을 다 assign하고 floorplan을 리턴
                return floorplan

        # print(f'iter={num_iter} Active cells length: {len(active_cells)}') info to debug to avoid infinite loop
        room_to_coordinates = dict_value_to_coordinates(floorplan)
        room_numbers = [room for room in expandable_rooms if room not in max_area_exceeded] # todo 그 때 그 때 저장하는 게 어때? 매번 하지 않도록 변경

        if not room_numbers:
            break  # 모든 방이 최대 면적을 초과하면 루프를 종료

        room_idx = random.choice(room_numbers)
        room = room_idx # todo 쓸데없는 중복
        all_cells = room_to_coordinates.get(room, [])
        current_active_cells = [c for c in all_cells if c in active_cells]

        if not current_active_cells:
            expandable_rooms.remove(room)  # 이 방은 확장할 수 없으므로 expandable_rooms에서 제거
            continue

        cell = random.choice(current_active_cells)

        # 현재 방의 면적이 최대 면적을 초과했는지 확인
        if reqs is None:
            reqs = Req()
        min_area, max_area = reqs.get_area_range(room)
        if max_area is not None and room_areas[room] >= max_area:
            max_area_exceeded.add(room)  # 방을 더 이상 확장하지 않도록 집합에 추가
            expandable_rooms.remove(room)  # 이 방은 더 이상 확장할 수 없으므로 expandable_rooms에서 제거
            continue

        new_cell, dr = choose_new_adjacent_cell(floorplan, cell)
        if new_cell is None:
            expandable_rooms.remove(room)  # 새로운 셀이 없으면 확장 불가능하므로 제거
            continue

        # 새로운 셀을 추가하고 면적 업데이트
        floorplan[new_cell] = floorplan[cell]
        room_areas[room] += 1  # 셀이 추가되었으므로 면적 증가
        update_active_cells(floorplan, cell, active_cells)
        update_active_cells(floorplan, new_cell, active_cells)

        # is_inside 함수로 경계 확인 후 색칠
        parallel_cells = [tuple(np.add(c, dr)) for c in all_cells if
                          c != cell and is_inside(floorplan, tuple(np.add(c, dr))) and floorplan[
                              tuple(np.add(c, dr))] == 0]

        for c in parallel_cells:
            floorplan[c] = floorplan[cell]
            room_areas[room] += 1  # 셀이 추가되었으므로 면적 증가
            update_active_cells(floorplan, c, active_cells)

        # filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step) # info to see process take comment off
        # GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms) # info to see process take comment off

    # 빈 셀 채우기: 할당되지 않은 셀에 인접한 셀의 방 번호를 할당
    # fill_empty_cells(floorplan) # underway 0904to see if it works
    floorplan = fill_unassigned_cells(floorplan)
    # filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step) # info to see process take comment off
    # GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)# info to see process take comment off
    return floorplan

def fill_empty_cell_with_value(floorplan, room_id):
    # 빈 셀 채우기: 할당되지 않은 셀에 인접한 셀의 방 번호를 할당
    empty_cells = {tuple(x) for x in np.argwhere(floorplan == 0)}  # 빈 셀들의 좌표 리스트
    if len(empty_cells) == 0:
        return floorplan

    # 빈 셀에 방 번호 배정, Fancy Indexing 사용
    x, y = zip(*empty_cells)
    floorplan[(x, y)] = room_id
    return floorplan



def is_room_split(floorplan, room_number):
    """
    주어진 방 번호의 덩어리가 여러 개로 나뉘어졌는지 확인하는 함수.

    Parameters:
    - floorplan: 2D numpy array, 플로어플랜
    - room_number: int, 확인할 방 번호

    Returns:
    - bool: 방이 여러 덩어리로 나뉘어진 경우 True, 아니면 False
    """
    binary_floorplan = (floorplan == room_number).astype(int)
    labeled_array, num_features = label(binary_floorplan)
    return num_features > 1


def allocate_rooms(floorplan, display=False, save=True, num_rooms=8):
    print(f'allocate_room')

    def choose_new_adjacent_cell(floorplan, cell):
        row, col = cell
        rows, cols = floorplan.shape
        # 셀 주변의 유효한 빈 셀 찾기
        valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if 0 <= row + dy < rows and 0 <= col + dx < cols and floorplan[row + dy, col + dx] == 0]
        if valid_offsets:
            dr = random.choice(valid_offsets)
            return (row + dr[0], col + dr[1]), dr
        return None, (0, 0)

    def update_active_cells(floorplan, cell, active_cells):
        # 셀에 인접한 빈 셀이 없으면 active_cells에서 제거
        if not has_empty_neighbor(floorplan, cell[0], cell[1]):
            active_cells.discard(cell)
        else:
            active_cells.add(cell)

        # 인접한 셀들에 대해서도 같은 작업 수행
        neighbors = all_active_neighbors(cell, floorplan)
        for adj_cell in neighbors:
            if not has_empty_neighbor(floorplan, adj_cell[0], adj_cell[1]):
                active_cells.discard(adj_cell)
            else:
                active_cells.add(adj_cell)

    options = Options()
    active_cells = process_valid_cells(floorplan)
    current_step = 0

    room_numbers = set(range(1, num_rooms + 1))  # 할당된 방 번호들
    while active_cells:  # active_cells가 남아있는 동안 반복
        room_to_coordinates = dict_value_to_coordinates(floorplan)

        for room in list(room_numbers):  # 할당된 방들에 대해 반복
            all_cells = room_to_coordinates.get(room, [])
            current_active_cells = [c for c in all_cells if c in active_cells]

            if not current_active_cells:
                room_numbers.discard(room)  # 더 이상 확장할 수 없는 방 제거
                continue

            # 활성 셀 중 하나를 선택하여 새로운 셀을 할당
            cell = random.choice(current_active_cells)
            new_cell, dr = choose_new_adjacent_cell(floorplan, cell)

            if new_cell is None:
                continue

            # 새로운 셀에 방 번호를 할당하고 active_cells 업데이트
            floorplan[new_cell] = floorplan[cell]
            update_active_cells(floorplan, cell, active_cells)
            update_active_cells(floorplan, new_cell, active_cells)

            # 병렬로 확장 가능한 셀들도 할당
            parallel_cells = [tuple(np.add(c, dr)) for c in all_cells
                              if c != cell and floorplan[tuple(np.add(c, dr))] == 0 and is_inside(floorplan,
                                                                                                  tuple(np.add(c, dr)))]

            for c in parallel_cells:
                floorplan[c] = floorplan[cell]
                update_active_cells(floorplan, c, active_cells)

        # 과정 중간에 floorplan을 보여주는 함수 호출 (display 설정에 따라)
        display_process(floorplan, num_rooms, options, prefix='Step', postfix=current_step)
        current_step += 1  # 다음 스텝으로 증가

    return floorplan


###################################
# info utilities
###################################


def all_active_neighbors(cell, floorplan):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell
    adjs = set()
    for dx, dy in directions:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1]:  # 범위내에 있으면
            if floorplan[new_row, new_col] > 0:  # 빈 셀은 제외
                adjs.add((new_row, new_col))
    return adjs



# 범위 내에 있고 옆에 하나라도 빈 셀이 있으면 True
def has_empty_neighbor(grid_assigning, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False

# 빈 이웃이 하나라도 있으면 그 이웃을 valid_neighbor_set에 추가해서 이를 리턴한다.
def get_unassigned_neighbor_set(cell, grid_assigning):
    # print(f'\t\tcollect_valid_adjacent_cells: {cell}')
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell[0], cell[1]
    valid_neighbor_cells = set()
    for dx, dy in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:  # 범위내에 있으면
            # print(f'\t\t\t({new_row}, {new_col}) inside')
            # if vacant(new_row, new_col, grid_assigning):
            if grid_assigning[new_row, new_col] == 0:  # vacant
                #    print(f'\t\t\t\tgrid_assigning[{new_row}, {new_col}]={grid_assigning[new_row, new_col]}.. so vacant.. adding neighbor_cells')
                neighbor_cell = (new_row, new_col)
                valid_neighbor_cells.add(neighbor_cell)
    # print(f'\t\tcollect_candidate_set:{cell} returning valid_neighbor_cells{valid_neighbor_cells}')
    return valid_neighbor_cells

# 아직 할당되지 않은 빈 이웃이 있는 모든 할당된 셀 구하기
def process_valid_cells(grid_assigning, insulated_cells=None, row_range=None):  # 두 개의 parameter가 필요없어서 기본값을 None으로 주었음
    valid_cells = set()
    #    print(f'insulated_cells={insulated_cells} in process_valid_cells')
    for row in range(grid_assigning.shape[0]):
        for col in range(grid_assigning.shape[1]):
            # if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0:
                if has_empty_neighbor(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells


def fill_unassigned_cells(floorplan):
    """
    빈 셀이 남아있을 때까지 반복해서 빈 셀을 인접한 방 번호로 채웁니다.
    """
    while True:
        empty_cells = [tuple(x) for x in np.argwhere(floorplan == 0)]  # 빈 셀들의 좌표 리스트
        if not empty_cells:
            break  # 빈 셀이 없으면 종료

        any_cell_filled = False  # 빈 셀이 채워졌는지 확인하는 플래그

        for cell in empty_cells:
            # 빈 셀에 인접한 방 번호들 추출
            neighbors_rooms = all_active_neighbors(cell, floorplan)

            if neighbors_rooms:  # 인접한 방 번호가 있는 경우
                selected_adj_cell = random.choice(list(neighbors_rooms))  # 무작위로 인접한 방 좌표 선택
                floorplan[cell] = floorplan[selected_adj_cell]  # 해당 빈 셀을 선택한 방 번호로 채움
                any_cell_filled = True  # 최소한 하나의 셀이 채워졌음을 기록
            else:
                print(f"No adjacent room for cell {cell}, skipping.")

        # 더 이상 채울 수 있는 셀이 없으면 루프 종료
        if not any_cell_filled:  # empty_cell이 비어서 iteration을 하지 못했거나, valid한 인접 방이 없어서 아무것도 못한 경우
            print("No more cells can be filled.")
            break

    return floorplan



# todo 2/3 이상 진행되었을 때 이걸 체크해서 한꺼번에 채우자
def all_identical(cells, floorplan):
    # Get the value at the first cell's coordinates
    first_value = floorplan[cells[0]]

    # Check if all other cells have the same value
    for cell in cells[1:]:
        if floorplan[cell] != first_value:
            return False
    return True


def get_unique_values(floorplan, cells):
    unique_values = set()
    for cell in cells:
        unique_values.add(floorplan[cell])
    return unique_values




# 색칠하지 말고 방향과 행렬 번호만 리턴하자.
def parallel_extention2(floorplan, obtainable_cells, display=False, save=True, num_rooms=7):
    # todo 여기서는 그냥 choose만 하고, return 값은 새 셀의 좌표와 방향
    def choose_new_adjacent_cell(floorplan, cell):
        row, col = cell
        rows, cols = floorplan.shape[0], floorplan.shape[1]
        # Ensure we are within the floorplan and the cell has not been colored yet
        if floorplan[row, col] > 0:  # Adjusted condition to ensure we're targeting uncolored cells

            valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                             if 0 <= row + dy < rows and 0 <= col + dx < cols and floorplan[row + dy, col + dx] == 0]

            if valid_offsets:
                dr = random.choice(valid_offsets)
                dy, dx = dr[0], dr[1]
                # floorplan[row + dy, col + dx] = floorplan[row, col]  # Color the neighbor
                return (row + dy, col + dx), dr
        else:
            return None, (0, 0)

    # 0인 neighbor가 없고 즉 더이상 확장할 수 없는데 그 셀이 actie_cells 리스트에 있으면 거기서 제거
    # 0인 네이버가 있으면 무조건 active cell에 추가
    def update_active_cells(floorplan, cell, active_cells):
        if not has_empty_neighbor(floorplan, cell[0], cell[1]):  # not( has_neighbor_zero = 범위 안에 있고 0인 인접셀이 하나라도 있으면)
            if cell in active_cells:  # 그 셀은 더이상 active하지 않으므로 active_cells에서 제거하고
                print(f'{cell} has not neighbor zero and in active_cells :why is this happening ')
                active_cells.remove(cell)
        else:
            active_cells.add(cell)

        # 이 셀이 더이상 확장 가능하거나 가능하지 않거나 상관없이 모든 인접셀에 대해서 다시 candidate를 구한다. if 문에서 들어가면
        for adj_cell in all_active_neighbors(cell, floorplan):
            if not len(get_unassigned_neighbor_set(adj_cell, floorplan)) > 0:  # has no candidate
                if adj_cell in active_cells:
                    active_cells.remove(adj_cell)

    insulated_cells = set()
    # todo .1 insulated_cell 을 이용하지 않았어
    # todo .2 obtainable_cells를 기껏 가져와서 process_valid_cells에서 다시 구했어. 아래 문장에서는  obtainable_cells와 insulated_cells를 모두 이용하지 않았어.
    valid_obtainable_cells = process_valid_cells(floorplan)
    current_step = 0

    while len(valid_obtainable_cells) > 0:
        room_to_coordinates = dict_value_to_coordinates(floorplan)
        # todo 각 룸마다 하나씩 패러랠하게 추가
        # todo room_to_coordinates로 for 문을 돌리면 같은 셀로 계속 반복하게 됨
        #  coordinates는 나중에 룸 번호보고 다시 선택하기로 하고, iteration은 room 위에 while 문 필요함.
        room_numbers = list(range(1, num_rooms + 1))
        while room_numbers:
            # for room, all_cells in room_to_coordinates.items():
            room_idx = random.randrange(len(room_numbers))
            room = room_numbers.pop(room_idx)
            all_cells = room_to_coordinates[room]
            active_cells = valid_obtainable_cells.copy()
            # valid한  coordsnates를 찾아라
            current_active_cells = [c for c in all_cells if c in active_cells]
            if len(current_active_cells) < 1:
                continue  # 현재 선택된 방의 모든 valid한 셀
            cell = random.choice(current_active_cells)  # todo cannot choose from an empty sequence
            # just pick one from each cell
            new_cell, dr = choose_new_adjacent_cell(floorplan,
                                                    cell)  # todo 이 버전은 값은 안변하고 셀만 구한다. 일단 간직만 하자. 변경하는 것은 나중에
            # assign_room
            # 바꾸려고 하는 셀이 원래셀이 아니고, 바꿀 대상 위치가 비어있고(0) 범위 내에 있는 것만 콜렉트
            parallel_cells = [tuple(np.add(c, dr)) for c in all_cells if
                              c is not cell and floorplan[tuple(np.add(c, dr))] == 0 and is_inside(floorplan, tuple(
                                  np.add(c, dr)))]

            # 색칠
            floorplan[new_cell] = floorplan[cell]
            update_active_cells(floorplan, cell, active_cells)  # todo active_cells를 잘봐
            update_active_cells(floorplan, new_cell, active_cells)

            #  parallel_cells를 색칠하면 됨.
            # 미리 구해놓은 parallel_cells 만 색칠
            for c in parallel_cells:
                floorplan[c] = floorplan[cell]
                update_active_cells(floorplan, c, active_cells)
                print(f'active_cells={active_cells}')

            valid_obtainable_cells = active_cells.copy()
        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    return floorplan


# commit된 것 복원 old_version. parallel_extension으로 대체
def place_room(floorplan, obtainable_cells, display=False, save=True, num_rooms=7):
    insulated_cells = set()
    # todo .1 insulated_cell 을 이용하지 않았어
    # todo .2 obtainable_cells를 기껏 가져와서 process_valid_cells에서 다시 구했어. 아래 문장에서는  obtainable_cells와 insulated_cells를 모두 이용하지 않았어.
    valid_obtainable_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
    insulated_cells = set(obtainable_cells) - valid_obtainable_cells
    current_step = 0
    while valid_obtainable_cells:
        active_cells = valid_obtainable_cells.copy()
        num_unique_value = len(get_unique_values(floorplan, active_cells))
        if num_unique_value > 1:
            for cell in valid_obtainable_cells:
                print(f'cell =  {cell} ')
                if cell in active_cells:
                    new_cell = choose_new_adjacent_cell(floorplan, cell)
                else:
                    continue  # 다음 셀을 실행한다.

                # 새 셀이 삽입되면 원래 셀, 본인. 본인의 네이버 모두 valid한지 체크해야 한다.
                print(f'\tnew_cell = {new_cell}')
                if not check_valid_current_cell(floorplan, cell):
                    if cell in active_cells:
                        active_cells.remove(cell)
                    insulated_cells.add(cell)

                if check_valid_current_cell(floorplan, new_cell):
                    active_cells.add(new_cell)
                else:  # if not valid
                    insulated_cells.add(new_cell)

                for adj_cell in all_active_neighbors(new_cell, floorplan):
                    if not len(get_unassigned_neighbor_set(adj_cell, floorplan)) > 0:  # has no candidate
                        if adj_cell in active_cells:
                            active_cells.remove(adj_cell)
                        insulated_cells.add(adj_cell)

                print(f'\tactive_cells:{active_cells}:{len(active_cells)}')

        # 모든 active_cell들의 값이 같으면 반복하지 말고 나머지 모든 셀을 그 값으로 채운다
        elif num_unique_value == 1:
            unique_value = floorplan[active_cells.pop()]
            print(f'all the active_cells having same value {unique_value}')
            floorplan[floorplan == 0] = unique_value
            active_cells = set()

        # gridname = trivial_utils.create_filename('png', 'Step')
        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

        print(
            f'==================\nobtainable_cells={valid_obtainable_cells} {len(valid_obtainable_cells)}\nfloorplan=\n{floorplan}')
        valid_obtainable_cells = active_cells.copy()
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')

    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    num_roms = read_config_int('constraint.ini')
    GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    return floorplan
#  현재 셀이 valid 한가
def check_valid_current_cell(grid_assigning, cell):
    row, col = cell[0], cell[1]
    if not has_empty_neighbor(grid_assigning, row, col):
        return False
    return True



def count_different_and_same_neighbors(floorplan, cell):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell
    current_value = floorplan[row][col]  # 수정된 부분
    different_count = 0
    same_count = 0

    for dx, dy in directions:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < len(floorplan) and 0 <= new_col < len(floorplan[0]):
            neighbor_value = floorplan[new_row][new_col]  # 수정된 부분
            if neighbor_value != current_value and neighbor_value > 0:
                different_count += 1
            elif neighbor_value == current_value:
                same_count += 1

    return different_count, same_count


# todo 이 짓은 의미기 없음 지우기는 하는데 나중에 무슨 쓸 일이 있는지 보자. 실제로 교환하는 게 같은 룸을 교환하는건데 이게 말이 되냐고
def exchange_extreme_cells(floorplan):
    cell_dict = rooms_cells(floorplan)

    # 각 방에서 가장 돌출된 셀과 가장 오목한 셀을 교환
    for room_number, cells in cell_dict.items():
        extreme_cell, concave_cell = find_extreme_and_concave_cells(floorplan, cells)
        print(f'room={room_number} extreme_cell={extreme_cell}, concave_cell = {concave_cell}')
        if extreme_cell and concave_cell:
            print(f'room={room_number} extreme_cell={extreme_cell}, concave_cell = {concave_cell} are equal')
            floorplan[extreme_cell[0], extreme_cell[1]], floorplan[concave_cell[0], concave_cell[1]] = (
                floorplan[concave_cell[0], concave_cell[1]], floorplan[extreme_cell[0], extreme_cell[1]]
            )
            print(f'floorplan[{extreme_cell[0]}, {extreme_cell[1]}], floorplan[{concave_cell[0]}, {concave_cell[1]}] = \
            {floorplan[extreme_cell[0], extreme_cell[1]]}, {floorplan[concave_cell[0], concave_cell[1]]} ')

    return floorplan


def rooms_cells(floorplan):
    unique_values = np.unique(floorplan)
    unique_values = unique_values[unique_values > 0]  # 방 번호만 추출
    cell_dict = {value: [] for value in unique_values}
    # 각 방 번호별로 셀을 분류
    for row in range(floorplan.shape[0]):
        for col in range(floorplan.shape[1]):
            if floorplan[row, col] > 0:
                cell_dict[floorplan[row, col]].append((row, col))
    return cell_dict


def find_extreme_and_concave_cells(floorplan, cells):
    extreme_cell = None
    concave_cell = None
    max_neighbors = -1
    min_neighbors = 5  # 가능한 최대 이웃 수는 4개이므로 5로 초기화

    for cell in cells:
        neighbors = all_active_neighbors(cell, floorplan)
        if len(neighbors) > max_neighbors:
            max_neighbors = len(neighbors)
            concave_cell = cell
        if len(neighbors) < min_neighbors:
            min_neighbors = len(neighbors)
            extreme_cell = cell

    return extreme_cell, concave_cell


def place_room2(floorplan, obtainable_cells):
    insulated_cells = set()
    valid_obtainable_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
    insulated_cells = set(obtainable_cells) - valid_obtainable_cells
    current_step = 0
    while valid_obtainable_cells:
        active_cells = valid_obtainable_cells.copy()
        num_unique_value = len(get_unique_values(floorplan, active_cells))
        if num_unique_value > 1:
            for cell in valid_obtainable_cells:
                print(f'cell =  {cell} ')
                if cell in active_cells:
                    new_cells = expand_room_with_square_shape(floorplan, cell)
                else:
                    continue  # 다음 셀을 실행한다.

                for new_cell in new_cells:
                    if not check_valid_current_cell(floorplan, cell):
                        if cell in active_cells:
                            active_cells.remove(cell)
                        insulated_cells.add(cell)

                    if check_valid_current_cell(floorplan, new_cell):
                        active_cells.add(new_cell)
                    else:  # if not valid
                        insulated_cells.add(new_cell)

                    for adj_cell in all_active_neighbors(new_cell, floorplan):
                        if not len(get_unassigned_neighbor_set(adj_cell, floorplan)) > 0:  # has no candidate
                            if adj_cell in active_cells:
                                active_cells.remove(adj_cell)
                            insulated_cells.add(adj_cell)

                print(f'\tactive_cells:{active_cells}:{len(active_cells)}')

        # 모든 active_cell들의 값이 같으면 반복하지 말고 나머지 모든 셀을 그 값으로 채운다
        elif num_unique_value == 1:
            unique_value = floorplan[active_cells.pop()]
            print(f'all the active_cells having same value {unique_value}')
            floorplan[floorplan == 0] = unique_value
            active_cells = set()

        # gridname = trivial_utils.create_filename('png', 'Step')
        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename)

        print(
            f'==================\nobtainable_cells={valid_obtainable_cells} {len(valid_obtainable_cells)}\nfloorplan=\n{floorplan}')
        valid_obtainable_cells = active_cells.copy()
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan


def expand_room_with_square_shape(floorplan, cell):
    """
    선택된 셀을 기준으로 사각형 모양을 유지하면서 인접한 셀을 확장하는 함수
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_cells = []

    for dx, dy in directions:
        new_row, new_col = cell[0] + dx, cell[1] + dy
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1] and floorplan[new_row, new_col] == 0:
            new_cells.append((new_row, new_col))
            floorplan[new_row, new_col] = floorplan[cell[0], cell[1]]

    # 사각형 모양을 유지하기 위해 추가 확장
    additional_cells = []
    for new_cell in new_cells:
        additional_cell = expand_to_keep_square(floorplan, cell, new_cell)
        if additional_cell:
            additional_cells.append(additional_cell)
            floorplan[additional_cell[0], additional_cell[1]] = floorplan[cell[0], cell[1]]

    return new_cells + additional_cells


def expand_to_keep_square(floorplan, cell, new_cell):
    """
    사각형 모양을 유지하기 위해 추가 확장하는 함수.
    """
    row_diff = new_cell[0] - cell[0]
    col_diff = new_cell[1] - cell[1]

    if abs(row_diff) == 1 and col_diff == 0:  # 상하 확장
        additional_row, additional_col = new_cell[0] + row_diff, new_cell[1]
    elif abs(col_diff) == 1 and row_diff == 0:  # 좌우 확장
        additional_row, additional_col = new_cell[0], new_cell[1] + col_diff
    else:
        return None  # 확장이 불가능한 경우

    if 0 <= additional_row < floorplan.shape[0] and 0 <= additional_col < floorplan.shape[1] and floorplan[
        additional_row, additional_col] == 0:
        return (additional_row, additional_col)
    return None


def choose_adjacent_cells_for_shape(floorplan, cell):
    """
    선택된 셀을 기준으로 인접한 셀을 선택하여 사각형 모양을 유지하는 새로운 셀들을 반환합니다.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_cells = []

    for dx, dy in directions:
        new_row, new_col = cell[0] + dx, cell[1] + dy
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1] and floorplan[new_row, new_col] == 0:
            new_cells.append((new_row, new_col))

    if len(new_cells) >= 1:
        return [new_cells[0]]
    return [None]


def expand_to_keep_square(floorplan, cell, new_cell):
    """
    사각형 모양을 유지하기 위해 추가 확장하는 함수.
    """
    row_diff = new_cell[0] - cell[0]
    col_diff = new_cell[1] - cell[1]

    if abs(row_diff) == 1 and col_diff == 0:  # 상하 확장
        additional_row, additional_col = new_cell[0] + row_diff, new_cell[1]
    elif abs(col_diff) == 1 and row_diff == 0:  # 좌우 확장
        additional_row, additional_col = new_cell[0], new_cell[1] + col_diff
    else:
        return None  # 확장이 불가능한 경우

    if 0 <= additional_row < floorplan.shape[0] and 0 <= additional_col < floorplan.shape[1] and floorplan[
        additional_row, additional_col] == 0:
        return (additional_row, additional_col)
    return None


def place_k_rooms_on_grid(grid_arr, k):
    # Randomly place k rooms within the floorplan
    rooms_placed = 0
    cells_coords = set()
    rooming_grid = grid_arr.copy()
    m, n = grid_arr.shape
    while rooms_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if rooming_grid[row, col] == 0:  # Ensure the cell is within the floorplan and unroomed
            rooming_grid[row, col] = rooms_placed + 1
            cells_coords.add((row, col))
            rooms_placed += 1

    return rooming_grid, cells_coords,


def manhattan_distance(cell1, cell2):
    """
    두 점 사이의 맨해튼 거리를 계산합니다.
    """
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])


def find_closest_pairs(room_cell_dict):
    distances = []
    for (room1, seed1), (room2, seed2) in itertools.combinations(room_cell_dict.items(), 2):
        distance = manhattan_distance(seed1, seed2)
        distances.append(((room1, room2), distance))
    distances.sort(key=lambda x: x[1])

    return distances


def is_position_correct(cell_positions, color, position, orientation_requirements):
    row, col = position
    orientation = orientation_requirements[color]

    for other_color, other_position in cell_positions.items():
        if other_color == color:
            continue
        other_row, other_col = other_position

        if orientation == 'north' and row > other_row:
            return False
        if orientation == 'south' and row < other_row:
            return False
        if orientation == 'east' and col < other_col:
            return False
        if orientation == 'west' and col > other_col:
            return False
    return True


def swap_positions(grid, pos1, pos2):
    grid[pos1[0], pos1[1]], grid[pos2[0], pos2[1]] = grid[pos2[0], pos2[1]], grid[pos1[0], pos1[1]]


def find_extreme_positions(cells_coords):
    minx = min(pos[0] for pos in cells_coords.values())
    norths = {room_no: pos for room_no, pos in cells_coords.items() if pos[0] == minx}
    maxx = max(pos[0] for pos in cells_coords.values())
    souths = {room_no: pos for room_no, pos in cells_coords.items() if pos[0] == maxx}
    miny = min(pos[1] for pos in cells_coords.values())
    wests = {room_no: pos for room_no, pos in cells_coords.items() if pos[1] == miny}
    maxy = max(pos[1] for pos in cells_coords.values())
    easts = {room_no: pos for room_no, pos in cells_coords.items() if pos[1] == maxy}

    return norths, souths, wests, easts


import numpy as np
import random


def relocate_by_orientation_and_adjacency(grid, cell_positions, orientation_requirements, adjacency_requirements):
    def find_extreme_positions(cell_positions):
        # 각 방향에서 가장 극단적인 위치를 찾는 함수
        norths = sorted(cell_positions, key=lambda x: x[0])  # 최북단
        souths = sorted(cell_positions, key=lambda x: -x[0])  # 최남단
        wests = sorted(cell_positions, key=lambda x: x[1])  # 최서단
        easts = sorted(cell_positions, key=lambda x: -x[1])  # 최동단
        return norths, souths, wests, easts

    def remove_position_from_all_directions(pos, direction_to_list):
        # 모든 방향 리스트에서 사용된 위치를 제거
        for direction in direction_to_list.values():
            if pos in direction:
                direction.remove(pos)

    norths, souths, wests, easts = find_extreme_positions(cell_positions)
    direction_to_list = {
        "north": norths,
        "south": souths,
        "east": easts,
        "west": wests
    }

    new_grid = np.copy(grid)

    # 1. 방향 요구사항 반영하여 방 배치
    for key, direction in orientation_requirements.items():
        extreme_positions = direction_to_list[direction]

        # 가장 극단적인 위치를 찾을 때까지 반복
        while extreme_positions:
            pos = extreme_positions.pop(0)  # 해당 방향의 가장 극단적인 위치 선택
            if pos in cell_positions:  # 위치가 아직 사용되지 않았는지 확인
                new_grid[pos] = key  # 해당 위치에 방 번호 설정
                cell_positions.remove(pos)  # 사용한 위치를 제거
                remove_position_from_all_directions(pos, direction_to_list)  # 모든 방향 리스트에서 위치 제거
                break  # 방 배치가 완료되었으므로 루프 탈출

    # 2. 인접성 요구사항 반영하여 방 배치
    for room_a, room_b in adjacency_requirements:
        if room_a in orientation_requirements or room_b in orientation_requirements:
            continue  # 이미 배치된 방에 대해서는 처리하지 않음

        if room_a in cell_positions:
            pos_a = random.choice(list(cell_positions))
            new_grid[pos_a] = room_a
            cell_positions.remove(pos_a)

        if room_b in cell_positions:
            pos_b = random.choice(list(cell_positions))
            new_grid[pos_b] = room_b
            cell_positions.remove(pos_b)

    return new_grid, cell_positions


def relocate_by_orientation(grid, cell_positions, orientation_requirements):
    def find_extreme_positions(cell_positions):
        # 각 방향에서 가장 극단적인 위치를 찾는 함수
        norths = sorted(cell_positions, key=lambda x: x[0])  # 최북단
        souths = sorted(cell_positions, key=lambda x: -x[0])  # 최남단
        wests = sorted(cell_positions, key=lambda x: x[1])  # 최서단
        easts = sorted(cell_positions, key=lambda x: -x[1])  # 최동단
        return norths, souths, wests, easts

    norths, souths, wests, easts = find_extreme_positions(cell_positions)
    direction_to_list = {
        "north": norths,
        "south": souths,
        "east": easts,
        "west": wests
    }

    new_grid = np.copy(grid)

    # orientation_requirements에 있는 방들만 설정
    for key, direction in orientation_requirements.items():
        extreme_positions = direction_to_list[direction]

        if extreme_positions:
            pos = extreme_positions[0]  # 해당 방향의 가장 극단적인 위치 선택
            new_grid[pos] = key  # 해당 위치에 방 번호 설정
            cell_positions.remove(pos)  # 사용한 위치를 제거

    return new_grid, cell_positions


import random

import numpy as np


def relocate_by_adjacency(grid, cell_positions, adjacency_requirements):
    m, n = grid.shape
    new_grid = np.copy(grid)

    def calculate_manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    adj_dict = to_adj_dict(adjacency_requirements)

    for room_pair in adjacency_requirements:
        room_a, room_b = room_pair
        pos_a = cell_positions[room_a]
        pos_b = cell_positions[room_b]

        # room_a와 다른 모든 방들과의 거리를 계산
        distances = []
        for other_room, other_pos in cell_positions.items():
            if other_room != room_a:
                distance = calculate_manhattan_distance(pos_a, other_pos)
                distances.append((other_room, other_pos, distance))

        # 거리 기준으로 정렬
        distances.sort(key=lambda x: x[2])

        # adj_list에 있는 방들과 비교하여 위치를 교환
        for other_room, other_pos, distance in distances:
            if other_room != room_b and distance < calculate_manhattan_distance(pos_a, pos_b):
                # room_b와 더 가까운 방을 찾으면 위치를 교환
                swap_positions(new_grid, pos_b, other_pos)
                cell_positions[room_b] = other_pos
                break

    return new_grid, cell_positions


def relocate_by_orientation_and_adjacency(grid, cell_positions, orientation_requirements, adjacency_requirements):
    def find_extreme_positions(cell_positions):
        # 각 방향에서 가장 극단적인 위치를 찾는 함수
        norths = sorted(cell_positions, key=lambda x: x[0])  # 최북단
        souths = sorted(cell_positions, key=lambda x: -x[0])  # 최남단
        wests = sorted(cell_positions, key=lambda x: x[1])  # 최서단
        easts = sorted(cell_positions, key=lambda x: -x[1])  # 최동단
        return norths, souths, wests, easts

    def remove_position_from_all_directions(pos, direction_to_list):
        # 모든 방향 리스트에서 사용된 위치를 제거
        for direction in direction_to_list.values():
            if pos in direction:
                direction.remove(pos)

    def calculate_manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    norths, souths, wests, easts = find_extreme_positions(cell_positions)
    direction_to_list = {
        "north": norths,
        "south": souths,
        "east": easts,
        "west": wests
    }

    new_grid = np.copy(grid)

    # 1. 방향 요구사항 반영하여 방 배치
    for key, direction in orientation_requirements.items():
        extreme_positions = direction_to_list[direction]

        while extreme_positions:
            pos = extreme_positions.pop(0)  # 해당 방향의 가장 극단적인 위치 선택

            if pos in cell_positions:  # 위치가 아직 사용되지 않았는지 확인
                new_grid[pos] = key  # 해당 위치에 방 번호 설정
                cell_positions.remove(pos)  # 사용한 위치를 제거
                remove_position_from_all_directions(pos, direction_to_list)  # 모든 방향 리스트에서 위치 제거
                break  # 방 배치가 완료되었으므로 루프 탈출
            # 만약 pos가 이미 제거된 위치라면, 루프가 계속되어 다음 위치를 시도합니다.


def to_adj_dict(adjacency_requirements):
    # 결과 딕셔너리 초기화
    adjacency_dict = {}

    # adjacency_requirement 리스트를 순회하며 딕셔너리 생성
    for pair in adjacency_requirements:
        key = pair[0]
        value = pair[1]
        if key in adjacency_dict:
            adjacency_dict[key].append(value)
        else:
            adjacency_dict[key] = [value]

    return adjacency_dict


# todo important: why not just assign random 8 position and later decide room_id
def to_np_array(grid):
    m, n = len(grid), len(grid[0]) if grid else 0
    np_arr = np.full((m, n), -1, dtype=int)  # Initialize all cells as -1
    return np.where(np.array(grid) == 1, 0, -1)


# NumPy 대신 Scipy의 희소 행렬 사용
def create_sparse_grid(rows, cols):
    return lil_matrix((rows, cols), dtype=int)


# cell의 네 방향을 탐색해서 범위에 있고 0이면 선택하고 색깔까지 칠한다.
def choose_new_adjacent_cell(floorplan, cell):
    row, col = cell
    # Ensure we are within the floorplan and the cell has not been colored yet
    if floorplan[row, col] <= 0:  # Adjusted condition to ensure we're targeting uncolored cells
        return False

    valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if
                     0 <= row + dy < floorplan.shape[0] and 0 <= col + dx < floorplan.shape[1] and floorplan[
                         row + dy, col + dx] == 0]

    if valid_offsets:
        dy, dx = random.choice(valid_offsets)
        floorplan[row + dy, col + dx] = floorplan[row, col]  # Color the neighbor
        return (row + dy, col + dx)
    return None


import numpy as np
from multiprocessing import Pool


# wrapper of process_valid_cells for multiprocessing
def get_valid_cell_coords_parallel(grid_assigning):
    if not isinstance(grid_assigning, np.ndarray):  # or not isinstance(grid, np.ndarray):
        raise ValueError("Both grid_assigning and grid must be numpy arrays.")

    num_processes = 1  # 프로세스 수 설정
    pool = Pool(num_processes)
    rows_per_process = grid_assigning.shape[0] // num_processes

    # 각 프로세스에 데이터 분할
    # tasks = [(grid_assigning, grid, range(i * rows_per_process, (i + 1) * rows_per_process))
    tasks = [(grid_assigning, range(i * rows_per_process, (i + 1) * rows_per_process))
             for i in range(num_processes)]

    # 병렬 처리 실행
    valid_cells = pool.starmap(process_valid_cells, tasks)

    # 결과 병합
    pool.close()
    pool.join()
    # * unpacks each set as a separate argument to the union() method
    return set().union(*valid_cells)


def get_valid_cell_coords_parallel(grid_assigning, insulated_cells):
    if not isinstance(grid_assigning, np.ndarray):  # or not isinstance(grid, np.ndarray):
        raise ValueError("Both grid_assigning and grid must be numpy arrays.")

    num_processes = 1  # 프로세스 수 설정
    pool = Pool(num_processes)
    rows_per_process = grid_assigning.shape[0] // num_processes

    # 각 프로세스에 데이터 분할
    # process_grid_get_valid_cells의 argument list
    # tasks = [(grid_assigning, grid, range(i * rows_per_process, (i + 1) * rows_per_process))
    tasks = [(grid_assigning, insulated_cells, range(i * rows_per_process, (i + 1) * rows_per_process))
             for i in range(num_processes)]

    # 병렬 처리 실행
    valid_cells = pool.starmap(process_valid_cells, tasks)

    # 결과 병합
    pool.close()
    pool.join()
    # * unpacks each set as a separate argument to the union() method
    return set().union(*valid_cells)


def exe_draw_grid():
    savepath_reversed = 'output_reversed.png'
    savepath = 'output.png'
    grid = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2),
        (4, 2),
        (5, 2), (5, 3)
    ]
    GridDrawer.draw_grid_reversed(grid, savepath_reversed)
    GridDrawer.draw_grid(grid, savepath)


def exe_build_floorplan():
    # Define your floorplan here
    m, n, k = 5, 5, 4
    floorshape = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    reqs = Req()
    grid = create_floorplan(m, n, k, floorshape, reqs)
    color_grid = plan_utils.get_color_coordinates(grid)
    savepath = 'output.png'
    GridDrawer.draw_grid_reversed(color_grid, savepath)


def main():
    modules = {
        "1": exe_build_floorplan
    }
    for key, fn in modules.items():
        print(f'{key}: {fn.__name__}')
    choice = input('Enter: ')

    if choice in modules:
        modules[choice]()
    exe_build_floorplan()

# if __name__ == '__main__':
# main()
