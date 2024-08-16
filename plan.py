import random
import sys

import networkx as nx
from scipy.sparse import lil_matrix
import trivial_utils
import numpy as np
import plan_utils
from GridDrawer import GridDrawer
import itertools
from measure import categorize_boundary_cells
from config_reader import read_constraint, read_config_boolean, read_str_constraints
from cells_utils import  is_valid_cell, is_inside
from plan_utils import dict_value_to_coordinates
import ast

# todo 1: 사이즈 작은 사이즈일 수록 빈 인접셀 적다. 이를 이용해서 초기화에 활용
# todo 2: 인접 리스트를 만들어서 해당 조건 만족하도록
# todo 3: 사이즈 contstraint를 이용하여 초기 셀 설정 시 적용. 예:  living room 크기가 bathroom 크기의 2배 이상인 경우, 인접 셀 두 개를 초기 셀에 할당
# todo 4: 방향 constraint를 이용하여 초기 셀 설정시 적용 가장 남쪽에 있는 셀이 Livingroom 북쪽에 있는 셀은 부엌 및 bathroom => done
# todo 5: 인접조건 neighbor 갯수가 가장 많은 것을 living room으로? 그래프 구조 활용
# todo 6: Graph의 구조가 같은지를 체크할 수 있도록 GraphGrid에 equality function들을 만들었다. 이를 활용해서 인접성 리스트를 optimize 하자

def check_adjacent_requirement(ini_filename):
    section = 'AdjacencyRequirements'
    edges_str = read_str_constraints(ini_filename,section )

    adjacency_list = ast.literal_eval(edges_str['adjacent'])  # TO LIST
    return adjacency_list


def create_req_graph(adjacency_list):
    adjacency_graph = nx.Graph()
    adjacency_graph.add_edges_from(adjacency_list)
    if not nx.check_planarity(adjacency_graph):
        return False
    return adjacency_graph

def locate_initial_cell(empty_grid, k):
    ini_filename = 'constraints.ini'
    adjacency_list = check_adjacent_requirement(ini_filename)
    initialized_grid, initial_cells = place_seed(to_np_array(empty_grid), k, adjacency_list )
    return initialized_grid, initial_cells

def create_floorplan(initialized_grid,initial_cells, k):
    ini_filename = 'constraints.ini'


    orientation_requirements=read_constraint(ini_filename, 'OrientationRequirements')
    display_process = read_config_boolean(ini_filename, 'RunningOptions', 'display_place_room_process')
    save_process = read_config_boolean(ini_filename, 'RunningOptions', 'save_place_room_process')
    # todo 20240814 room_number 가 언제 할당되나 조사
    # initialized_grid, initial_cells = place_k_rooms_on_grid(to_np_array(empty_grid), k) # todo place_seed에서 그래프 만족시키는 seed 새로 만듦
    path = trivial_utils.create_folder_by_datetime()
    full_path = trivial_utils.create_filename(path, 'Init0', '', '', 'png')
    GridDrawer.color_cells_by_value(initialized_grid, full_path, display=display_process, save=save_process,
                                    num_rooms=k)

    initialized_grid, initial_cells = relocate_by_orientation(initialized_grid, initial_cells, orientation_requirements)
    full_path = trivial_utils.create_filename(path, 'Init1', '', '', 'png')
    print(f'adjacency considered={initialized_grid}')
    GridDrawer.color_cells_by_value(initialized_grid, full_path, display = display_process, save=save_process, num_rooms=k)
    print(f'relocated:{initial_cells}\n{initialized_grid}')

    floorplan = allocate_rooms(initialized_grid, initial_cells, display = display_process, save=save_process, num_rooms=k)
    return floorplan, initial_cells


#  현재 셀이 valid 한가
def check_valid_current_cell(grid_assigning, cell):
    row, col = cell[0], cell[1]
    if not has_neighbor_zero(grid_assigning, row, col):
        return False
    return True


def all_active_neighbors(cell, floorplan):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell
    adjs = set()
    for dx, dy in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1]:  # 범위내에 있으면
            if floorplan[new_row, new_col] > 0:  # 빈 셀은 할 필요가 없을 듯
                adjs.add((new_row, new_col))
    return adjs



# 빈 이웃이 하나라도 있으면 그 이웃을 valid_neighbor_set에 추가해서 이를 리턴한다.
def collect_candidate_set(cell, grid_assigning):
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

# todo active_cells를 복사하지 않고 바로 이용
def allocate_rooms(floorplan, obtainable_cells, display=False, save=True, num_rooms=7):

    # todo 여기서는 그냥 choose만 하고, return 값은 새 셀의 좌표와 방향
    def choose_new_adjacent_cell(floorplan, cell):
        row, col = cell
        rows, cols = floorplan.shape
        # Ensure we are within the floorplan and the cell has not been colored yet
        if floorplan[row, col] > 0:  # Adjusted condition to ensure we're targeting uncolored cells
            valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                             if 0 <= row + dy < rows and 0 <= col + dx < cols and floorplan[row + dy, col + dx] == 0]
            if valid_offsets:
                dr = random.choice(valid_offsets)
                return (row + dr[0], col + dr[1]), dr
        else:
            return None, (0,0)

    # 0인 neighbor가 없고 즉 더이상 확장할 수 없는데 그 셀이 actie_cells 리스트에 있으면 거기서 제거
    # 0인 네이버가 있으면 무조건 active cell에 추가
    def update_active_cells(floorplan,cell, active_cells):
        if not has_neighbor_zero(floorplan, cell[0], cell[1]): # not( has_neighbor_zero = 범위 안에 있고 0인 인접셀이 하나라도 있으면)
            if cell in active_cells: # 그 셀은 더이상 active하지 않으므로 active_cells에서 제거하고
                active_cells.remove(cell)
        else:
            active_cells.add(cell)

        # 이 셀이 더이상 확장 가능하거나 가능하지 않거나 상관없이 모든 인접셀에 대해서 다시 candidate를 구한다. if 문에서 들어가면
        for adj_cell in all_active_neighbors(cell, floorplan):
            if not len(collect_candidate_set(adj_cell, floorplan)) > 0:  # has no candidate
                if adj_cell in active_cells:
                    active_cells.remove(adj_cell)

    active_cells = process_valid_cells(floorplan)
    current_step = 0

    while active_cells:
        room_to_coordinates = dict_value_to_coordinates(floorplan)
        room_numbers = list(range(1, num_rooms+1))
        while room_numbers:
            room_idx = random.randrange(len(room_numbers))
            room = room_numbers.pop(room_idx)
            all_cells = room_to_coordinates.get(room, [])
            # valid한  coordsnates를 찾아라
            current_active_cells = [c for c in all_cells if c in active_cells]
            if not current_active_cells :
                continue# 현재 선택된 방의 모든 valid한 셀
            cell = random.choice(current_active_cells) #todo cannot choose from an empty sequence
            # just pick one from each cell
            new_cell, dr = choose_new_adjacent_cell(floorplan, cell) # todo 이 버전은 값은 안변하고 셀만 구한다. 일단 간직만 하자. 변경하는 것은 나중에
            if new_cell is None:
                continue
            # assign_room
            # 바꾸려고 하는 셀이 원래셀이 아니고, 바꿀 대상 위치가 비어있고(0) 범위 내에 있는 것만 콜렉트
            parallel_cells =[tuple(np.add(c, dr)) for c in all_cells if
               c != cell and floorplan[tuple(np.add(c, dr))] == 0 and is_inside(floorplan, tuple(np.add(c, dr)))]

            #색칠
            floorplan[new_cell] = floorplan[cell]
            update_active_cells(floorplan, cell, active_cells) # todo active_cells를 잘봐
            update_active_cells(floorplan, new_cell, active_cells)

            for c in parallel_cells:
                floorplan[c] = floorplan[cell]
                update_active_cells(floorplan, c, active_cells)

        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    return floorplan

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
            return None, (0,0)
    # 0인 neighbor가 없고 즉 더이상 확장할 수 없는데 그 셀이 actie_cells 리스트에 있으면 거기서 제거
    # 0인 네이버가 있으면 무조건 active cell에 추가
    def update_active_cells(floorplan,cell, active_cells):
        if not has_neighbor_zero(floorplan, cell[0], cell[1]): # not( has_neighbor_zero = 범위 안에 있고 0인 인접셀이 하나라도 있으면)
            if cell in active_cells: # 그 셀은 더이상 active하지 않으므로 active_cells에서 제거하고
                print(f'{cell} has not neighbor zero and in active_cells :why is this happening ')
                active_cells.remove(cell)
        else:
            active_cells.add(cell)

        # 이 셀이 더이상 확장 가능하거나 가능하지 않거나 상관없이 모든 인접셀에 대해서 다시 candidate를 구한다. if 문에서 들어가면
        for adj_cell in all_active_neighbors(cell, floorplan):
            if not len(collect_candidate_set(adj_cell, floorplan)) > 0:  # has no candidate
                if adj_cell in active_cells:
                    active_cells.remove(adj_cell)

    insulated_cells = set()
    # todo .1 insulated_cell 을 이용하지 않았어
    # todo .2 obtainable_cells를 기껏 가져와서 process_valid_cells에서 다시 구했어. 아래 문장에서는  obtainable_cells와 insulated_cells를 모두 이용하지 않았어.
    valid_obtainable_cells = process_valid_cells(floorplan)
    current_step = 0

    while len(valid_obtainable_cells) > 0 :
        room_to_coordinates = dict_value_to_coordinates(floorplan)
        # todo 각 룸마다 하나씩 패러랠하게 추가
        # todo room_to_coordinates로 for 문을 돌리면 같은 셀로 계속 반복하게 됨
        #  coordinates는 나중에 룸 번호보고 다시 선택하기로 하고, iteration은 room 위에 while 문 필요함.
        room_numbers = list(range(1, num_rooms+1))
        while room_numbers:
        # for room, all_cells in room_to_coordinates.items():
            room_idx = random.randrange(len(room_numbers))
            room = room_numbers.pop(room_idx)
            all_cells = room_to_coordinates[room]
            active_cells = valid_obtainable_cells.copy()
            # valid한  coordsnates를 찾아라
            current_active_cells = [c for c in all_cells if c in active_cells]
            if len(current_active_cells) < 1 :
                continue# 현재 선택된 방의 모든 valid한 셀
            cell = random.choice(current_active_cells) #todo cannot choose from an empty sequence
            # just pick one from each cell
            new_cell, dr = choose_new_adjacent_cell(floorplan, cell) # todo 이 버전은 값은 안변하고 셀만 구한다. 일단 간직만 하자. 변경하는 것은 나중에
            # assign_room
            # 바꾸려고 하는 셀이 원래셀이 아니고, 바꿀 대상 위치가 비어있고(0) 범위 내에 있는 것만 콜렉트
            parallel_cells =[tuple(np.add(c, dr)) for c in all_cells if
               c is not cell and floorplan[tuple(np.add(c, dr))] == 0 and is_inside(floorplan, tuple(np.add(c, dr)))]

            #색칠
            floorplan[new_cell] = floorplan[cell]
            update_active_cells(floorplan, cell, active_cells) # todo active_cells를 잘봐
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
def place_room(floorplan, obtainable_cells, display = False, save = True, num_rooms=7):
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
                    if not len(collect_candidate_set(adj_cell, floorplan)) > 0:  # has no candidate
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



# chatGPT가 고쳐준 것
def place_room2(floorplan, obtainable_cells):
    insulated_cells = set()
    valid_obtainable_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
    insulated_cells = set(obtainable_cells) - valid_obtainable_cells
    current_step = 0

    while valid_obtainable_cells:
        num_unique_value = len(get_unique_values(floorplan, valid_obtainable_cells))
        if num_unique_value > 1:
            updated_cells = set()
            for cell in valid_obtainable_cells:
                if cell in updated_cells:
                    continue
                new_cell = choose_new_adjacent_cell(floorplan, cell)
                print(f'cell =  {cell}, new_cell = {new_cell}')

                # 유효성 검사 통합
                is_cell_valid = check_valid_current_cell(floorplan, cell)
                is_new_cell_valid = check_valid_current_cell(floorplan, new_cell)

                if not is_cell_valid:
                    insulated_cells.add(cell)
                if is_new_cell_valid:
                    updated_cells.add(new_cell)
                else:
                    insulated_cells.add(new_cell)

                # 인접 셀 유효성 검사
                for adj_cell in all_active_neighbors(new_cell, floorplan):
                    if adj_cell in valid_obtainable_cells and not check_valid_current_cell(floorplan, adj_cell):
                        insulated_cells.add(adj_cell)

            valid_obtainable_cells = updated_cells.copy()
            print(f'active_cells:{valid_obtainable_cells}:{len(valid_obtainable_cells)}')

        elif num_unique_value == 1:
            unique_value = floorplan[valid_obtainable_cells.pop()]
            floorplan[floorplan == 0] = unique_value
            valid_obtainable_cells = set()

        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename)

        print(
            f'==================\nobtainable_cells={valid_obtainable_cells} {len(valid_obtainable_cells)}\nfloorplan=\n{floorplan}')

    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    # 교환 로직 추가
    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    GridDrawer.color_cells_by_value(floorplan, filename)

    return floorplan

# 제대로 동작하는 place_room
def place_room3(floorplan, obtainable_cells):
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
                    if not len(collect_candidate_set(adj_cell, floorplan)) > 0:  # has no candidate
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

    # 교환 로직 추가
    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    GridDrawer.color_cells_by_value(floorplan, filename)

    return floorplan

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
                        if not len(collect_candidate_set(adj_cell, floorplan)) > 0:  # has no candidate
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

    if 0 <= additional_row < floorplan.shape[0] and 0 <= additional_col < floorplan.shape[1] and floorplan[additional_row, additional_col] == 0:
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


def place_seed(grid_arr, k, adjacency_graph):
    # Randomly place k rooms within the floorplan
    rooms_placed = 0
    cells_coords = set()
    rooming_grid = grid_arr.copy()
    m, n = grid_arr.shape
    room_cell_dict = dict()
    while rooms_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if rooming_grid[row, col] == 0:  # Ensure the cell is within the floorplan and unroomed
            rooms_placed += 1
            rooming_grid[row, col] = rooms_placed
            cells_coords.add((row, col))
            room_cell_dict[rooms_placed] = (row, col)


# todo reassign_seeds_based_on_proximity에서 동일 셀에게 다른 룸을 부여하는 문제가 생김 그래서 일단 이것은 스킵하고 나중에 다시
#    closest_room_pairs = find_closest_pairs(room_cell_dict)
#    room_cell_dict = reassign_seeds_based_on_proximity(closest_room_pairs, adjacency_graph, room_cell_dict)
#   for room_num, pos in room_cell_dict.items():
#        rooming_grid[pos] = room_num


    return rooming_grid, room_cell_dict

# todo 이것은 제대로 동작하지 않는다. 나중에 다시 보도록 합시다.
def reassign_seeds_based_on_proximity(closest_room_pairs, adjacency_requirements, room_seeds):
    new_seeds = room_seeds.copy()
    used_rooms = set()

    for (req_room1, req_room2) in adjacency_requirements:
        for (close_room1, close_room2), _ in closest_room_pairs:
            if close_room1 not in used_rooms and close_room2 not in used_rooms:
                # 인접 요구사항과 가장 가까운 방 쌍을 매칭
                if (req_room1, req_room2) == (close_room1, close_room2) or (req_room2, req_room1) == (close_room1, close_room2):
                    new_seeds[req_room1], new_seeds[req_room2] = new_seeds[close_room1], new_seeds[close_room2]
                    used_rooms.add(req_room1)
                    used_rooms.add(req_room2)
                    break
                # 위치 교환
                elif req_room1 == close_room1 or req_room2 == close_room2:
                    new_seeds[req_room1], new_seeds[req_room2] = new_seeds[close_room2], new_seeds[close_room1]
                    used_rooms.add(req_room1)
                    used_rooms.add(req_room2)
                    break

    return new_seeds


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
    norths ={room_no:pos for room_no, pos in cells_coords.items() if pos[0] == minx}
    maxx = max(pos[0] for pos in cells_coords.values())
    souths ={room_no:pos for room_no, pos in cells_coords.items() if pos[0] == maxx}
    miny = min(pos[1] for pos in cells_coords.values())
    wests ={room_no:pos for room_no, pos in cells_coords.items() if pos[1] == miny}
    maxy = max(pos[1] for pos in cells_coords.values())
    easts ={room_no:pos for room_no, pos in cells_coords.items() if pos[1] == maxy}

    return norths, souths, wests, easts
def relocate_by_orientation(grid, cell_positions, orientation_requirements):

    norths, souths, wests, easts = find_extreme_positions(cell_positions)
    direction_to_list = {
        "north": norths,
        "south": souths,
        "east": easts,
        "west": wests
    }

    m, n = grid.shape
    new_grid = np.copy(grid)
    # cell_positions = {grid[row, col]: (row, col) for row, col in cells_coords.items()}

    for room_num, ot_req  in orientation_requirements.items():
        cell_list_on_direction = direction_to_list[ot_req] #norths, souths, easts, wests 중 하나
        if room_num not in cell_list_on_direction.keys(): # 만일 room_num의 constraint가 south인데 room_num이 souths에 없다면 souths 중 하나를 찾아서 swap
            new_room_num, new_pos = random.choice(list(cell_list_on_direction.items()))
            swap_positions(grid, cell_positions[room_num], new_pos)
            current_pos = cell_positions[room_num]
            cell_positions[room_num] = new_pos
            cell_positions[new_room_num] = current_pos

    return grid, cell_positions


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


def process_valid_cells(grid_assigning, insulated_cells=None, row_range=None): # 두 개의 parameter가 필요없어서 기본값을 None으로 주었음
    valid_cells = set()
    #    print(f'insulated_cells={insulated_cells} in process_valid_cells')
    for row in range(grid_assigning.shape[0]):
        for col in range(grid_assigning.shape[1]):
            # if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0:
                if has_neighbor_zero(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells


# 범위 내에 있고 옆에 하나라도 빈 셀이 있으면 True
def has_neighbor_zero(grid_assigning, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False


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
    grid = create_floorplan(m, n, k, floorshape)
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


if __name__ == '__main__':
    main()
