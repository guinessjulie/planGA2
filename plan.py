import random
import sys

from scipy.sparse import lil_matrix
import trivial_utils
import numpy as np
import plan_utils
from GridDrawer import GridDrawer
from measure import categorize_boundary_cells
from config_reader import read_constraint, read_config_boolean
from cells_utils import  is_valid_cell, is_in

# todo 1: 사이즈 작은 사이즈일 수록 빈 인접셀 적다. 이를 이용해서 초기화에 활용
# todo 2: 인접 리스트를 만들어서 해당 조건 만족하도록
# todo 3: 사이즈 contstraint를 이용하여 초기 셀 설정 시 적용. 예:  living room 크기가 bathroom 크기의 2배 이상인 경우, 인접 셀 두 개를 초기 셀에 할당
# todo 4: 방향 constraint를 이용하여 초기 셀 설정시 적용 가장 남쪽에 있는 셀이 Livingroom 북쪽에 있는 셀은 부엌 및 bathroom => done
# todo 5: 인접조건 neighbor 갯수가 가장 많은 것을 living room으로? 그래프 구조 활용
# todo 6: Graph의 구조가 같은지를 체크할 수 있도록 GraphGrid에 equality function들을 만들었다. 이를 활용해서 인접성 리스트를 optimize 하자


def create_floorplan(empty_grid, k):
    orientation_requirements=read_constraint('constraints.ini', 'OrientationRequirements')
    display_process = read_config_boolean('constraints.ini', 'RunningOptions', 'display_place_room_process')
    save_process = read_config_boolean('constraints.ini', 'RunningOptions', 'save_place_room_process')

    initialized_grid, initial_cells = place_k_rooms_on_grid(to_np_array(empty_grid), k)
    initialized_grid, cells_coords = relocate_by_orientation(initialized_grid, initial_cells, orientation_requirements)
    print(f'initialized_grid={initialized_grid}')
    path = trivial_utils.create_folder_by_datetime()
    full_path = trivial_utils.create_filename(path, 'Init', '', '', 'png')
    GridDrawer.color_cells_by_value(initialized_grid, full_path, display = display_process, save=save_process, num_rooms=k)
    print(f'initial_cells:{initial_cells}\n{initialized_grid}')
    # floorplan = place_room(initialized_grid, initial_cells)
    # floorplan = place_room(initialized_grid, initial_cells, display = display_process, save=save_process) # todo to revert if follow not working
    floorplan = parallel_extention(initialized_grid, initial_cells, display = display_process, save=save_process, num_rooms=k)
    return floorplan


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



def parallel_extention(floorplan, obtainable_cells, display=False, save=True, num_rooms=7):

    def same_room_cells(floorplan, cell):
        return np.argwhere(floorplan == floorplan[cell])

    def choose_new_adjacent_cell(floorplan, cell):
        row, col = cell
        rows, cols = floorplan.shape[0], floorplan.shape[1]
        # Ensure we are within the floorplan and the cell has not been colored yet
        if floorplan[row, col] <= 0:  # Adjusted condition to ensure we're targeting uncolored cells
            return False

        valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if 0 <= row + dy < rows and 0 <= col + dx < cols and floorplan[row + dy, col + dx] == 0]

        if valid_offsets:
            dr = random.choice(valid_offsets)
            dy, dx = dr[0], dr[1]
            floorplan[row + dy, col + dx] = floorplan[row, col]  # Color the neighbor
            return (row + dy, col + dx), dr

        return None
    def update_active_cells(floorplan,cell, active_cells):
        if not has_neighbor_zero(floorplan, cell[0], cell[1]): # not( has_neighbor_zero = 범위 안에 있고 0인 인접셀이 하나라도 있으면)
            if cell in active_cells: # 그 셀은 더이상 active하지 않으므로 active_cells에서 제거하고
                print(f'{cell} has not neighbor zero and in active_cells :why is this happening ')
                active_cells.remove()
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

    while valid_obtainable_cells:
        active_cells = valid_obtainable_cells.copy()
        num_unique_value = len(get_unique_values(floorplan, active_cells)) # 중복 제거
        if num_unique_value > 1:
            for cell in valid_obtainable_cells:
                if cell in active_cells:
                    new_cell, dr= choose_new_adjacent_cell(floorplan, cell)
                    update_active_cells(floorplan, new_cell, active_cells)
                    parallel_cells = list(np.argwhere(floorplan == floorplan[cell]))  # 같은 룸에서
                    parallel_cells = [p for p in parallel_cells if not (cell[0] == p[0] and cell[1] == p[1])]
                    print(f'{cell} parallel_cell = {parallel_cells}')
                    for sc in parallel_cells :
                        # 같은 방향으로 이동 가능한 것을 찾아서 모두 바꾸어준다.

                        new_same_room_cell = sc[0] + dr[0], sc[1] +  dr[1]

                        if is_in(floorplan, new_same_room_cell) and  floorplan[new_same_room_cell] == 0 and (new_same_room_cell in active_cells):
                            floorplan[new_same_room_cell] = floorplan[cell]
                            update_active_cells(floorplan, new_same_room_cell)
                else:
                    continue

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

    floorplan = exchange_extreme_cells(floorplan)
    filename, current_step = trivial_utils.create_filename_in_order('png', 'Reform', current_step)
    GridDrawer.color_cells_by_value(floorplan, filename, display=display, save=save, num_rooms=num_rooms)

    return floorplan

# commit된 것 복원
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

    return rooming_grid, cells_coords



def relocate_by_orientation(grid, cells_coords, orientation_requirements):
    def is_position_correct(cell_positions, color, position):
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
        north = min(cells_coords, key=lambda x: x[0])
        south = max(cells_coords, key=lambda x: x[0])
        west = min(cells_coords, key=lambda x: x[1])
        east = max(cells_coords, key=lambda x: x[1])
        return north, south, west, east

    m, n = grid.shape
    new_grid = np.copy(grid)
    cell_positions = {grid[row, col]: (row, col) for row, col in cells_coords}

    any_changes = True
    while any_changes:
        any_changes = False
        north, south, west, east = find_extreme_positions(cell_positions.values())

        for color, position in list(cell_positions.items()):
            if color in orientation_requirements:
                if not is_position_correct(cell_positions, color, position):
                    if position == north:
                        new_position = (position[0] - 1, position[1])  # Move north
                    elif position == south:
                        new_position = (position[0] + 1, position[1])  # Move south
                    elif position == east:
                        new_position = (position[0], position[1] + 1)  # Move east
                    elif position == west:
                        new_position = (position[0], position[1] - 1)  # Move west
                    else:
                        continue

                    if (0 <= new_position[0] < m and 0 <= new_position[1] < n and
                            new_grid[new_position[0], new_position[1]] == 0):
                        swap_positions(new_grid, position, new_position)
                        cell_positions[color] = new_position
                        any_changes = True

    new_cells_coords = set(cell_positions.values())
    return new_grid, new_cells_coords


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
    print(f'\t\tmxn iteration: process_valid_cells() global valid_cells = {valid_cells}')
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
