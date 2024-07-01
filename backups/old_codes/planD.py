# working version saved
import numpy as np
import random
from PIL import Image, ImageDraw
from scipy.sparse import lil_matrix
import trivial_utils

import plan_utils
from GridDrawer import GridDrawer

# todo 1: 사이즈 작은 사이즈일 수록 빈 인접셀 적다. 이를 이용해서 초기화에 활용
# todo 2: 인접 리스트를 만들어서 해당 조건 만족하도록

def update_assigned_cells(assigned_cells, floorplan, grid) -> object:
    while assigned_cells:
        new_boundary_cells = set()
        for cell in assigned_cells:
            if choose_new_adjacent_cell(grid, cell):
                new_boundary_cells.update(get_valid_cell_coords_parallel(grid, floorplan))
        assigned_cells = new_boundary_cells
        print(f'while assigned_cells: {assigned_cells}')
    print(f'outside the while assigned_cells:{assigned_cells}')

def create_floorplan(empty_grid, k):
    initialized_grid, initial_cells = place_k_colors_on_grid(to_np_array(empty_grid), k)
    gridname = trivial_utils.create_filename_with_datetime('png', 'Init')
    GridDrawer.color_cells_by_value(initialized_grid, gridname)
    print(f'initial_cells:{initial_cells}\n{initialized_grid}')
    floorplan = place_room(initialized_grid, initial_cells)
    return floorplan

#  현재 셀이 valid 한가
def check_valid_current_cell(grid_assigning, cell):
    row, col = cell[0], cell[1]
    if not has_neighbor_zero(grid_assigning, row, col):
        return False
    return True
def all_active_neighbors(cell, floorplan):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell[0], cell[1]
    adjs= set()
    for dx, dy in directions:
        new_row, new_col = row+dy, col+dx
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1]:  # 범위내에 있으면
            if floorplan[new_row, new_col] > 0: #빈 셀은 할 필요가 없을 듯
                adjs.add((new_row, new_col))
    return adjs
# 빈 이웃이 하나라도 있으면 그 이웃을 valid_neighbor_set에 추가해서 이를 리턴한다.
def collect_candidate_set(cell, grid_assigning):
    # print(f'\t\tcollect_valid_adjacent_cells: {cell}')
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row,col = cell[0], cell[1]
    valid_neighbor_cells = set()
    for dx, dy in directions:
        new_row, new_col = row+dy, col+dx
        if 0 <= new_row <grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]: # 범위내에 있으면
            # print(f'\t\t\t({new_row}, {new_col}) inside')
            # if vacant(new_row, new_col, grid_assigning):
            if grid_assigning[new_row, new_col] == 0 : #vacant
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



def place_room(floorplan, obtainable_cells):
    insulated_cells = set()
    valid_obtainable_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
    insulated_cells = set(obtainable_cells) - valid_obtainable_cells
    current_step = 0
    while valid_obtainable_cells:
        active_cells=valid_obtainable_cells.copy()
        num_unique_value = len(get_unique_values(floorplan, active_cells))
        if num_unique_value > 1:
            for cell in valid_obtainable_cells:
                print(f'cell =  {cell} ' )
                if cell in active_cells :
                    new_cell = choose_new_adjacent_cell(floorplan, cell)
                else: continue # 다음 셀을 실행한다.

                # 새 셀이 삽입되면 원래 셀, 본인. 본인의 네이버 모두 valid한지 체크해야 한다.
                print(f'\tnew_cell = {new_cell}')
                if not check_valid_current_cell(floorplan, cell):
                    if cell in active_cells:
                        active_cells.remove(cell)
                    insulated_cells.add(cell)

                if check_valid_current_cell(floorplan, new_cell):
                    active_cells.add(new_cell)
                else: # if not valid
                    insulated_cells.add(new_cell)

                for adj_cell in all_active_neighbors(new_cell, floorplan):
                    if not len(collect_candidate_set(adj_cell, floorplan)) > 0: # has no candidate
                        if adj_cell in active_cells:
                            active_cells.remove(adj_cell)
                        insulated_cells.add(adj_cell)

                print(f'\tactive_cells:{active_cells}:{len(active_cells)}')

        # 모든 active_cell들의 값이 같으면 반복하지 말고 나머지 모든 셀을 그 값으로 채운다
        elif num_unique_value == 1 :
            unique_value = floorplan[active_cells.pop()]
            print(f'all the active_cells having same value {unique_value}')
            floorplan[floorplan == 0] = unique_value
            active_cells=set()

        # gridname = trivial_utils.create_filename('png', 'Step')
        filename, current_step = trivial_utils.create_filename_in_order('png', 'Step', current_step)
        GridDrawer.color_cells_by_value(floorplan, filename)

        print(f'==================\nobtainable_cells={valid_obtainable_cells} {len(valid_obtainable_cells)}\nfloorplan=\n{floorplan}')
        valid_obtainable_cells = active_cells.copy()
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan

def place_room_debug(floorplan, obtainable_cells):
    insulated_cells = set()
    valid_obtainable_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
    insulated_cells = set(obtainable_cells) - valid_obtainable_cells
    active_cells = set(obtainable_cells) - insulated_cells#todo to see what is crossed

    while valid_obtainable_cells:
        active_cells=valid_obtainable_cells.copy()
        num_unique_value = len(get_unique_values(floorplan, active_cells))
        if num_unique_value > 1:

        # active_cells = active_cells.intersection(insulated_cells)
        # print(f'while obtainable_cells {valid_obtainable_cells}:{len(valid_obtainable_cells)}, active_cells= {active_cells}:{len(active_cells)} ')
            for cell in valid_obtainable_cells:
                # print(f'obtainable_cells = {valid_obtainable_cells}')
                print(f'cell =  {cell} ' )
                if cell in active_cells :
                    new_cell = choose_new_adjacent_cell(floorplan, cell)
                # 새 셀이 삽입되면 원래 셀, 본인. 본인의 네이버 모두 valid한지 체크해야 한다.
                if new_cell:
                    print(f'\tnew_cell = {new_cell}')
                    # floorplan[new_cell[0], new_cell[1]] = floorplan[cell[0], cell[1]]# todo 필요없음 choose_new_adjacent_cell에서 이미 했음
                    if not check_valid_current_cell(floorplan, cell):
                        if cell in active_cells:
                            active_cells.remove(cell)
                            print(f'\t\t({cell}) removed from active_cells')
                        insulated_cells.add(cell)
                        print(f'\t\t({cell}) added to insulated_cells')
                    if check_valid_current_cell(floorplan, new_cell):
                        active_cells.add(new_cell)
                        print(f'\t\t({new_cell} added to active_cells')
                    else: # if not valid
                        print(f'\t\t({new_cell}) not added')
                        print(f'\t\t({new_cell}) added to insulated_cells')
                        insulated_cells.add(new_cell)


                    for adj_cell in all_active_neighbors(new_cell, floorplan):
                        if not len(collect_candidate_set(adj_cell, floorplan)) > 0: # has no candidate
                            if adj_cell in active_cells:
                                active_cells.remove(adj_cell)
                                print(f'\t\tremove ({adj_cell}) from active_cells')
                            insulated_cells.add(adj_cell)
                            print(f'\t\t({adj_cell}) added to insulated_cells')

                    print(f'\tactive_cells:{active_cells}:{len(active_cells)}')

                else:
                    print(f'if not new_cell = {new_cell}') #todo to see when this happends
        elif num_unique_value == 1 :
            unique_value = floorplan[active_cells.pop()]
            print(f'all the active_cells having same value {unique_value}')
            floorplan[floorplan == 0] = unique_value
            active_cells=set()
        else:
            print(f'num_unique_value= {num_unique_value}')
        print(f'==================\nobtainable_cells={valid_obtainable_cells} {len(valid_obtainable_cells)}\nfloorplan=\n{floorplan}')
        valid_obtainable_cells = active_cells.copy()
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan
def place_k_colors_on_grid(grid_arr, k):
    # Randomly place k colors within the floorplan
    colors_placed = 0
    cells_coords = set()
    coloring_grid = grid_arr.copy()
    m, n = grid_arr.shape
    while colors_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if coloring_grid[row, col] == 0:  # Ensure the cell is within the floorplan and uncolored
            coloring_grid[row, col] = colors_placed + 1
            cells_coords.add((row, col))
            colors_placed += 1
    return coloring_grid, cells_coords


"""
input grid[][]를 입력받아 np.array 출력
"""
def to_np_array(grid):
    m, n = len(grid), len(grid[0]) if grid else 0
    np_arr = np.full((m, n), -1, dtype=int)  # Initialize all cells as -1
    return np.where(np.array(grid)==1, 0, -1)

def has_neighbor_zero(grid_assigning, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        print(f'cheking has_neighbor_zero ({new_row},{new_col})')
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            print(f'\tinside')
            if grid_assigning[new_row, new_col] == 0:
                print(f'\t[{new_row, new_col} ]==0\n\t returning True')
                return True
    print(f'condition inside and 0 not met returning False')
    return False

# NumPy 대신 Scipy의 희소 행렬 사용
def create_sparse_grid(rows, cols):
    return lil_matrix((rows, cols), dtype=int)

def has_neighbor_zero(grid_assigning, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False

# NumPy 대신 Scipy의 희소 행렬 사용
def create_sparse_grid(rows, cols):
    return lil_matrix((rows, cols), dtype=int)


def choose_new_adjacent_cell(floorplan, cell):
    row, col = cell
    # Ensure we are within the floorplan and the cell has not been colored yet
    if floorplan[row, col] <= 0:  # Adjusted condition to ensure we're targeting uncolored cells
        return False

    valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if
                     0 <= row + dy < floorplan.shape[0] and 0 <= col + dx < floorplan.shape[1] and floorplan[row + dy, col + dx] == 0]

    if valid_offsets:
        dy, dx = random.choice(valid_offsets)
        floorplan[row + dy, col + dx] = floorplan[row, col]  # Color the neighbor
        return (row+dy, col+dx)
        # return True
    return None


import numpy as np
from multiprocessing import Pool


def process_grid_get_valid_cells_safe(grid_assigning, grid, row_range):
    working_cells = set()
    for row in row_range:
        for col in range(grid.shape[1]):
            #if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0 :
                if has_neighbor_zero(grid_assigning, row, col):
                    working_cells.add((row, col))
    return working_cells
# 할당가능한 이웃이 남아있는 셀 집합을 구한다.
# def process_grid_segment(grid_assigning, row_range): #rename
def process_grid_get_valid_cells(grid_assigning, row_range):
    valid_cells = set()
    for row in row_range:
        for col in range(grid_assigning.shape[1]):
            if grid_assigning[row, col] > 0 :
                if has_neighbor_zero(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells
def process_valid_cells(grid_assigning, insulated_cells, row_range):
    valid_cells = set()
#    print(f'insulated_cells={insulated_cells} in process_valid_cells')
    for row in row_range:
        for col in range(grid_assigning.shape[1]):
            #if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0 :
                if has_neighbor_zero(grid_assigning, row, col):
                    valid_cells.add((row, col))
    print(f'\t\tmxn iteration: process_valid_cells() global valid_cells = {valid_cells}')
    return valid_cells


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

    if not isinstance(grid_assigning, np.ndarray):# or not isinstance(grid, np.ndarray):
        raise ValueError("Both grid_assigning and grid must be numpy arrays.")

    num_processes = 1  # 프로세스 수 설정
    pool = Pool(num_processes)
    rows_per_process = grid_assigning.shape[0] // num_processes

    # 각 프로세스에 데이터 분할
    #tasks = [(grid_assigning, grid, range(i * rows_per_process, (i + 1) * rows_per_process))
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

    if not isinstance(grid_assigning, np.ndarray):# or not isinstance(grid, np.ndarray):
        raise ValueError("Both grid_assigning and grid must be numpy arrays.")

    num_processes = 1  # 프로세스 수 설정
    pool = Pool(num_processes)
    rows_per_process = grid_assigning.shape[0] // num_processes

    # 각 프로세스에 데이터 분할
    # process_grid_get_valid_cells의 argument list
    #tasks = [(grid_assigning, grid, range(i * rows_per_process, (i + 1) * rows_per_process))
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
    grid = create_floorplan(m, n, k,floorshape)
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

def place_room_safe(floorplan, room_assigned_cells):
    print(f'place_room:')
    # Fill the floorplan with room assigned
    while room_assigned_cells:
        new_boundary_cells = set()
        # for every colored cell
        for cell in room_assigned_cells:
            print(f'\tcell in room_assigned_cells:{cell}')
            if choose_new_adjacent_cell(floorplan, cell):
                new_boundary_cells.update(get_valid_cell_coords_parallel(floorplan))
                print(f'\t\t\t if assign_zero_neighbour_value: updated new_boundary_cells={new_boundary_cells}')
        room_assigned_cells = new_boundary_cells
        print(f'\tinside while room_assigned_cells: {floorplan}')
    print(f'outside room_assigned_cells')
    return floorplan
