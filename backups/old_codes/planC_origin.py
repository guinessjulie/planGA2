# working version saved
import numpy as np
import random
from PIL import Image, ImageDraw
from scipy.sparse import lil_matrix

import plan_utils
from GridDrawer import GridDrawer


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
    #print(f'initialize_floorplan returned: {initialized_grid}, input empty_grid = {empty_grid}')
    # Initialize boundary cells based on the updated empty_grid
    # initial_cells = get_valid_cell_coords_parallel(initialized_grid)
    print(f'initial_cells:{initial_cells}\n{initialized_grid}')
    floorplan = place_room(initialized_grid, initial_cells)
    return floorplan

#  현재 셀이 valid 한가
def check_valid_current_cell(grid_assigning, cell):

    row, col = cell[0], cell[1]
    if not has_neighbor_zero(grid_assigning, row, col):
        print(f'\t\tFailed check_valid_current_cell:{row},{col} in grid\n{grid_assigning}')
        return False
    return True


def place_room(floorplan, obtainable_cells):
    print(f'place_room:')
    insulated_cells = set()
    # Fill the floorplan with room assigned
    while obtainable_cells:
        active_cells = set()
        print(f'while obtainable_cells {obtainable_cells}:{len(obtainable_cells)} ,  active_cells={active_cells}:{len(active_cells)}')
        print(f'for cell in room_assigned_cells')
        for cell in obtainable_cells:
            if cell not in insulated_cells:
                print(f'\tcell:{cell}')
                new_cell = choose_new_adjacent_cell(floorplan, cell)
                if new_cell:
                    # update the active_cells set with union of itself and returned value of process_valid_cells
                    active_cells = process_valid_cells(floorplan, insulated_cells, range(floorplan.shape[0]))
                    print(f'\t\treturned active_cells  from process_valid_cells = {active_cells}:{len(active_cells)}')
                    if not check_valid_current_cell(floorplan, cell):
                        if cell in active_cells: active_cells.remove(cell)
                        insulated_cells.add(cell)
                    # active_cells.update(new_valid_set) # todo tried to remove debug step 1. infact process_valid_cell에서 boundary cell을 새로고침한다.
                    # print(f'active_cells after update{active_cells}')
                    # active_cells.update(get_valid_cell_coords_parallel(floorplan, insulated_cells))
                    print(f'\t\tnew_cell: {new_cell}={floorplan[cell[0], cell[1]]}, \n\t\tactive_cells:{active_cells}:{len(active_cells)},  \n\t\tobtainable_cells {obtainable_cells}:{len(obtainable_cells)} ')
                else:
                    insulated_cells.add(cell) # no longer neibhbor cells to color is available
                    print(f'\t\tinsulated cells={insulated_cells}:{len(insulated_cells)}')
                    if cell in active_cells: active_cells.remove(cell)
                    print(f'removing:{cell} from active_cells {active_cells}:{len(active_cells)}')
                    print(f'\t\tno cell for {cell}={floorplan[cell[0], cell[1]]}')
        obtainable_cells = active_cells
        print(f'\n\tobtainable_cells={obtainable_cells} {len(obtainable_cells)}\n\tfloorplan=\n\t{floorplan}')
        print(f'----for loop finished----')
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


def get_valid_cell_coords_optim(grid_assigning, grid):
    rows, cols = grid_assigning.shape
    working_cells = set()
    status_dict = {}  # 셀 상태를 저장하는 사전

    # 초기 상태 설정
    for row in range(rows):
        for col in range(cols):
            if grid_assigning[row, col] > 0 and grid[row][col] == 1:
                status_dict[(row, col)] = has_neighbor_zero(grid_assigning, row, col)
                if status_dict[(row, col)]:
                    working_cells.add((row, col))

    return working_cells

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
