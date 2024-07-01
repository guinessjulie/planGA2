import numpy as np
import random
from PIL import Image, ImageDraw
from scipy.sparse import lil_matrix

import plan_utils
from GridDrawer import GridDrawer


def create_floorplan_save(m, n, k, floorplan):
    grid = initialize_floorplan(floorplan, m, n, k)
    print(f'k-color placed: {grid}')
    # Initialize boundary cells based on the updated grid
    initial_cells = get_valid_cell_coords(grid, floorplan)
    print(f'initial_cells:{initial_cells}')
    # Fill the grid
    update_assigned_cells(initial_cells, floorplan, grid)
    return grid


def update_assigned_cells(assigned_cells, floorplan, grid) -> object:
    while assigned_cells:
        new_boundary_cells = set()
        for cell in assigned_cells:
            if assign_zero_neighbour_value(grid, cell):
                new_boundary_cells.update(get_valid_cell_coords_optim(grid, floorplan))
        assigned_cells = new_boundary_cells
        print(f'while assigned_cells: {assigned_cells}')
    print(f'outside the while assigned_cells:{assigned_cells}')

def create_floorplan(m, n, k, empty_grid):
    initialized_grid = initialize_floorplan(empty_grid, m, n, k)

    #print(f'initialize_floorplan returned: {initialized_grid}, input empty_grid = {empty_grid}')
    # Initialize boundary cells based on the updated empty_grid
    initial_cells = get_valid_cell_coords_optim(initialized_grid, empty_grid)
    print(f'initial_cells:{initial_cells}')
    print(f'initialized_grid:{initialized_grid}')
    floorplan = place_room(initialized_grid, empty_grid, initial_cells)
    return floorplan


def place_room(floorplan, grid, room_assigned_cells):
    # Fill the floorplan with room assigned
    while room_assigned_cells:
        new_boundary_cells = set()
        for cell in room_assigned_cells:
            if assign_zero_neighbour_value(floorplan, cell):
                new_boundary_cells.update(get_valid_cell_coords_optim(floorplan, grid))
        room_assigned_cells = new_boundary_cells
        print(f'while room_assigned_cells: {floorplan}')
    print(f'outside room_assigned_cells')
    return floorplan


def initialize_floorplan(empty_grid, m, n, k):
    np_converted_grid = list_grid_to_np_array(empty_grid, m, n)
    initial_color_grid = place_initial_k_color(np_converted_grid, k, m, n)
    print(f'inside the initialize_floorplan')
    print(f'np_converted_grid={np_converted_grid}, initial_color_grid = {initial_color_grid}')
    return initial_color_grid


def place_initial_k_color(grid_arr, k, m, n):
    # Randomly place k colors within the floorplan
    colors_placed = 0
    coloring_grid = grid_arr.copy()
    while colors_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if coloring_grid[row, col] == 0:  # Ensure the cell is within the floorplan and uncolored
            coloring_grid[row, col] = colors_placed + 1
            colors_placed += 1
    return coloring_grid


"""
input grid[][]를 입력받아 np.array 출력
"""
def list_grid_to_np_array(grid, m, n):
    grid_array = np.full((m, n), -1, dtype=int)  # Initialize all cells as -1
    # Set cells within the floorplan to 0 (uncolored)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                grid_array[i, j] = 0
    return grid_array


"""
"""
def get_boundary_cells_save(grid, floorplan):
    # Examine 4-way neighbours of the cell given by row, col in grid
    # return True if a one of the neighbour cell not assigned a value(color or room number) yet
    # return False all none of any neighbour cell is empty. No empty cell is available
    # when return False no neighbour is available
    def has_neighbor_zero(grid, row, col):
        if grid[row, col] == -1:  # Ignore cells outside the floorplan
            return False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1] and grid[new_row, new_col] == 0:
                return True
        return False

    boundary_cells = set()
    for row in range(grid.shape[0]): # for each rows
        for col in range(grid.shape[1]): # for each cols
            if (floorplan[row][col] == 1 #
                    and grid[row, col] > 0
                    and has_neighbor_zero(grid, row, col)):
                boundary_cells.add((row, col))
    return boundary_cells
"""
grid 내부 cell을 대상으로 
이웃 중  빈 셀이 있고(has_neighbor_zero()), 
유효하고(grid[][]==1, 컬러가 이미 assign된) 셀들의 집합을 리턴
"""
def get_valid_cell_coords(grid_assigning, grid):
    # Examine 4-way neighbours of the cell given by row, col in grid
    # return True if a one of the neighbour cell not assigned a value(color or room number) yet
    # return False none of any neighbour cell is empty. No empty cell is available
    # when return False no neighbour is available
    def has_neighbor_zero(grid_assigning, row, col):
        if grid_assigning[row, col] == -1:  # Ignore cells outside the floorplan
            return False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1] and grid_assigning[new_row, new_col] == 0:
                return True
        return False

    working_cells = set()

    # print(f'grid:{grid}')
    for row in range(grid_assigning.shape[0]): # for each rows
        for col in range(grid_assigning.shape[1]): # for each cols
            if (grid[row][col] == 1 #
                    and grid_assigning[row, col] > 0
                    and has_neighbor_zero(grid_assigning, row, col)):
                working_cells.add((row, col))
    return working_cells


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
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False

# NumPy 대신 Scipy의 희소 행렬 사용
def create_sparse_grid(rows, cols):
    return lil_matrix((rows, cols), dtype=int)


def get_valid_cell_coords_cache(grid_assigning, grid):
    # Examine 4-way neighbours of the cell given by row, col in grid
    # return True if a one of the neighbour cell not assigned a value(color or room number) yet
    # return False none of any neighbour cell is empty. No empty cell is available
    # when return False no neighbour is available
    cache={}
    def has_neighbor_zero(grid_assigning, row, col):
        # return cached result
        if (row, col) in cache:
            return cache[(row, col)]

        if grid_assigning[row, col] == -1:  # Ignore cells outside the floorplan
            cache[(row, col)] = False
            return False

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dy, col + dx

            if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
                if grid_assigning[new_row, new_col] == 0:
                    cache[(row, col)] = True
                    return True

        cache[(row, col)] = False
        return False

    working_cells = set()

    for row in range(grid_assigning.shape[0]): # for each rows
        for col in range(grid_assigning.shape[1]): # for each cols
            if grid_assigning[row, col] > 0 and grid[row][col] == 1:
                if has_neighbor_zero(grid_assigning, row, col):
                    working_cells.add((row, col))

    return working_cells
import numpy as np
from scipy.sparse import lil_matrix

def get_valid_cell_coords(grid_assigning, grid):
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
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False

# NumPy 대신 Scipy의 희소 행렬 사용
def create_sparse_grid(rows, cols):
    return lil_matrix((rows, cols), dtype=int)


def assign_zero_neighbour_value(grid, cell):
    row, col = cell
    # Ensure we are within the floorplan and the cell has not been colored yet
    if grid[row, col] <= 0:  # Adjusted condition to ensure we're targeting uncolored cells
        return False

    valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if
                     0 <= row + dy < grid.shape[0] and 0 <= col + dx < grid.shape[1] and grid[row + dy, col + dx] == 0]

    if valid_offsets:
        dy, dx = random.choice(valid_offsets)
        grid[row + dy, col + dx] = grid[row, col]  # Color the neighbor
        return True
    return False


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
