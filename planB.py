# working version saved
import numpy as np
import random
from PIL import Image, ImageDraw
from scipy.sparse import lil_matrix
from graph_operation import add_connection, delete_connection, delete_node
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
    print(f'\t\tcheck_valid_current_cell {cell}')
    row, col = cell[0], cell[1]
    if not is_empty_adjacent_exist(grid_assigning, row, col):
        # print(f'grid_assining={grid_assigning} of ({row}, {col}) = ??? error')
        # print(f'\t\tno neighbor_zero:({row},{col}) = {grid_assigning[row, col]} in grid_assigning\n\t\t{grid_assigning}')
        #print(f'\t\tcheck_valid_current_cell{cell} retuning False at ({row}, {col})')
        print(f'\t\t\treturnning False')
        return False
    print(f'\t\tretuning True at ({row}, {col})')
    return True

def can_be_assigned(grid_assigning, cell):
    print(f'\t\tcan_be_assigned {cell}')
    row, col = cell[0], cell[1]
    if not is_empty_adjacent_exist(grid_assigning, row, col):

        print(f'\t\t\treturnning False')
        return False
    print(f'\t\t\tretuning True at ({row}, {col})')
    return True


import random

import random


def place_room(floorplan, obtainable_cells):
    print('place_room:')
    active_cells = set(obtainable_cells)
    visited_cells = set()

    while active_cells:
        # Take a snapshot of active_cells to iterate over
        active_cells_current = active_cells.copy()
        active_cells.clear()  # Prepare for the next set of cells to activate

        for cell in active_cells_current:
            if floorplan[cell[0], cell[1]] == 0:
                continue  # Skip if the cell is already zero (not likely needed depending on initialization)
            valid_neighbors = collect_valid_adjacent_cells(cell, floorplan)
            for neighbor in valid_neighbors:
                if floorplan[neighbor[0], neighbor[1]] == 0:  # If the neighbor cell is zero
                    floorplan[neighbor[0], neighbor[1]] = floorplan[cell[0], cell[1]]  # Paint it
                    active_cells.add(neighbor)  # Add to active cells to propagate the color

        print(f'\tactive_cells:{active_cells}')

    return floorplan


def place_room_another_error_gpt(floorplan, obtainable_cells):
    print('place_room:')
    insulated_cells = set()
    valid_graph = {}
    active_cells = obtainable_cells.copy()

    while active_cells:
        # Take a snapshot of active_cells to iterate over
        active_cells_current = active_cells.copy()
        print(f'\tobtainable_cells:{obtainable_cells}\n\tactive_cells:{active_cells}')

        # Create a new set to track cells for the next iteration
        next_active_cells = set()

        for cell in active_cells_current:
            print(f'cell:{cell}')
            process_current_cell_valid(cell, floorplan, active_cells, insulated_cells)
            valid_neighbors = collect_valid_adjacent_cells(cell, floorplan, active_cells, insulated_cells)

            if not valid_neighbors:
                insulated_cells.add(cell)
                print(
                    f'\tno new_cell for {cell}={floorplan[cell[0], cell[1]]} is available: \n\t\tadded to insulated cells={insulated_cells}:{len(insulated_cells)}')
                continue  # Skip to the next iteration

            new_cell = random.choice(list(valid_neighbors)) if len(valid_neighbors) > 1 else valid_neighbors.pop()
            print(f'\tnew_cell: {new_cell}={floorplan[new_cell[0], new_cell[1]]}')

            if vacant(new_cell[0], new_cell[1], floorplan):
                floorplan[new_cell[0], new_cell[1]] = floorplan[cell[0], cell[1]]
                next_active_cells.add(new_cell)

            print(
                f'\tafter vacant ({new_cell}) = {floorplan[cell[0], cell[1]]} \n\t\tactive_cells:{active_cells}:{len(active_cells)}')
            print(f'\tcollect_valid_adjacent_cells {new_cell}')
            valid_neighbors = collect_valid_adjacent_cells(new_cell, floorplan, active_cells, insulated_cells)

            for valid_neighbor in valid_neighbors:
                add_connection(valid_graph, new_cell, valid_neighbor)

        # Update active_cells with the cells for the next iteration
        active_cells = next_active_cells
        print(f'active_cells={active_cells}\nactive_cells_current={active_cells_current}')
        print(f'\nobtainable_cells={obtainable_cells} {len(obtainable_cells)}\n\tfloorplan=\n\t{floorplan}')

    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan


def place_room_error(floorplan, obtainable_cells):
    print(f'place_room:')
    insulated_cells = set()
    valid_graph = {}
    # Fill the floorplan with room assigned
    active_cells_current = active_cells = obtainable_cells.copy()

    print(f'\tobtainable_cells:{obtainable_cells}\n\tactive_cells:{active_cells}')
    while active_cells:
        # print(f'for cell in room_assigned_cells')
        for cell in active_cells_current:
            print(f'cell:{cell}')
            process_current_cell_valid(cell, floorplan, active_cells, insulated_cells)
            valid_neighbors = collect_valid_adjacent_cells(cell, floorplan, active_cells, insulated_cells)  #
            if not valid_neighbors:
                insulated_cells.add(cell) # no longer neibhbor cells to color is available
                print(f'\tno new_cell for {cell}={floorplan[cell[0], cell[1]]} is available: \n\t\tadded to insulated cells={insulated_cells}:{len(insulated_cells)}')
                if cell in active_cells: active_cells.remove(cell)
                print(f'removing:{cell} from active_cells active_cells = {active_cells}:{len(active_cells)}')
                print(f'\tno cell ')
                break
            new_cell = random.choice(list(valid_neighbors)) if len(valid_neighbors) > 1 else valid_neighbors.pop()
            print(f'\tnew_cell: {new_cell}={floorplan[new_cell[0], new_cell[1]]} ')# \n\t\tactive_cells:{active_cells}:{len(active_cells)}')

            if vacant(new_cell[0], new_cell[1], floorplan): #todo choose_new...cells()에서 이미 valid_neighbor_cells에서만 선택하므로 이것은 필요없을 덧하다. 그런데 여기서 active_cells과 insulated_cells를 다시 계산하는 것으로 보아서 이걸 다시 봐야 되긴 할 것 같다. 필요없을 듯 다시봐도
                floorplan[new_cell[0], new_cell[1]] = floorplan[cell[0], cell[1]]
                active_cells.add(new_cell)
            print(f'\tafter vacant ({new_cell}) = {floorplan[cell[0], cell[1]]} \n\t\tactive_cells:{active_cells}:{len(active_cells)}')
            # 새 셀과 인접한 셀이 valid한지 체크하고
            print(f'\tcollect_valid_adjacent_cells {new_cell}')
            valid_neighbors = collect_valid_adjacent_cells(new_cell, floorplan, active_cells, insulated_cells) # todo 이걸 왜하냐하면 valid_neighbor(변수바꿔야) 가 없으면 검사하지 말아야하기 때문에 insulated_cells에 담거나 해야 한다. 가 없으면 이건 왜 필요한지 몰라, 해당 neighver의 모든 neighboer들이 valid한지 확인하는 건데 # 이 함수 안쪽에 해당 이웃의 valid 셀을 가진다.#
            for valid_neighbor in valid_neighbors:
                add_connection(valid_graph, new_cell, valid_neighbor) # todo 이걸 왜 하는지 까먹었다. 이미 added되어서 다시는 valid 계산할 필요가 없는데? 왜지? 왜지?
            # print(f'\tvalid_graph={valid_graph}')
            print(f'\tafter collect_candidate_set for {new_cell} {floorplan[cell[0], cell[1]]}, \n\t\tactive_cells:{active_cells}:{len(active_cells)}')
            active_cells_current = active_cells.copy()
        print(f'active_cells={active_cells}\nactive_cells_current={active_cells_current}')
        print(f'\nobtainable_cells={obtainable_cells} {len(obtainable_cells)}\n\tfloorplan=\n\t{floorplan}')#floorplan이 업데이트되지 않음
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan

# 만약에 assign 가능한 이웃 셀이하나도 없으면 insulated_cells를 업데이트하고, active_cell 에서도 제거한다.
def process_current_cell_valid(cell, floorplan, active_cells, insulated_cells):
    print(f'\t\tprocess_current_cell_valid:{cell}')# in active_cells:{active_cells}')
    # f not check_valid_current_cell(floorplan, cell):  # 현재 셀 확인
    if not is_empty_adjacent_exist(floorplan, cell[0], cell[1]):
        if cell in active_cells: active_cells.remove(cell)
        insulated_cells.add(cell)
        if cell in active_cells: active_cells.remove(cell)
        print(f'\t\t{cell} not valid. added {cell} to insulated_cells={insulated_cells}\n\t\tremoved {cell} from active_cells={active_cells}')
        print(f'\t\treturning False')
        return False
    
    print(f'\t\tprocess_current_cell_valid{cell} returning True ')#with active_cells:{active_cells}')
    return True

# 주어진 셀의 네 이웃이 모두 valid 한지 확인
def collect_candidate_set(cell, grid_assigning, active_cells, insulated_cells):
    print(f'\t\tcollect_valid_adjacent_cells: {cell}')
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row,col = cell[0], cell[1]
    valid_neighbor_cells = set()
    for dx, dy in directions:
        new_row, new_col = row+dy, col+dx
        if 0 <= new_row <grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]: # 범위내에 있으면
            print(f'\t\t\t({new_row}, {new_col}) inside')
            if vacant(new_row, new_col, grid_assigning):
                print(f'\t\t\t\tgrid_assigning[{new_row}, {new_col}]={grid_assigning[new_row, new_col]}.. so vacant.. adding neighbor_cells')
                neighbor_cell = (new_row, new_col)
                valid_neighbor_cells.add(neighbor_cell)
            else: # cell already assigned
                neighbor_cell = (new_row, new_col)
                # insulated_cells.add(neighbor_cell)
            #process_current_cell_valid(neighbor_cell, grid_assigning, active_cells, insulated_cells) #active_cell에서 제거하는 루틴을 짭시다.
    print(f'\t\tcollect_valid_adjacent_cells:{cell} returning valid_neighbor_cells{valid_neighbor_cells}')
    return valid_neighbor_cells


def collect_candidate_set(cell, floorplan):
    # Mock implementation to collect valid adjacent cells
    neighbors = [
        (cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]),
        (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)
    ]
    return [n for n in neighbors if
            0 <= n[0] < len(floorplan) and 0 <= n[1] < len(floorplan[0])]  # Assuming rectangular grid




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
                status_dict[(row, col)] = is_empty_adjacent_exist(grid_assigning, row, col)
                if status_dict[(row, col)]:
                    working_cells.add((row, col))

    return working_cells
# 빈 인접노드가 하나라도 있으면 True
def is_empty_adjacent_exist(grid_assigning, row, col):
    print(f'\t\t\t\tis_empty_adjacent_exist({row}, {col})')
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        print(f'\t\t\t\t\tneighbor = ({new_row}, {new_col})')
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            print(f'\t\t\t\t\t\tinside')
            if grid_assigning[new_row, new_col] == 0:
                print(f'\t\t\t\t\t\tfloorplan[{new_row, new_col}]==0 (empty)\n\t\t\t\tis_empty_adjacent_exist returning True')
                return True
    print(f'\t\t\t\tcondition inside and 0 not met returning False')
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

def choose_new_adjacent_cell_from_valid_neighbor_cells(valid_neighbor_cells):
    print(f'\tchoose_new_adjacent_cell_from_valid_neighbor_cells:{valid_neighbor_cells}')
    new_cell = random.choice(list(valid_neighbor_cells)) if valid_neighbor_cells else None
    return new_cell
def choose_new_adjacent_cell(floorplan, cell):
    print(f'\tchoose_new_adjacent_cell:{cell}')
    row, col = cell
    # Ensure we are within the floorplan and the cell has not been colored yet
    if floorplan[row, col] <= 0:  # Adjusted condition to ensure we're targeting uncolored cells
        return False

    valid_offsets = get_valid_neighbor(row, col, floorplan)

    if valid_offsets:
        dy, dx = random.choice(valid_offsets)
        floorplan[row + dy, col + dx] = floorplan[row, col]  # Color the neighbor
        return (row+dy, col+dx)
        # return True
    return None


# 네 개의 이웃 중, 배치도 상에 위치하고 아직 값이 초기상태인 이웃을 구해라.
def get_valid_neighbor(row,col, floorplan):

    valid_offsets = [(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 네 개의 이웃이
                     if
                     0 <= row + dy < floorplan.shape[0] and \
                     0 <= col + dx < floorplan.shape[1] and \
                     floorplan[row + dy, col + dx] == 0]  # 배치도 상에 위치하고 값이 아직 assign되지 않았을 때
    return valid_offsets


import numpy as np
from multiprocessing import Pool


def process_grid_get_valid_cells_safe(grid_assigning, grid, row_range):
    working_cells = set()
    for row in row_range:
        for col in range(grid.shape[1]):
            #if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0 :
                if is_empty_adjacent_exist(grid_assigning, row, col):
                    working_cells.add((row, col))
    return working_cells
# 할당가능한 이웃이 남아있는 셀 집합을 구한다.
# def process_grid_segment(grid_assigning, row_range): #rename
def process_grid_get_valid_cells(grid_assigning, row_range):
    valid_cells = set()
    for row in row_range:
        for col in range(grid_assigning.shape[1]):
            if grid_assigning[row, col] > 0 :
                if is_empty_adjacent_exist(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells
# grid 의 모든 cell 을 조사해서 valid_cells 만 리턴
def process_valid_cells(grid_assigning, insulated_cells, row_range):
    print(f'\tprocess_valid_cells:')
    valid_cells = set()
#    print(f'insulated_cells={insulated_cells} in process_valid_cells')
    for row in row_range:
        for col in range(grid_assigning.shape[1]):
            #if grid_assigning[row, col] > 0 and grid[row][col] == 1:
            if grid_assigning[row, col] > 0 :
                if is_empty_adjacent_exist(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells

def vacant(new_row, new_col, grid_assigning):
    print(f'\t\t\tvacant({new_row}, {new_col})')
    #dx, dy는 이미 계산된 값이므로 필요없음
    # 행렬 인덱스가 0보다 크고 그리드 행렬 크기보다 작을 때
    if grid_assigning[new_row, new_col] == 0: #값 체크
        print(f'\t\t\t\t{new_row},{new_col} = {grid_assigning[new_row, new_col] } returning True')
        return True
    else:
        print(f'\t\t\t\t{new_row},{new_col} = {grid_assigning[new_row, new_col] } returning False')
        # print(f'\t\t\t\treturning False')
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

def place_room_save_05042012(floorplan, obtainable_cells):
    print(f'place_room:')
    insulated_cells = set()
    # Fill the floorplan with room assigned
    while obtainable_cells:
        active_cells = set()
        print(f'\while obtainable_cells {obtainable_cells}:{len(obtainable_cells)} ')
        # print(f'for cell in room_assigned_cells')
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

                    print(f'\t\tafter check_valid_cuirrent_cell({cell}) active_cells:{active_cells},{len(active_cells)}')
                    # active_cells.update(new_valid_set) # todo tried to remove debug step 1. infact process_valid_cell에서 boundary cell을 새로고침한다.
                    # print(f'active_cells after update{active_cells}')
                    # active_cells.update(get_valid_cell_coords_parallel(floorplan, insulated_cells))
                    print(f'\t\tnew_cell: {new_cell}={floorplan[cell[0], cell[1]]}, \n\t\tactive_cells:{active_cells}:{len(active_cells)}')
                else: #만일 현재 셀에 assign 가능한 이웃이 없다면
                    insulated_cells.add(cell) # no longer neibhbor cells to color is available
                    print(f'\t\tinsulated cells={insulated_cells}:{len(insulated_cells)}')
                    if cell in active_cells: active_cells.remove(cell)
                    print(f'removing:{cell} from active_cells {active_cells}:{len(active_cells)}')
                    print(f'\t\tno cell for {cell}={floorplan[cell[0], cell[1]]}')
        obtainable_cells = active_cells
        print(f'\n\tobtainable_cells={obtainable_cells} {len(obtainable_cells)}\n\tfloorplan=\n\t{floorplan}')
    print(f'----')
    print(f'insulated_cells={insulated_cells}: total {len(insulated_cells)}')
    return floorplan
