import numpy as np


def place_room(floorplan, obtainable_cells):
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


def get_unique_values(floorplan, cells):
    unique_values = set()
    for cell in cells:
        unique_values.add(floorplan[cell])
    return unique_values


def check_valid_current_cell(grid_assigning, cell):
    row, col = cell
    if not has_neighbor_zero(grid_assigning, row, col):
        return False
    return True


def all_active_neighbors(cell, floorplan):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell
    adjs = set()
    for dx, dy in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < floorplan.shape[0] and 0 <= new_col < floorplan.shape[1]:
            if floorplan[new_row, new_col] > 0:
                adjs.add((new_row, new_col))
    return adjs


def collect_candidate_set(cell, grid_assigning):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    row, col = cell
    valid_neighbor_cells = set()
    for dx, dy in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                neighbor_cell = (new_row, new_col)
                valid_neighbor_cells.add(neighbor_cell)
    return valid_neighbor_cells


def has_neighbor_zero(grid_assigning, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in directions:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid_assigning.shape[0] and 0 <= new_col < grid_assigning.shape[1]:
            if grid_assigning[new_row, new_col] == 0:
                return True
    return False


def process_valid_cells(grid_assigning, insulated_cells, row_range):
    valid_cells = set()
    for row in row_range:
        for col in range(grid_assigning.shape[1]):
            if grid_assigning[row, col] > 0:
                if has_neighbor_zero(grid_assigning, row, col):
                    valid_cells.add((row, col))
    return valid_cells

