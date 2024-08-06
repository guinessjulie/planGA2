from PIL import Image, ImageDraw
import constants
import numpy as np


####################################################################################

def dict_value_to_coordinates(floorplan):

    # 딕셔너리 초기화
    value_to_coordinates = {}

    # 각 값의 좌표를 찾아서 딕셔너리에 추가
    for value in range(1, np.max(floorplan) + 1):
        coordinates = np.argwhere(floorplan == value)
        value_to_coordinates[value] = [tuple(coord) for coord in coordinates]

    return value_to_coordinates

def get_color_coordinates(grid):
    """
    그리드에서 각 색상별로 해당 색상의 좌표를 리스트로 반환하는 함수.

    Parameters:
    grid (list of lists): 2차원 그리드. 각 요소는 색상 값을 나타내는 정수.

    Returns:
    dict: 각 색상별 좌표 리스트를 포함하는 딕셔너리. 키는 색상 값, 값은 해당 색상의 좌표 리스트.
    """
    color_coordinates = {}  # 색상별 좌표를 저장할 딕셔너리

    for row_idx, row in enumerate(grid):
        for col_idx, color in enumerate(row):
            if color not in color_coordinates:
                color_coordinates[color] = []
            color_coordinates[color].append((row_idx, col_idx))

    return color_coordinates


def get_color_at(grid, row, col):
    """

    :param grid : list of lists consists of row,col. int element represent color
    :param row:  int
    :param col: int
    :return:  color value of element (row,col) of grid
    """
    if row < len(grid) and row >= 0 and col >= 0 and col < len(grid[0]):
        return grid[row][col]
    else:
        return None

def same_room_cells(floorplan, cell):
    return np.argwhere(floorplan == floorplan[cell])


# 좌표를 그리드로 변환
### 좌표 coordinates =  [
#     (0, 0), (0, 1), (0, 2), (0, 3),
#     (1, 0), (1, 1), (1, 2), (1, 3),
#     (2, 0), (2, 1), (2, 2),
#     (3, 0), (3, 1), (3, 2),
#                     (4, 2),
#                     (5, 2), (5, 3)
# ]
### 리턴 grid =
def coordinates_to_grid(coordinates):
    # 좌표 리스트에서 가장 큰 행과 열 값 찾기
    max_row = max(coordinates, key=lambda x: x[0])[0]
    max_col = max(coordinates, key=lambda x: x[1])[1]

    # 그리드 초기화 (가장 큰 인덱스에 +1 해서 크기를 결정)
    grid = [[0 for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # 주어진 좌표에 해당하는 부분을 1로 설정
    for row, col in coordinates:
        grid[row][col] = 1

    return grid


# the other way around
def grid_to_coordinates(grid):
    coordinates = []

    for x, row in enumerate(grid):
        for y, value in enumerate(row):
            if value == 1:
                coordinates.append((x, y))

    return coordinates


def expand_grid(grid, scale=2):
    """
    Expand the given grid by the specified scale factor.
    Each cell in the original grid is expanded to a scale x scale block in the new grid.

    Parameters:
    grid (list of list of int): The input grid to expand.
    scale (int): The scale factor to expand each cell by. Default is 2.

    Returns:
    list of list of int: The expanded grid.
    """
    m, n = len(grid), len(grid[0])
    expanded_grid = np.zeros((m * scale, n * scale), dtype=int)

    for i in range(m):
        for j in range(n):
            for k in range(scale):
                for l in range(scale):
                    expanded_grid[i * scale + k][j * scale + l] = grid[i][j]

    return expanded_grid.tolist()


####################### control  / test functiion ####################
def main_loop(modules, arg=None):
    try:
        while True:
            print(f'testing plan_utils')
            for key, testing_module in modules.items():
                print(f'[{key}]: {testing_module.__name__} ')
            module = input('Choose testing module: ')
            if module == '9':
                break
            if module in modules:
                if arg is not None:
                    modules[module](arg)
                else:
                    modules[module]()

    except KeyboardInterrupt:
        print('Finished')


def print_grid(grid, title='input'):
    print(title)
    for row in grid:
        for el in row:
            print(el, end=' ')
        print()


def exe_coordinates_to_grid():
    coord = constants.coordinates
    print(f'input coordinates = {coord}')
    grid = coordinates_to_grid(coord)
    print(f'output grid =')
    print_grid(grid)


def print_dics(pairs, title='output'):
    if title == 'output': print()
    for key, val in pairs.items():
        print(f'{key}:{val}')


def exe_get_color_coordinates():
    grid = constants.test_grid
    print_grid(grid)
    colors = get_color_coordinates(grid)
    print_dics(colors, 'output')


def exe_get_color_at():
    grid = constants.test_grid
    print_grid(grid)
    row = int(input('select row in grid:'))
    col = int(input('select col in grid:'))

    color = get_color_at(grid, row, col)
    print(f'color[{row}, {col}]={color}')


def finish():
    pass


def main():
    modules = {
        "1": exe_coordinates_to_grid,
        "2": exe_get_color_coordinates,
        "3": exe_get_color_at,
        "9": finish
    }

    main_loop(modules)


if __name__ == '__main__':
    main()
