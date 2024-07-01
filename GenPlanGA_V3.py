# Augmentation

import random
import copy

# utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re
from math import ceil
from datetime import datetime
from PIL import Image
import os

def plot_merged_imaage_in_folder(directory = 'F:\\2024\\dev\\planGA',\
                                 grid_image_path = 'F:\\2024\\dev\\planGA\\grid_img.png'):

    # Parameters

    interval = 0.1
    # Space between images
    grid_size = (5, 2)  # Grid size as (columns, rows)

    # Load all images
    images = [Image.open(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.png')]
    if not images:
        raise ValueError("No PNG images found in the directory.")

    # Assume all images are the same size
    image_width, image_height = images[0].size

    # Calculate total grid size
    total_width = int(((image_width * grid_size[0]) + (interval * (grid_size[0] - 1))) - 20)
    total_height = int((image_height * grid_size[1]) + (interval * (grid_size[1] - 1)))

    # Create a new image with a white background
    grid_image = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images into the grid
    for index, image in enumerate(images):
        column = index % grid_size[0]
        row = index // grid_size[0]
        x = int(column * (image_width + interval))
        y = int( row * (image_height + interval))
        grid_image.paste(image, (x, y))

    # Save or show the grid image

    grid_image.save(grid_image_path)
    # Or display it directly if you're using a Jupyter notebook
    # grid_image.show()

'''
주어진 행렬의 크기에 대해 각 셀이 가질 수 있는 같은 색의 최대 이웃 수를 계산하고, 
그 결과를 같은 크기의 행렬로 반환. 
여기서 "이웃"은 상, 하, 좌, 우 방향으로 인접한 셀을 의미하며, 
각 셀의 위치에 따라 최대 이웃 수가 달라.
중앙에 위치한 셀은 최대 4개의 이웃(상, 하, 좌, 우)을 가질 수 있고, 
가장자리에 위치한 셀은 3개, 모서리에 위치한 셀은 2개의 이웃을 가질 수 있다.'''
def calculate_max_neighbors_matrix(rows, cols):
    # 결과 행렬 초기화
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            neighbors = 4  # 중앙 셀 기준으로 최대 이웃 수

            # 모서리 검사
            if row in [0, rows - 1] and col in [0, cols - 1]:
                neighbors = 2
            # 가장자리 검사 (단, 모서리는 제외)
            elif row in [0, rows - 1] or col in [0, cols - 1]:
                neighbors = 3

            matrix[row][col] = neighbors

    # 가장자리와 모서리의 이웃 수 조정
    for row in [0, rows - 1]:
        for col in [0, cols - 1]:
            matrix[row][col] = 2  # 모서리 셀
        for col in range(1, cols - 1):
            matrix[row][col] = 3  # 상단 및 하단 가장자리 셀
    for col in [0, cols - 1]:
        for row in range(1, rows - 1):
            matrix[row][col] = 3  # 좌측 및 우측 가장자리 셀

    return matrix

'''
# 3x5 행렬 예시
rows, cols = 3, 5
max_neighbors_matrix = calculate_max_neighbors_matrix(rows, cols)

# 결과 출력
for row in max_neighbors_matrix:
    print(row)

# output
[2, 3, 3, 3, 2]
[3, 4, 4, 4, 3]
[2, 3, 3, 3, 2]
'''



def count_unique_values(matrix):
    unique_values = set()
    for row in matrix:
        for value in row:
            unique_values.add(value)
    return len(unique_values)

# 예시 매트릭스
'''matrix = [
    [1, 2, 3],
    [4, 5, 1],
    [1, 3, 4]
]

# 고유한 값의 개수를 구함
unique_count = count_unique_values(matrix)
print("Unique values count:", unique_count)
'''


def count_neighbors(grid):
    # 그리드의 행과 열의 크기를 가져옵니다.
    rows, cols = len(grid), len(grid[0])

    # 이웃을 확인할 8가지 방향을 정의합니다. (상, 하, 좌, 우, 대각선 4방향)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 결과를 저장할 리스트를 초기화합니다.
    neighbor_count = [[0 for _ in range(cols)] for _ in range(rows)]

    # 그리드의 각 셀을 순회하며 이웃을 세어줍니다.
    for row in range(rows):
        for col in range(cols):
            # 현재 셀의 값(색)을 가져옵니다.
            current_value = grid[row][col]
            # 이웃의 수를 세기 위한 카운터를 초기화합니다.
            count = 0
            for d in directions:
                # 이웃 셀의 위치를 계산합니다.
                nr, nc = row + d[0], col + d[1]
                # 이웃 셀이 그리드 범위 내에 있고, 같은 색(값)인지 확인합니다.
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == current_value:
                    count += 1
            # 계산된 이웃의 수를 결과 리스트에 저장합니다.
            neighbor_count[row][col] = count

    return neighbor_count

'''
# 주어진 그리드
grid = [[3, 3, 3, 1, 1],
        [3, 3, 2, 1, 1],
        [3, 2, 2, 1, 1]]

# 같은 색의 이웃 개수를 세어 출력합니다.
neighbor_counts = count_neighbors(grid)
for row in neighbor_counts:
    print(row)
'''

def generate_filename_with_timestamp(prefix: str, extension: str, steps=None) -> str:
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    if steps is not None:
        time_str = steps
    filename = f"{prefix}_{time_str}.{extension}"
    return filename


# (gen,png, 1) 인경우 gen_01.png 리턴, gen, pnt, 11인 경우 gen_11.png
def generate_filename_with_sequence(prefix: str, extension: str, steps=None) -> str:
    pattern = r'^[0-9]$'
    filename = '0' + str(steps) if re.match(pattern, str(steps)) else str(steps)
    filename = f"{prefix}_{steps}.{extension}"
    return filename


def plot_grid_single(grid, name='temp.png'):
    plt.figure()
    colors = mpl.colormaps['Accent']
    plt.matshow(grid, cmap=colors)
    plt.axis('off')
    plt.savefig(name)
    plt.show()


def plot_grids(grids, m, n, k, rows=None, cols=5, fitness_scores=None, savefilename=None):
    """
    Creates a plot with multiple subplots arranged in specified rows and columns, each displaying a grid.

    Parameters:
    - m (int): The number of rows in each grid.
    - n (int): The number of columns in each grid.
    - k (int): The number of color groups in each grid.
    - rows (int): The number of subplot rows.
    - cols (int): The number of subplot columns.
    """
    if rows is None:
        cols = int(cols)
        rows = int(ceil(len(grids) / cols))
    if cols is None:
        rows = int(rows)
        cols = int(ceil(len(grids) / rows))
    rows, cols = int(rows), int(cols)
    fig, axes = plt.subplots(rows, cols, figsize=(n * cols, m * rows))
    xy_coords = [(x, y) for x in range(rows) for y in range(cols)]
    for ij, grid in enumerate(grids):
        colors = mpl.colormaps['Accent']
        ax = axes[xy_coords[ij]]
        ax.matshow(grid, cmap=colors)
        if fitness_scores is not None:
            ax.set_title(f'{fitness_scores[ij]:.2f}', fontsize=20)
        ax.axis('off')

    if savefilename is not None:
        plt.savefig(savefilename)
    plt.show()


# initial Population

def has_neighbor_zero(grid, row, col):
    nrows, ncols = len(grid), len(grid[0])
    # Define relative positions of neighbors (up, down, left, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in neighbors:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
            return True
    return False

def assgin_zero_neighbour_value(grid, cell):
    nrows, ncols = grid.shape  # 그리드의 행과 열 크기
    row, col = cell
    # 이웃의 상대적 위치: 위, 아래, 왼쪽, 오른쪽
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    better_nonzero_offsets = [
        offset for i, offset in enumerate(adjacent_offsets)
        if 0 <= row + offset[0] < nrows and 0 <= col + offset[1] < ncols and grid[row + offset[0], col + offset[1]] == 0
    ]
    if not better_nonzero_offsets:
        return False
    selected_neighbour_offset = random.choice(better_nonzero_offsets)
    selected_neighbour_loc = (row + selected_neighbour_offset[0], col + selected_neighbour_offset[1])
    grid[selected_neighbour_loc] = grid[row, col]


def count_identical_neighbors(grid):
    # 그리드의 행과 열의 크기를 가져옵니다.
    rows, cols = len(grid), len(grid[0])

    # 이웃을 확인할 8가지 방향을 정의합니다. (상, 하, 좌, 우, 대각선 4방향)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 결과를 저장할 리스트를 초기화합니다.
    neighbor_count = [[0 for _ in range(cols)] for _ in range(rows)]

    # 그리드의 각 셀을 순회하며 이웃을 세어줍니다.
    for row in range(rows):
        for col in range(cols):
            # 현재 셀의 값(색)을 가져옵니다.
            current_value = grid[row][col]
            # 이웃의 수를 세기 위한 카운터를 초기화합니다.
            count = 0
            for d in directions:
                # 이웃 셀의 위치를 계산합니다.
                nr, nc = row + d[0], col + d[1]
                # 이웃 셀이 그리드 범위 내에 있고, 같은 색(값)인지 확인합니다.
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == current_value:
                    count += 1
            # 계산된 이웃의 수를 결과 리스트에 저장합니다.
            neighbor_count[row][col] = count

    return neighbor_count

def count_neighbour_call():
    # 주어진 그리드
    grid = [[3, 3, 3, 1, 1],
            [3, 3, 2, 1, 1],
            [3, 2, 2, 1, 1]]

    # 같은 색의 이웃 개수를 세어 출력합니다.
    neighbor_counts = count_identical_neighbors(grid)
    return neighbor_counts

def count_different_color_neighbors(grid):
    # 그리드의 행과 열의 크기를 가져옵니다.
    rows, cols = len(grid), len(grid[0])

    # 상, 하, 좌, 우 방향 정의
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 결과를 저장할 리스트를 초기화합니다.
    different_neighbor_count = [[0 for _ in range(cols)] for _ in range(rows)]

    # 그리드의 각 셀을 순회합니다.
    for row in range(rows):
        for col in range(cols):
            # 현재 셀의 색깔을 가져옵니다.
            current_color = grid[row][col]
            # 다른 색의 이웃을 세기 위한 카운터
            count = 0
            for d in directions:
                # 이웃 셀의 위치를 계산합니다.
                nr, nc = row + d[0], col + d[1]
                # 이웃 셀이 그리드 범위 내에 있고, 현재 셀과 색깔이 다른지 확인합니다.
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != current_color:
                    count += 1
            # 계산된 다른 색의 이웃의 수를 결과 리스트에 저장합니다.
            different_neighbor_count[row][col] = count

    return different_neighbor_count



def select_random_with_wight(matrix) :
    # 2차원 배열 예시
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    # 2차원 배열을 1차원 배열로 변환
    flat_matrix = matrix.flatten()

    # 각 원소의 값에 비례하여 가중치 생성
    weights = flat_matrix / flat_matrix.sum()

    # 가중치를 사용하여 1차원 배열에서 원소의 인덱스를 무작위로 선택
    selected_index = np.random.choice(len(flat_matrix), p=weights)

    # 1차원 인덱스를 2차원 인덱스로 변환
    row_index = selected_index // matrix.shape[1]
    col_index = selected_index % matrix.shape[1]

    selected_element = matrix[row_index, col_index]
    print(f"Selected element: {selected_element} at ({row_index}, {col_index})")

    return row_index, col_index, selected_element
    # 선택된 원소 출력


def mutate_find_and_change_color(grid):
    # 자기 자신과 색깔이 다른 이웃의 개수를 계산합니다.
    different_neighbor_counts = count_different_color_neighbors(grid)
    print(f'sum of diff={sum(sum(row) for row in different_neighbor_counts)}')
    rows, cols = len(grid), len(grid[0])
    # 상, 하, 좌, 우 방향 정의
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_grid = copy.deepcopy(grid)

    row,col,val =select_random_with_wight(different_neighbor_counts)

    different_color_neighbors = []
    current_color = new_grid[row][col]
    for d in directions:
        nr, nc = row + d[0], col + d[1]
        if 0 <= nr < rows and 0 <= nc < cols and new_grid[nr][nc] != current_color:
            different_color_neighbors.append((nr, nc))
    # 이웃 중 하나를 임의로 선택하여 색깔 변경
    if different_color_neighbors:
        nr, nc = random.choice(different_color_neighbors)
        # 선택된 이웃의 색깔을 현재 셀의 색깔로 변경
        new_grid[nr][nc] = current_color
        return new_grid  # 변경 후 그리드 반환

    return new_grid

    # 3, 2, 1 순으로 조건에 맞는 셀 찾기
    # for target_count in [3, 2, 1]:
    #     for row in range(rows):
    #         for col in range(cols):
    #             if different_neighbor_counts[row][col] == target_count:
    #                 current_color = new_grid[row][col]
    #                 # 조건에 맞는 셀의 이웃 중 색깔이 다른 이웃을 찾습니다.
    #                 different_color_neighbors = []
    #                 for d in directions:
    #                     nr, nc = row + d[0], col + d[1]
    #                     if 0 <= nr < rows and 0 <= nc < cols and new_grid[nr][nc] != current_color:
    #                         different_color_neighbors.append((nr, nc))
    #                 # 이웃 중 하나를 임의로 선택하여 색깔 변경
    #                 if different_color_neighbors:
    #                     nr, nc = random.choice(different_color_neighbors)
    #                     # 선택된 이웃의 색깔을 현재 셀의 색깔로 변경
    #                     new_grid[nr][nc] = current_color
    #                     return new_grid  # 변경 후 그리드 반환

    return new_grid  # 조건에 맞는 셀이 없는 경우 원본 그리드 반환
def calculate_simplicity_score2(neighbor_count, max_matrix, k):
    unique_count = len(set(value for row in neighbor_count for value in row))

    weights = {i:100 / (2 ** i) for i in range(unique_count)}
    counts = {i: 0 for i in range(unique_count)}
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    rows, cols = len(neighbor_count), len(neighbor_count[0]) if neighbor_count else 0     # 열의 개수 (첫 번째 행의 길이를 사용)
    scores = scores = np.array(neighbor_count)/np.array(max_matrix)
    simplicity_score = np.mean(scores)

    return simplicity_score
    # Out[1]: {0: 0, 1: 6, 2: 7, 3: 2, 4: 0}
def calculate_simplicity_score(matrix):
    # 각 값의 개수를 저장할 딕셔너리
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0     # 열의 개수 (첫 번째 행의 길이를 사용)

    k = 5
    sequence = [100 / (2 ** i) for i in range(k)]

    # 매트리스의 각 셀 값을 순회하며 개수를 센다
    for row in matrix:
        for cell in row:
            if cell in counts:
                counts[cell] += 1

    # 수정된 가중치 설정
    weights = {0: 100, 1: 50, 2: 25, 3: 12, 4: 0}

    # 총 가치 계산
    total_value = sum(counts[value] * weights[value] for value in counts)
    simplicity_fitness = total_value // (len(matrix) * len(matrix[1]))
    return simplicity_fitness

def random_grid(m, n, k):    # Initialize the grid with k different colors randomly
    grid = np.zeros((m, n), dtype=int)
    for i in range(k):
        cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[cell] != 0:
            cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        grid[cell] = i + 1
    return grid


def create_plan(m, n, k):
    grid = np.zeros((m, n), dtype=int)

    # Initialize the grid with k different colors randomly
    for i in range(k):
        cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[cell] != 0:
            cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        grid[cell] = i + 1

    # Create a list of offsets to look for adjacent cells
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Randomly choose cells to color until all cells are filled
    steps = 0
    while np.any(grid == 0):
        subset_cells_nonzero = [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if
                                has_neighbor_zero(grid, row, col) and grid[row, col] != 0]
        cell = tuple(subset_cells_nonzero[random.randint(0, len(subset_cells_nonzero) - 1)])

        # colored cell 중에서 neibhbour가 0인 with_zero_neibhbour_cell
        random.shuffle(adjacent_offsets)  # 여기서 Shuffle을 하지 말고, possible shuffle을 해야 한다.
        assgin_zero_neighbour_value(grid, cell)

        steps += 1

        # Save Initialization Steps one by one into one large file
        # plot_grid_k_distinct_color(grid, k + 1, steps)

    return grid

def create_plan(m, n, k):
    grid = np.zeros((m, n), dtype=int)

    # Initialize the grid with k different colors randomly
    for i in range(k):
        cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[cell] != 0:
            cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        grid[cell] = i + 1

    # Create a list of offsets to look for adjacent cells
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Randomly choose cells to color until all cells are filled
    steps = 0
    while np.any(grid == 0):
        subset_cells_nonzero = [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if
                                has_neighbor_zero(grid, row, col) and grid[row, col] != 0]
        cell = tuple(subset_cells_nonzero[random.randint(0, len(subset_cells_nonzero) - 1)])

        # colored cell 중에서 neibhbour가 0인 with_zero_neibhbour_cell
        random.shuffle(adjacent_offsets)  # 여기서 Shuffle을 하지 말고, possible shuffle을 해야 한다.
        assgin_zero_neighbour_value(grid, cell)

        steps += 1

        # Save Initialization Steps one by one into one large file
        # plot_grid_k_distinct_color(grid, k + 1, steps)

    return grid

def mutate(grid):
    steps = 0
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while np.any(grid == 0):
        subset_cells_nonzero = [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if
                                has_neighbor_zero(grid, row, col) and grid[row, col] != 0]
        cell = tuple(subset_cells_nonzero[random.randint(0, len(subset_cells_nonzero) - 1)])

        # colored cell 중에서 neibhbour가 0인 with_zero_neibhbour_cell
        random.shuffle(adjacent_offsets)  # 여기서 Shuffle을 하지 말고, possible shuffle을 해야 한다.
        assgin_zero_neighbour_value(grid, cell)

        steps += 1

def mutate2(grids, m,n,k, max_matrix):
    next_grids=[]
    simplicity_scores = []
    for parent in grids:
        # num_diff_neighbors_parent = count_different_color_neighbors(parent)
        neighbor_counts_parents  = count_neighbors(parent)
        parent_fit = calculate_simplicity_score2(neighbor_counts_parents, max_matrix, k)
        # print(f'parent={parent}, fit={parent_fit}')


        mutated = mutate_find_and_change_color(parent)
        if (count_unique_values(mutated) < k ) :
            mutated_fit = 0
        else:
            neighbor_counts_mutated=count_neighbors(mutated)
            mutated_fit = calculate_simplicity_score2(neighbor_counts_mutated, max_matrix, k)
        # print(f'mutated={mutated}, fit = {mutated_fit}')
        next_grid = parent if (parent_fit > mutated_fit ) else mutated
        next_fit = parent_fit if  (parent_fit > mutated_fit ) else mutated_fit
        next_grids.append(next_grid)
        simplicity_scores.append(next_fit)
    return next_grids, simplicity_scores



def module_main():
    m, n, k, num_pops, cols = 3, 5, 4, 10, 5
    n_generations = 10
    rows = ceil(num_pops / cols)
    max_matrix = calculate_max_neighbors_matrix(m,n)

    grids = []
    next_grids = []
    initial_fits = []


    # create population
    for pop in range(num_pops):
        grid = create_plan(m, n, k)
        count_grid = count_neighbors(grid)
        initial_fits.append(calculate_simplicity_score2(count_grid,max_matrix, k))
        print(grid)
        grids.append(grid)

    plotfilename = generate_filename_with_sequence('init', 'png', 1)
    plot_grids(grids, m, n, k, rows, cols, initial_fits, plotfilename)


    for gen in range(n_generations):
        grids, simplicity_scores = mutate2(grids, m, n, k, max_matrix)

        plotfilename = generate_filename_with_sequence('gen', 'png', gen)
        plot_grids(grids, m, n, k, rows, cols, simplicity_scores, plotfilename)

    plot_merged_imaage_in_folder()
if __name__ == "__main__":
    module_main()


