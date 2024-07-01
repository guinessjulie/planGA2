

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import numpy as np
import random
from math import ceil
from scipy.ndimage import label


def create_new_plan_v6(num_grids, m, n, k):
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


def plot_grid_k_distinct_color(grid, k, steps):
    if k <= 10:
        cmap = plt.cm.get_cmap('Accent', k)
    elif k <= 20:
        cmap = plt.cm.get_cmap('tab20', k)
    else:
        cmap = plt.cm.get_cmap('Set3', k)

    norm = mcolors.Normalize(vmin=0, vmax=k - 1)

    fig, ax = plt.subplots()

    # Use imshow instead of manually plotting rectangles
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    ax.axis('off')
    # Now, adding colorbar should work without issues
    # plt.colorbar(im, ax=ax, ticks=range(k), spacing='proportional', shrink=0.92, aspect=40, orientation='horizontal')

    plt.savefig(generate_filename_with_timestamp('step', 'png', steps))
    # plt.show()
    plt.close()


def generate_filename_with_timestamp(prefix: str, extension: str, steps=None) -> str:
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    if steps is not None:
        time_str = steps
    filename = f"{prefix}_{time_str}.{extension}"
    return filename


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


def has_neighbor_zero(grid, row, col):
    nrows, ncols = len(grid), len(grid[0])
    # Define relative positions of neighbors (up, down, left, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in neighbors:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
            return True
    return False


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
        rows = ceil(len(grids) / cols)
    if cols is None:
        cols = ceil(len(grids)/ rows)
    fig, axes = plt.subplots(rows, cols, figsize=(n * cols, m * rows))
    xy_coords = [(x, y) for x in range(rows) for y in range(cols)]
    for ij, grid in enumerate(grids):
        colors = mpl.colormaps['Accent']
        ax = axes[xy_coords[ij]]
        ax.matshow(grid, cmap=colors)
        if fitness_scores is not None:
            ax.set_title(f'{fitness_scores[ij]:.2f}', fontsize=40)
        ax.axis('off')

    if savefilename is not None:
        plt.savefig(savefilename)
    plt.show()

def plot_grids_sample(grids, m, n, k, rows=None, cols=5, fitness_scores=None, savefilename=None, nsample=20):
    """
    Creates a plot with multiple subplots arranged in specified rows and columns, each displaying a grid.

    Parameters:
    - m (int): The number of rows in each grid.
    - n (int): The number of columns in each grid.
    - k (int): The number of color groups in each grid.
    - rows (int): The number of subplot rows.
    - cols (int): The number of subplot columns.
    """
    grids = random.sample(grids, nsample)
    if rows is None:
        rows = ceil(len(grids) / cols)
    if cols is None:
        cols = ceil(len(grids)/ rows)
    fig, axes = plt.subplots(rows, cols, figsize=(n * cols, m * rows))
    xy_coords = [(x, y) for x in range(rows) for y in range(cols)]
    for ij, grid in enumerate(grids):
        colors = mpl.colormaps['Accent']
        ax = axes[xy_coords[ij]]
        ax.matshow(grid, cmap=colors)
        if fitness_scores is not None:
            ax.set_title(f'{fitness_scores[ij]:.2f}', fontsize=40)
        ax.axis('off')

    if savefilename is not None:
        plt.savefig(savefilename)
    plt.show()


def get_neighbors(cell, grid_shape):
    i, j = cell
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    return [neighbor for neighbor in neighbors if 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]]


def calculate_fitness(grid):
    """
    Calculate the fitness of a given grid based on the ratio of boundary length to the number of internal cells.
    Lower ratio means the shape is closer to a rectangle.
    """
    # Initialize fitness to 0
    fitness = 0
    # Calculate fitness for each color group
    for color in np.unique(grid):
        if color == 0:  # Skip the background
            continue
        # Find the cells of the current color group
        cells = np.argwhere(grid == color)
        # Calculate the boundary length and the number of internal cells
        boundary_length = 0
        internal_cells = 0
        for cell in cells:
            neighbors = get_neighbors(cell, grid.shape)
#            print(f'[56] neighbors={len(neighbors)}..........in calculate_fitness()')
            colored_neighbors = sum(grid[neighbor] == color for neighbor in neighbors)
            if colored_neighbors < 4:  # Less than 4 colored neighbors means it's on the boundary
                boundary_length += 1
            else:
                internal_cells += 1
        # Update fitness: Higher number of internal cells per boundary length is better
        if boundary_length > 0:
            fitness += internal_cells / boundary_length
    return fitness


import math


def calculate_simplicity(grid):
    # Find unique colors
    unique_colors = np.unique(grid)

    simplicity_scores = {}

    for color in unique_colors:
        # Find cells of the current color
        color_cells = np.argwhere(grid == color)

        # Calculate area as the number of cells
        area = len(color_cells)

        # Calculate perimeter
        perimeter = 0
        for cell in color_cells:
            x, y = cell
            # Check all four neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for nx, ny in neighbors:
                if nx < 0 or nx >= grid.shape[0] or ny < 0 or ny >= grid.shape[1] or grid[nx, ny] != color:
                    perimeter += 1

        # Calculate simplicity as area/perimeter

        simplicity = area / perimeter if perimeter > 0 else 0
        simplicity = (16 * area) / math.pow(perimeter, 2) if perimeter > 0 else 0
        simplicity_scores[color] = simplicity
    return sum(simplicity_scores.values())/len(simplicity_scores)
    # return simplicity_scores


    # simplicity_scores = calculate_simplicity(grid)
    # for color, simplicity in simplicity_scores.items():
    #     print(f"Color {color}: Simplicity = {simplicity:.2f}")
    #


def mutate(grid, mutation_rate):
    """
    Mutates the grid by randomly changing the group of a cell to that of its neighbors.

    Parameters:
    - grid: 2D numpy array representing the grid.
    - mutation_rate: Probability of a cell being mutated.

    Returns:
    - grid: The mutated grid.
    """
    m, n = grid.shape
    for i in range(m):
        for j in range(n):
            if random.random() < mutation_rate:
                neighbors = get_neighbors((i, j), grid.shape)
                if neighbors:
                    # Choose the group of one of the neighbors
                    neighbor_group = grid[random.choice(neighbors)]
                    grid[i, j] = neighbor_group
    return grid



def create_mating_pool(grids, fitness):
    mating_pool=[]
    # for i in range(0, len(grids)):
    while len(mating_pool) < len(grids):
        parent1, parent2 = roulette_wheel_selection(grids, list(fitness))
        parents = mutate(parent1, mutation_rate=0.1), mutate(parent2, mutation_rate=0.1)
        print(parents)
        mating_pool.extend(parents) # #TODO compare with roulet_wheel module below somewhere
    return mating_pool


def select_betters(grids, fitness):
    new_grids=[]
    # for i in range(0, len(grids)):
    while len(new_grids) < len(grids)*3:
        parents = roulette_wheel_selection(grids, list(fitness))
        new_grids.extend(parents) # #TODO compare with roulet_wheel module below somewhere
    return new_grids


def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    # return random.choices(population, weights=selection_probs, k=2)#temp
    return random.choices(population, weights=fitnesses, k=2)


def get_group_boundaries(grid):
    boundaries = []
    for i in range(1, grid.shape[1]):
        if not np.all(grid[:, i] == grid[:, i - 1]):
            boundaries.append(i)
    return boundaries



def constrained_crossover(parent1, parent2):
    # Find the group boundaries in both parents.
    boundaries1 = get_group_boundaries(parent1)
    boundaries2 = get_group_boundaries(parent2)

    # Find common boundaries to use as potential crossover points.
    common_boundaries = list(set(boundaries1) & set(boundaries2))

    if not common_boundaries:
        # If there are no common boundaries, revert to a basic crossover.
        crossover_point = np.random.randint(1, parent1.shape[1])
    else:
        # If there are common boundaries, choose one at random.
        crossover_point = random.choice(common_boundaries)

    # Perform the crossover at the chosen point.
    child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
    child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))

    return child1, child2



def evolve_v6(num_generations, population_size, m, n, k):
    # 초기 해집단 생성
    per_row = 5  # Number of grids per row

    grids = [create_new_plan_v6(population_size, m, n, k) for _ in range(population_size)]
    rows, cols = 2, 5
    fit_gen = []
    #
    # plot_name = generate_filename_with_timestamp('init', 'png')
    # plot_grids_sample(plot_name, grids, m, n, k, 5)
    fitness_scores = [calculate_simplicity(individual) for individual in grids]
    fit_gen.append(np.average(fitness_scores))
    print(np.average(fitness_scores))
    plot_grids_sample(grids, m, n, k,rows=rows, cols=cols, fitness_scores=fitness_scores,savefilename='init', nsample=10)
    #
    for generation in range(num_generations):
        # 적합도 계산
        #fitness_scores = [calculate_fitness(individual) for individual in grids]
        fitness_scores = [calculate_simplicity(individual) for individual in grids]
        fit_gen.append(np.average(fitness_scores))
        cols = min(5, n)
        #plot_grids(grids, m, n, k,rows=None, cols=cols, fitness_scores=fitness_scores)
        genname = 'gen_' + str(generation)+'.png'

        # grids = select_betters(grids,fitness_scores)[:len(grids)]
        # plot_grids_sample(grids, m, n, k,rows=None, cols=cols, fitness_scores=fitness_scores,savefilename=genname, nsample=10)
        # 새로운 세대의 해집단을 저장할 리스트
        new_grids = []
        mating_pool = create_mating_pool(grids, fitness_scores)
        new_grids = random.sample(mating_pool, population_size)
        while len(new_grids) <= population_size:
            # 교배 풀 생성 및 부모 선택 (이 부분은 여러분의 함수에 따라 다름)
            # mating_pool = create_mating_pool(grids, fitness_scores)
            mating_pool = create_mating_pool(grids, fitness_scores)
            # parent1, parent2 = random.sample(mating_pool, 2)

            # 크로스오버를 통해 자식 생성
            # child1, child2 = constrained_crossover(parent1, parent2)
            # 새로운 세대에 자식 추가
            # new_grids.extend([child1, child2])
        #해집단 업데이트
        grids = new_grids[:population_size]  # 해집단 크기를 유지
        plot_grids_sample(grids, m, n, k,rows=None, cols=cols, fitness_scores=fitness_scores,savefilename=genname, nsample=10)
        # genname='gen'+str(generation)
        # plot_name = generate_filename_with_timestamp(genname, 'png')
        #plot_grids(grids, m, n, k, rows=None, cols=5, fitness_scores=None, savefilename=plot_name)
    #     plot_grids_v6(plot_name, grids, m, n, k,)
    #     # 세대별 진행 상황 출력 (옵션)
    #     print(f"Generation {generation + 1} completed")
    # # 최종 해집단 반환
    # plot_name = generate_filename_with_timestamp('final', 'png')
    # plot_grids_sample(plot_name, grids, m, n, k, 5)
    #
    plt.figure()
    plt.plot(fit_gen)
    plt.savefig('fitness changes as generation')
    plt.show()
    # return grids

def main():
    num_gen = 1
    num_grids =10
    m = 3
    n = 5
    k = 3

    grids =evolve_v6(num_gen,num_grids, m, n, k)

    # plot_step(grid, m, n, k)


if __name__ == "__main__":
    # main()
    main()

