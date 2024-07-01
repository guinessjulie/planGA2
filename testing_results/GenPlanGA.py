
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import numpy as np
import random
from scipy.ndimage import label

def get_neighbors(cell, grid_shape):
    i, j = cell
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    return [neighbor for neighbor in neighbors if 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]]

def fill_region(start_cell, grid, color, max_cells):
    """
    Recursively fill the region starting from 'start_cell' with 'color' up to 'max_cells' cells.
    """
    if max_cells <= 0:
        return 0  # Stop filling if the region size limit is reached

    m, n = grid.shape
    if grid[start_cell] != 0:
        return 0  # Already colored cell, do not overwrite

    # Color the start cell
    grid[start_cell] = color
    colored_cells = 1

    # Randomly order the neighbors to create more organic shapes
    neighbors = get_neighbors(start_cell, (m, n))
    random.shuffle(neighbors)

    for neighbor in neighbors:
        colored_cells += fill_region(neighbor, grid, color, max_cells - colored_cells)
        if colored_cells >= max_cells:
            break  # Stop if we have filled enough cells

    return colored_cells

def create_new_plan_v2(m, n, k):
    grid = np.zeros((m, n), dtype=int)
    # After the grid is filled, identify each unique group
    labeled_grid, num_features = label(grid > 0)
    max_cells_per_color = (m * n) // k  # Maximum cells of the same color

    for color in range(1, k + 1):
        start_cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[start_cell] != 0:
            start_cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        fill_region(start_cell, grid, color, max_cells_per_color)

    return grid



def expand_color(grid, cell, color):
    print(f'expanding_color of {grid}')
    m, n = grid.shape
    queue = [cell]
    while queue:
        current_cell = queue.pop(0)
        for neighbor in get_neighbors(current_cell, (m, n)):
            if grid[neighbor] == 0:
                grid[neighbor] = color
                queue.append(neighbor)

# def get_neighbors(cell, grid_shape):
#     i, j = cell
#     neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
#     return [neighbor for neighbor in neighbors if 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]]

def create_new_plan_v1(m, n, k):
    grid = np.zeros((m, n), dtype=int)
    for i in range(k):
        cell = (random.randint(0, m-1), random.randint(0, n-1))
        while grid[cell] != 0:
            cell = (random.randint(0, m-1), random.randint(0, n-1))
        grid[cell] = i + 1
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while np.any(grid == 0):
        colored_cells = np.argwhere(grid != 0)
        for cell in colored_cells:
            random.shuffle(adjacent_offsets)
            for offset in adjacent_offsets:
                adjacent_cell = tuple(np.add(cell, offset))
                if 0 <= adjacent_cell[0] < m and 0 <= adjacent_cell[1] < n and grid[adjacent_cell] == 0:
                    grid[adjacent_cell] = grid[tuple(cell)]
                    break
    return grid

def create_new_plan_v3(m, n, k):
    grid = np.zeros((m, n), dtype=int)

    for i in range(k):
        cell = (random.randint(0, m-1), random.randint(0, n-1))
        while grid[cell] != 0:
            cell = (random.randint(0, m-1), random.randint(0, n-1))
        grid[cell] = i + 1

    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while np.any(grid == 0):
        colored_cells = np.argwhere(grid != 0)
        for cell in colored_cells:
            random.shuffle(adjacent_offsets)
            for offset in adjacent_offsets:
                adjacent_cell = tuple(np.add(cell, offset))
                if 0 <= adjacent_cell[0] < m and 0 <= adjacent_cell[1] < n and grid[adjacent_cell] == 0:
                    grid[adjacent_cell] = grid[tuple(cell)]
                    break
    # print(f'[111] grid: \n{grid}')
    # After the grid is filled, identify each unique group
    s = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labeled_grid, num_features = label(grid, structure=s)
    # print(f'[111] labeled_grid: \n{labeled_grid}, num_features, {num_features}')

    # Count the size of each group
    group_sizes = np.bincount(labeled_grid.ravel())[1:]  # Skip the count for label 0


    # Find the smallest groups and their labels
    smallest_groups = np.argsort(group_sizes)[:num_features - k]
    smallest_labels = smallest_groups + 1  # Labels are 1-indexed
    print(f'smallest_groups = ={smallest_groups}, smallest_labels ={smallest_labels }')
    # Remove smallest groups
    for label_to_remove in smallest_labels:
        grid[labeled_grid == label_to_remove] = 0

    # Optionally, reassign color groups to have sequential numbers from 1 to k
    new_color = 1
    for remaining_label in range(1, num_features + 1):
        if remaining_label in smallest_labels:
            continue
        grid[labeled_grid == remaining_label] = new_color
        new_color += 1
    return grid

def populate(num_grids, m, n, k):
    # grids= [create_new_plan_v3(m, n, k) for _ in range(num_grids)]
    grids= [create_new_plan_v6(num_grids, m, n, k) for _ in range(num_grids)]
#   print(f'[101] grids: {grids}')
    return grids

def get_neighbors(cell, grid_shape):
    """
    Get the neighbors of a given cell within the grid boundaries.
    """
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

def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    return random.choices(population, weights=selection_probs, k=2)


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


def constrained_crossover_backup(parent1, parent2):
    crossover_point = np.random.randint(1, parent1.shape[1])
    child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
    child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))
    return child1, child2

def plot_grid_single(grids, m, n, k, per_row):
    num_rows = (len(grids) + per_row - 1) // per_row
    for i, grid in enumerate(grids):
        plt.figure(figsize=(n, m))
        plt.matshow(grid, cmap=cm.get_cmap('hsv', k+1), fignum=1)
        plt.axis('off')
        plt.savefig(f'grid_{i+1}.png')
        plt.close()


def plot_grids(plotname, grids, m, n, k, per_row):
    num_grids = len(grids)
    num_rows = (num_grids + per_row - 1) // per_row
    fig, axes = plt.subplots(num_rows, per_row, figsize=(n * per_row / 2, m * num_rows / 2))

    # If there's only one row or column, we don't have a 2D array of axes, so we fix that for consistency
    if num_rows == 1:
        axes = [axes]
    if per_row == 1:
        axes = [[ax] for ax in axes]

    # Flatten the axes array for easy iteration
    axes_flat = [ax for sublist in axes for ax in sublist]

    # Hide the unused subplots if the total number of grids is not a multiple of per_row
    for i in range(num_grids, num_rows * per_row):
        axes_flat[i].axis('off')

    for i, grid in enumerate(grids):
        ax = axes_flat[i]
        ax.matshow(grid, cmap=plt.cm.get_cmap('hsv', k+1))
        ax.axis('off')

    plt.tight_layout()

    plt.savefig(plotname)
    plt.close()


def plot_grids_sample2(plotname, grids, m, n, k, per_row, percentage=100):
    num_grids = len(grids)  # Total number of grids
    num_grids_to_plot = max(int(num_grids * (percentage / 100)), 1)  # Calculate the number of grids to plot, at least 1

    # Select a portion of grids based on the calculated number
    grids_to_plot = grids[:num_grids_to_plot]

    num_rows = (num_grids_to_plot + per_row - 1) // per_row
    fig, axes = plt.subplots(num_rows, per_row, figsize=(n * per_row / 2, m * num_rows / 2))

    # Adjust for a single row or single column of plots
    if num_rows == 1:
        axes = [axes]
    if per_row == 1:
        axes = [[ax] for ax in axes]

    # Flatten the axes array for easy iteration
    axes_flat = [ax for sublist in axes for ax in sublist]

    # Hide the unused subplots
    for i in range(num_grids_to_plot, num_rows * per_row):
        axes_flat[i].axis('off')

    for i, grid in enumerate(grids_to_plot):
        ax = axes_flat[i]
        ax.matshow(grid, cmap=plt.cm.get_cmap('hsv', k+1))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(plotname)
    plt.close()

def plot_grids_sample(plotname, grids, m, n, k, per_row, percentage=10, num_plot=10):
    # Calculate the number of grids to plot based on the percentage
    #num_grids_to_plot = max(int(len(grids) * (percentage / 100)), 1)  # Ensure at least one grid is plotted
    #num_grids_to_plot = min(max(int(len(grids) * (percentage / 100)), 1) , num_plot) # Ensure at least one grid is plotted
    num_grids_to_plot = num_plot
    selected_grids = grids[:num_grids_to_plot]  # Select the first X% of the grids

    num_grids = len(selected_grids)
    num_rows = (num_grids + per_row - 1) // per_row
    fig, axes = plt.subplots(num_rows, per_row, figsize=(n * per_row / 2, m * num_rows / 2))

    # Adjust for the case of a single row or column
    if num_rows == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)  # Ensure axes is 2D for consistency

    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # Hide the unused subplots
    for i in range(num_grids, num_rows * per_row):
        axes_flat[i].axis('off')

    # Plot the selected grids
    for i, grid in enumerate(selected_grids):
        ax = axes_flat[i]
        ax.matshow(grid, cmap=plt.cm.get_cmap('hsv', k))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(plotname)
    plt.close()

def generate_filename_with_timestamp(prefix: str, extension: str) ->str:
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{prefix}_{time_str}.{extension}"
    return filename


def evolve_save(num_generations, population_size, m, n, k):
    grids = populate(population_size, m, n, k)

    # Example usage:
    plot_name = generate_filename_with_timestamp('init', 'png')

    plot_grids_sample(plot_name, grids, m, n, k, 5)
    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(grid) for grid in grids]
        new_grids = []
        while len(new_grids) < population_size:
            parents = roulette_wheel_selection(grids, fitness_scores)
            for parent1, parent2 in zip(*[iter(parents)]*2):
                child1, child2 = constrained_crossover(parent1, parent2)
                new_grids.extend([child1, child2])
                # Assuming child1 and child2 are the result of a crossover
                # mutation_rate = 0.01  # 1% chance of mutation per cell
                # child1 = mutate(child1, mutation_rate)
                # child2 = mutate(child2, mutation_rate)
        grids = new_grids[:population_size]


        print(f"Generation {generation + 1} completed")
    plot_name = generate_filename_with_timestamp('final', 'png')
    plot_grids_sample(plot_name, grids, m, n, k, 5)

def mutate_older(grid, mutation_rate=0.01):
    """
    Mutates the grid by randomly altering the values of some cells,
    ensuring that cells still form groups with their neighbors.

    Parameters:
    - grid: A 2D numpy array representing the grid.
    - mutation_rate: The probability of any given cell mutating.

    Returns:
    - A mutated grid with the same constraints.
    """
    m, n = grid.shape
    for i in range(m):
        for j in range(n):
            if random.random() < mutation_rate:
                neighbors = get_neighbors(i, j, m, n)
                if neighbors:
                    # Choose a random neighbor to adopt its value
                    neighbor = random.choice(neighbors)
                    grid[i, j] = grid[neighbor[0], neighbor[1]]
    return grid


def mutate_norworking(grid, mutation_rate):
    """
    Mutate the grid by randomly altering cell values while considering neighboring constraints.

    Parameters:
    - grid (np.array): The grid to be mutated.
    - mutation_rate (float): The probability of each cell being mutated.
    """
    m, n = grid.shape
    for i in range(m):
        for j in range(n):
            if random.random() < mutation_rate:
                neighbors = get_neighbors((i, j), grid.shape)
                if neighbors:
                    # Option 1: Swap with a neighbor
                    # neighbor = random.choice(neighbors)
                    # grid[i, j], grid[neighbor] = grid[neighbor], grid[i, j]

                    # Option 2: Change to a neighbor's value to maintain grouping
                    neighbor_value = grid[random.choice(neighbors)]
                    grid[i, j] = neighbor_value
    return grid


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


# Use this function to mutate a single grid
def mutate_individual(grid, mutation_rate):
    # Randomly choose a cell to mutate
    m, n = grid.shape
    cell_to_mutate = (random.randint(0, m - 1), random.randint(0, n - 1))

    # Get the neighbors of the chosen cell
    neighbors = get_neighbors(cell_to_mutate, grid.shape)

    # If there are neighbors, choose a neighbor's value to mutate to
    if neighbors:
        new_value = grid[random.choice(neighbors)]
        grid[cell_to_mutate] = new_value

    return grid




def evolve(num_generations, population_size, m, n, k):

    grids = populate(population_size, m, n, k)

    # Example usage:
    plot_name = generate_filename_with_timestamp('init', 'png')

    plot_grids_sample(plot_name, grids, m, n, k, 5)
    mutation_rate = 0.01  # For example, 1% mutation rate
    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(grid) for grid in grids]
        new_grids = []
        while len(new_grids) < population_size:
            parents = roulette_wheel_selection(grids, fitness_scores)
            for parent1, parent2 in zip(*[iter(parents)]*2):
                child1, child2 = constrained_crossover(parent1, parent2)

#                child1 = mutate(child1, mutation_rate)
#                child2 = mutate(child2, mutation_rate)

                new_grids.extend([child1, child2])
                # Assuming child1 and child2 are the result of a crossover
                # Assuming you have a list of grids and a mutation rate

        grids = new_grids[:population_size]
        # for grid in grids:
        #     mutate_individual(grid, mutation_rate)

        genname='gen'+str(generation)
        plot_name = generate_filename_with_timestamp(genname, 'png')
        plot_grids_sample(plot_name, grids, m, n, k, 5)

        print(f"Generation {generation + 1} completed")
    plot_name = generate_filename_with_timestamp('final', 'png')
    plot_grids_sample(plot_name, grids, m, n, k, 5)

def create_mating_pool(grids, fitness):
    mating_pool=[]
    # for i in range(0, len(grids)):
    while len(mating_pool) < len(grids):
        parents = roulette_wheel_selection(grids, list(fitness))
        mating_pool.extend(parents) # #TODO compare with roulet_wheel module below somewhere
    return mating_pool


def evolve_v6(num_generations, population_size, m, n, k):
    # 초기 해집단 생성
    per_row = 5  # Number of grids per row
    grids = populate(population_size, m, n, k)
    plot_name = generate_filename_with_timestamp('init', 'png')
    plot_grids_sample(plot_name, grids, m, n, k, 5)

    for generation in range(num_generations):
        # 적합도 계산
        fitness_scores = [calculate_fitness(individual) for individual in grids]
        # 새로운 세대의 해집단을 저장할 리스트
        new_grids = []

        while len(new_grids) <= population_size:
            # 교배 풀 생성 및 부모 선택 (이 부분은 여러분의 함수에 따라 다름)
            mating_pool = create_mating_pool(grids, fitness_scores)
            parent1, parent2 = random.sample(mating_pool, 2)

            # 크로스오버를 통해 자식 생성
            child1, child2 = constrained_crossover(parent1, parent2)

            # 새로운 세대에 자식 추가
            new_grids.extend([child1, child2])
        # 해집단 업데이트
        grids = new_grids[:population_size]  # 해집단 크기를 유지
        genname='gen'+str(generation)
        plot_name = generate_filename_with_timestamp(genname, 'png')
        plot_grids_sample(plot_name, grids, m, n, k, 5)
        plot_grids_v6(plot_name, grids, m, n, k,)
        # 세대별 진행 상황 출력 (옵션)
        print(f"Generation {generation + 1} completed")
    # 최종 해집단 반환
    plot_name = generate_filename_with_timestamp('final', 'png')
    plot_grids_sample(plot_name, grids, m, n, k, 5)

    return grids


def create_new_plan_v5(num_grids, m, n, k):
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
    while np.any(grid == 0):
        colored_cells = np.argwhere(grid != 0)
        cell = tuple(colored_cells[random.randint(0, len(colored_cells) - 1)])
        random.shuffle(adjacent_offsets)
        for offset in adjacent_offsets:
            adjacent_cell = (cell[0] + offset[0], cell[1] + offset[1])
            if (0 <= adjacent_cell[0] < m) and (0 <= adjacent_cell[1] < n) and grid[adjacent_cell] == 0:
                grid[adjacent_cell] = grid[cell]
                break

    return grid


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
        colored_cells = np.argwhere(grid != 0)  # indices of non-zero element of that condition
        subset_cells_nonzero = [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if
                                has_neighbor_zero(grid, row, col) and grid[row, col] != 0]
        cell = tuple(subset_cells_nonzero[random.randint(0, len(subset_cells_nonzero) - 1)])
        # colored cell 중에서 neibhbour가 0인 with_zero_neibhbour_cell
        random.shuffle(adjacent_offsets)  # 여기서 Shuffle을 하지 말고, possible shuffle을 해야 한다.
        assgin_zero_neighbour_value(grid, cell)
        steps += 1
        plot_grid_k_distinct_color(grid, k + 1, steps)

    return grid


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime


def plot_grid_k_distinct_color(grid, k, steps):
    if k <= 10:
        cmap = plt.cm.get_cmap('Accent', k)
    elif k <= 20:
        cmap = plt.cm.get_cmap('tab20', k)
    else:
        cmap = plt.cm.get_cmap('Set3', k)

    norm = mcolors.Normalize(vmin=0, vmax=k - 1)

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    ax.axis('off')
    plt.savefig(generate_filename_with_timestamp('step', 'png', steps))
    plt.close()

def plot_grids_v6(plotname, grids, nrows, ncols, k):
    print(f'k={k}')

    cmap = plt.cm.get_cmap('tab10', k)
    norm = mcolors.Normalize(vmin=0, vmax=k - 1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))
    # Flatten the axis array if necessary
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        if idx < len(grids):
            # Display image
            ax.imshow(grids[idx], cmap=cmap, norm=norm)
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Ensure extra subplots are invisible

    plt.tight_layout()
    plt.savefig(plotname)
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


# 함수들 (calculate_fitness, create_mating_pool, select_parents_idx, custom_crossover)은 여러분이 구현해야 합니다.


def main_gen(num_gen):
    num_generations = num_gen
    population_size = 10
    m, n, k = 6, 8, 3
    # evolve(num_generations, population_size, m, n, k)
    evolve_v6(num_generations, population_size, m, n, k)



# Example usage
# Assuming `grids` is a list of NumPy arrays representing your grids
# m, n, k = 10, 15, 6  # Grid dimensions and number of color groups
# per_row = 5  # Number of grids per row in the plot
# plot_grids(grids, m, n, k, per_row)

if __name__ == "__main__":
    # main()
    main_gen(10)
