import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import os


def create_new_plan(m, n, k):
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

def populate(num_grids, m, n, k):
    return [create_new_plan(m, n, k) for _ in range(num_grids)]

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
            print(f'[56] neighbors={len(neighbors)}..........in calculate_fitness()')
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

def constrained_crossover(parent1, parent2):
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


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

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

def plot_grids_sample(plotname, grids, m, n, k, per_row, percentage=10):
    # Calculate the number of grids to plot based on the percentage
    num_grids_to_plot = max(int(len(grids) * (percentage / 100)), 1)  # Ensure at least one grid is plotted
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
        ax.matshow(grid, cmap=plt.cm.get_cmap('hsv', k + 1))
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(plotname)
    plt.close()


# Example usage
# Assume 'grids' is your list of grid arrays
plotname = 'selected_grids.png'
m, n, k = 10, 15, 6  # Dimensions and number of color groups
per_row = 5  # Number of grids per row in the plot
percentage = 10  # Percentage of grids to plot


# plot_grids(plotname, grids, m, n, k, per_row, percentage)

# Adjust the parameters as needed when calling plot_grids
# plot_grids(grids, m, n, k, per_row)

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
                print(f'[274] neighbors={neighbors.shape}..........in mutate()')
                if neighbors:
                    # Choose a random neighbor to adopt its value
                    neighbor = random.choice(neighbors)
                    grid[i, j] = grid[neighbor[0], neighbor[1]]
    return grid

import numpy as np
import random

def mutate(grid, mutation_rate):
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

def evolve(num_generations, population_size, m, n, k):
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
                mutation_rate = 0.1  # 1% chance of mutation per cell
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
        grids = new_grids[:population_size]


        print(f"Generation {generation + 1} completed")
    plot_name = generate_filename_with_timestamp('final', 'png')
    plot_grids_sample(plot_name, grids, m, n, k, 5)


def main():
    num_generations = 10
    population_size = 18
    m, n, k = 4, 6, 6
    evolve(num_generations, population_size, m, n, k)

def main_gen(num_gen):
    num_generations = num_gen
    population_size = 100
    m, n, k = 4, 6, 6
    evolve(num_generations, population_size, m, n, k)



# Example usage
# Assuming `grids` is a list of NumPy arrays representing your grids
# m, n, k = 10, 15, 6  # Grid dimensions and number of color groups
# per_row = 5  # Number of grids per row in the plot
# plot_grids(grids, m, n, k, per_row)

if __name__ == "__main__":
    # main()
    main_gen(3)
