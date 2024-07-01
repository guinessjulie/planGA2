
import numpy as np
import random

def create_plan(m, n, k):
    grid = np.zeros((m, n), dtype=int)
    boundary_cells = set()

    # Initialize the grid with k different colors randomly
    for i in range(k):
        cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[cell] != 0:
            cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        grid[cell] = i + 1
        boundary_cells.update(get_boundary_cells(grid, cell))

    # Create a list of offsets to look for adjacent cells
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Randomly choose cells to color until all cells are filled
    steps = 0
    while np.any(grid == 0):
        cell = random.choice(list(boundary_cells))
        if assign_zero_neighbour_value(grid, cell):
            boundary_cells.update(get_boundary_cells(grid, cell))
        boundary_cells.discard(cell)
        steps += 1
        print(grid)
    return grid

def has_neighbor_zero(grid, row, col):
    nrows, ncols = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
            return True
    return False

def assign_zero_neighbour_value(grid, cell):
    nrows, ncols = grid.shape
    row, col = cell
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid_offsets = [
        offset for offset in adjacent_offsets
        if 0 <= row + offset[0] < nrows and 0 <= col + offset[1] < ncols and grid[row + offset[0], col + offset[1]] == 0
    ]
    if not valid_offsets:
        return False
    selected_offset = random.choice(valid_offsets)
    selected_neighbour_loc = (row + selected_offset[0], col + selected_offset[1])
    grid[selected_neighbour_loc] = grid[row, col]
    return True

def get_boundary_cells(grid, cell):
    """Returns a set of cells that are on the boundary (next to an uncolored cell)."""
    nrows, ncols = grid.shape
    boundary_cells = set()
    row, col = cell
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
            boundary_cells.add((new_row, new_col))
    return boundary_cells

def module_main():
    m, n, k = 3,5,4
    grid = create_plan(m, n, k)
    print(grid)

if __name__ == "__main__":
    module_main()