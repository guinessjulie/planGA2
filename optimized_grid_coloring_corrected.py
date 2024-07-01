
import numpy as np
import random

import numpy as np
import random


def create_floorplan(m, n, k, floorplan):
    grid = np.full((m, n), -1, dtype=int)  # Initialize all cells as -1

    # Set cells within the floorplan to 0 (uncolored)
    for i in range(m):
        for j in range(n):
            if floorplan[i][j] == 1:
                grid[i, j] = 0
    print(f'grid:{grid}')
    # Randomly place k colors within the floorplan
    colors_placed = 0
    while colors_placed < k:
        row, col = random.randint(0, m - 1), random.randint(0, n - 1)
        if grid[row, col] == 0:  # Ensure the cell is within the floorplan and uncolored
            grid[row, col] = colors_placed + 1
            colors_placed += 1
    print(f'colored grid:{grid}')

    # Initialize boundary cells based on the updated grid
    boundary_cells = get_boundary_cells(grid, floorplan)
    print(f'boundary_cells： {boundary_cells}')
    # Fill the grid
    while boundary_cells:
        new_boundary_cells = set()
        for cell in boundary_cells:
            if assign_zero_neighbour_value(grid, cell):
                new_boundary_cells.update(get_boundary_cells(grid, floorplan))
        boundary_cells = new_boundary_cells
        print(f'boundary_cells: {boundary_cells}')
    return grid


def get_boundary_cells_old2(grid, floorplan):
    boundary_cells = set()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] > 0 and has_neighbor_zero(grid, row, col):
                boundary_cells.add((row, col))
                print(f'get_boundary_cells: added ({row},{col})')
    return boundary_cells

def get_boundary_cells(grid, floorplan):
    boundary_cells = set()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if floorplan[row][col] == 1 and grid[row, col] > 0 and has_neighbor_zero(grid, row, col):
                boundary_cells.add((row, col))
    return boundary_cells
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


def has_neighbor_zero(grid, row, col):
    if grid[row, col] == -1:  # Ignore cells outside the floorplan
        return False
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1] and grid[new_row, new_col] == 0:
            return True
    return False


def create_plan(m, n, k):
    grid = np.zeros((m, n), dtype=int)
    boundary_cells = set()

    # Initialize the grid with k different colors randomly
    for i in range(k):
        cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        while grid[cell] != 0:
            cell = (random.randint(0, m - 1), random.randint(0, n - 1))
        grid[cell] = i + 1
        boundary_cells.update(get_boundary_cells(grid, cell, True))

    # Randomly choose cells to color until all cells are filled
    steps = 0
    while np.any(grid == 0) and boundary_cells:
        progress_made = False
        for cell in list(boundary_cells):
            if assign_zero_neighbour_value(grid, cell):
                progress_made = True
                boundary_cells.update(get_boundary_cells(grid, cell, False))
            boundary_cells.discard(cell)
        
        if not progress_made:  # Break if no progress is made in a full iteration over boundary cells
            break

        steps += 1

    return grid

def has_neighbor_zero_old(grid, row, col):
    nrows, ncols = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dy, col + dx
        if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
            return True
    return False

def assign_zero_neighbour_value_old(grid, cell):
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

def get_boundary_cells_old(grid, cell, initial):
    """Returns a set of cells that are on the boundary (next to an uncolored cell)."""
    nrows, ncols = grid.shape
    boundary_cells = set()
    if initial:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row, col = cell
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < nrows and 0 <= new_col < ncols and grid[new_row][new_col] == 0:
                boundary_cells.add((new_row, new_col))
    else:
        row, col = cell
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < nrows and 0 <= new_col < ncols:
                if grid[new_row][new_col] != 0 and has_neighbor_zero(grid, new_row, new_col):
                    boundary_cells.add((new_row, new_col))
    return boundary_cells


import numpy as np


def calculate_area_by_color(grid):
    # 색상별 면적(셀의 수)을 저장할 딕셔너리
    area_by_color = {}

    # 그리드를 순회하며 각 색상별로 셀의 수를 계산
    for row in grid:
        for cell in row:
            if cell not in area_by_color:
                area_by_color[cell] = 1
            else:
                area_by_color[cell] += 1

    # -1 (비활성 셀)은 제외하고 반환
    if -1 in area_by_color:
        del area_by_color[-1]

    return area_by_color


import numpy as np


def find_group_edges(grid):
    nrows, ncols = grid.shape
    edge_cells = {}
    visited = np.zeros_like(grid, dtype=bool)

    # 그리드 경계 확인을 위한 조건을 추가하는 함수
    def is_valid(r, c):
        return 0 <= r < nrows and 0 <= c < ncols

    # 인접한 셀이 같은 색상의 그룹에 속하는지 확인하는 함수
    def is_edge(row, col, color):
        if not is_valid(row, col):
            return False
        return grid[row, col] == color and not visited[row, col]

    # 각 색상별로 경계 셀을 찾는 로직
    for row in range(nrows):
        for col in range(ncols):
            color = grid[row, col]
            if color == -1 or visited[row, col]:
                continue

            stack = [(row, col)]
            edges = []
            while stack:
                r, c = stack.pop()
                if visited[r, c]:
                    continue
                visited[r, c] = True

                # 인접한 셀을 확인하고 경계 셀을 식별
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc) and grid[nr, nc] != color:
                        edges.append((r, c))
                        break
                    elif is_edge(nr, nc, color):
                        stack.append((nr, nc))

            if color not in edge_cells:
                edge_cells[color] = set()
            edge_cells[color].update(edges)

    return edge_cells


def calculate_polygon_features(grid):
    edge_cells = find_group_edges(grid)
    vertices_count = {}
    edges_count = {}

    # For each color, calculate vertices and edges
    for color, edges in edge_cells.items():
        vertices = set()
        for cell in edges:
            r, c = cell
            # A vertex is an edge cell with less than 4 adjacent edge cells
            adjacent_edges = sum((nr, nc) in edges for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 for nr, nc in [(r + dr, c + dc)])
            if adjacent_edges < 4:
                vertices.add(cell)

        vertices_count[color] = len(vertices)
        edges_count[color] = len(edges)

    return vertices_count, edges_count
def calculate_shapes(grid):
    nrows, ncols = grid.shape
    color_shapes = {}

    # Directions for orthogonal and diagonal adjacency
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def is_valid(r, c):
        return 0 <= r < nrows and 0 <= c < ncols

    for row in range(nrows):
        for col in range(ncols):
            color = grid[row, col]
            if color not in color_shapes:
                color_shapes[color] = {'vertices': 0, 'edges': 0}

            if color == -1:  # Ignore inactive cells
                continue

            # Count vertices and edges
            vertex_count = 0
            edge_count = 0
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if not is_valid(r, c) or grid[r, c] != color:
                    edge_count += 1
                    if (dr, dc) in directions[4:]:  # Diagonal directions
                        vertex_count += 1

            # Adjust counts based on adjacency type
            if vertex_count > 0:
                color_shapes[color]['vertices'] += 1
            if edge_count > 0:
                color_shapes[color]['edges'] += max(1, edge_count // 2)  # Approximation

    # Post-process to adjust for overcounts
    for color, shape in color_shapes.items():
        shape['edges'] = max(shape['vertices'], shape['edges'] // 2)  # Refine edge count

    return color_shapes



def module_main_rectangle_grid():
    m, n, k = 3,5,4
    grid = create_plan(m, n, k)
    print(grid)


def module_main_non_rectangle_grid():
    # Define your floorplan here
    m, n, k = 5,5,4
    floorplan = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    grid = create_floorplan(m, n, k, floorplan)
    # # Example grid
    grid = np.array([
        [2, 2, 2, -1, -1],
        [4, 1, 1, -1, -1],
        [4, 1, 3, -1, -1],
        [4, 4, 3, 3, 3],
        [4, 4, 3, 3, 3]
    ])


    shapes = calculate_shapes(grid)
    areas = calculate_area_by_color(grid)

    print(grid)
    print(f'area={areas}')
    for color, info in shapes.items():
        if color != -1:  # Ignore inactive cells
            print(f"Color {color}: Vertices = {info['vertices']}, Edges = {info['edges']}")

    # Example usage




if __name__ == "__main__":
    module_main_non_rectangle_grid()