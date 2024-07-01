import numpy as np
from constants import UNIT_SCALE,UNIT_MEASURE
def real_boundary_Length(unit_boundary_length):
    return unit_boundary_length * UNIT_SCALE

def real_area(room):
    room_cells = {(0,0),(0,1)}
    AREA_SCALE = UNIT_SCALE / 1000 if UNIT_MEASURE == 'mm' else UNIT_SCALE
    return AREA_SCALE  * AREA_SCALE * len(room_cells)

def categorize_boundary_cells(grid, room_id):
  """Categorizes grid cells into horizontal and vertical boundary cells.

  Args:
      grid (list of lists): A 2D representation of the grid layout, where each element is a row of cell values (0 for empty, 1 for occupied).

  Returns:
      tuple: A tuple containing two lists:
          - horizontal_boundary_cells: A list of horizontal boundary cell coordinates (row, column)
          - vertical_boundary_cells: A list of vertical boundary cell coordinates (row, column)
  """

  horizontal_boundary_cells = []
  vertical_boundary_cells = []
  rows, cols = len(grid), len(grid[0])

  for row_index in range(rows):
      for col_index in range(cols):
          if grid[row_index][col_index] == room_id:  # Check if cell is occupied
              # Check horizontal boundaries (including all edges)
              if row_index == 0 or row_index == rows - 1:
                  horizontal_boundary_cells.append((row_index, col_index))
              else:
                  # Check for horizontal neighbors only if not on the top or bottom edge
                  if grid[row_index - 1][col_index] != room_id or grid[row_index + 1][col_index] != room_id:
                      horizontal_boundary_cells.append((row_index, col_index))

              # Check vertical boundaries (including all edges and corners)
              if col_index == 0 or col_index == cols - 1:
                  vertical_boundary_cells.append((row_index, col_index))
              else:
                  # Check for vertical boundaries if not on edge and has empty neighbors on both sides
                  if grid[row_index][col_index - 1] != room_id or grid[row_index][col_index + 1] != room_id:
                      vertical_boundary_cells.append((row_index, col_index))
  rooms_cells(grid)
  return horizontal_boundary_cells, vertical_boundary_cells

def boundary_length(grid, room_id):
    horizontal_boundary_cells, vertical_boundary_cells = categorize_boundary_cells(grid, room_id)
    boundary_length = len(horizontal_boundary_cells)+len(vertical_boundary_cells)
    room_boundary_length = real_boundary_Length(boundary_length)
    return room_boundary_length

def rooms_cells(grid):
    rooms = np.unique(grid)
    print(f'rooms={rooms}')


# # Sample grid (replace with your actual grid)
# grid = [
#     [0, 0, 0, 0],
#     [1, 1, 1, 1],
#     [0, 1, 0, 1],
#     [0, 0, 0, 0]
# ]
#
# horizontal_boundary_cells, vertical_boundary_cells = categorize_boundary_cells(grid,1)
#
# print("Horizontal boundary cells:", horizontal_boundary_cells)
# print("Vertical boundary cells:", vertical_boundary_cells)
