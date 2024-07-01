
import numpy as np

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

def main():

    grid = np.array([
        [2, 2, 2, -1, -1],
        [4, 1, 1, -1, -1],
        [4, 1, 3, -1, -1],
        [4, 4, 3, 3, 3],
        [4, 4, 3, 3, 3]
    ])

    shapes = calculate_shapes(grid)  # TODO 2. to calculate the #edge #vertices of the polygon shape, refer to fitness.py of last year code
    areas = calculate_area_by_color(grid)


    print(f'shape={shapes}')
    print(f'area={areas}')
    for color, info in shapes.items():
        if color != -1:  # Ignore inactive cells
            print(f"Color {color}:  모서리수 = {info['vertices']}, 면개수 = {info['edges']}")


if __name__ == '__main__':
    main()