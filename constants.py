DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 인접 셀을 확인하기 위한 방향 벡터
NUM_SPACE = 4
UNIT_SCALE = 1000
UNIT_MEASURE = 'mm'
# todo test

coordinates = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2),
                        (4, 2),
                        (5, 2), (5, 3)
    ]
floor_grid = [
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
]

floor_grid = [
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 0]
]
simple_grid=[
    [1,0],
    [1,1],
    [1,0],
    ]
test_grid = [
    [2, 2, 2, -1, -1],
    [4, 1, 1, -1, -1],
    [4, 1, 3, -1, -1],
    [4, 4, 3, 3, 3],
    [4, 4, 3, 3, 3]
]
test_grid2 = [
    [ 2,  2,  2, -1, -1],
    [ 2,  2,  4, -1, -1],
    [ 3,  4,  4, -1, -1],
    [ 3,  4,  4,  1,  1],
    [ 3,  3,  1,  1,  1]
]

test_grid_side = [
    [-2, -2, -2, -2, -2, -2, -2],
    [-5, 2, 2, 2, -1, -1, -4],
    [-5, 4, 1, 1, -1, -1, -4],
    [-5, 4, 1, 3, -1, -1, -4],
    [-5, 4, 4, 3, 3, 3, -4],
    [-5, 4, 4, 3, 3, 3, -4],
    [-3, -3, -3, -3, -3, -3, -3]
]
