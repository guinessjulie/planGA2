import numpy as np
from GridDrawer import GridDrawer
import trivial_utils
import random
import itertools
import numpy as np

# working version as of 2024-07-19


def grid_print(sample_grid, no=0, format='png', prefix='test', ):
    filename, current_step = trivial_utils.create_filename_in_order(format, prefix, no)
    GridDrawer.color_cells_by_value(sample_grid, filename)


def is_consecutive_neighbors_same_room(floorplan, cell, direction):
    directions = {'north': [(-1, 0), (-1, -1), (-1, 1)],
                  'south': [(1, 0), (1, -1), (1, 1)],
                  'west': [(0, -1), (-1, -1), (1, -1)],
                  'east': [(0, 1), (-1, 1), (1, 1)]
                  }
    consecutive3 = []
    rows, cols = floorplan.shape
    for d in directions[direction]:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        # 경계를 벗어났는지 확인
        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
            consecutive3.append(neighbor)

    if len(consecutive3) < 3:
        return False
    is_consecutive = all([floorplan[n] == floorplan[cell] for n in consecutive3])  # 현재셀과 세 개의 연속셀이 모두 같으면
    # print(f'[is_consecutive_neighbors_same_room]{cell} to  is_consecutive = {is_consecutive}')
    return is_consecutive
# 방은 이미 같다.
def is_same_direction(floorplan, cell1, cell2, d):
    return  (cell1[0]+d[0], cell1[1]+d[1]) == cell2


def is_continuous_direction(directions, different_neighbors):
    for i in range(len(different_neighbors)-1):
        d1 = directions.index(different_neighbors[i%len(different_neighbors)][1])
        d2 = directions.index(different_neighbors[i+1%len(different_neighbors)][1])

        if(d1 + 1) % 4 == d2 or (d1 -1) % 4 == d2:
            return True


def count_two_continuous_directional_neighbors(floorplan, cell, directions, room_value):
    invalid_neighbors=[]
    diffs = []
    sames = []
    for idx, d in enumerate(directions):
        neighbors_neighbors = {}
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if not is_valid_cell(floorplan, neighbor):
            invalid_neighbors.append((neighbor, d))
        else: # if neighbor is valid
            if floorplan[neighbor] != room_value: # 현재 인접셀이 다른 방일때
                diffs.append((neighbor, d))
            else: # 현재 인접셀이 같은 방일 경우 todo 같은 방의 인접셀이 2 개 이상일 때만 이 작업이 필요함 모든 경우에 대해서 핲 필요없음
                sames.append((neighbor, d))
    return invalid_neighbors, diffs, sames
# todo is_boundary 추가됨. 다른 경우도 고려

def is_two_by_two_block_same_room(floorplan, cell, directions):
    rows, cols = floorplan.shape
    neighbors_dir = [((cell[0] + d[0], cell[1] + d[1]), d) for d in directions if
                              0 < cell[0] + d[0] < rows and 0 < cell[1] + d[1] < cols and floorplan[cell] == floorplan[
                                  cell[0] + d[0], cell[1] + d[1]]] # 같은 방이면서 네이버인 셀들
    diagonals = [(x1 + x2, y1 + y2) for (a, (x1, y1)), (b, (x2, y2)) in itertools.combinations(neighbors_dir, 2)if abs(x1 + x2) == 1 and abs(y1 + y2)  == 1]
    if len( diagonals) > 0 :
        if len(diagonals) > 1:
            print(f'diagonals={len(diagonals)} > 1')
    return True

    # is_block =  [(x1 + x2, y1 + y2) for (x1, y1), (x2, y2) in itertools.combinations(neighbors_dir, 2) if abs(x1 + x2) == 1 and abs(y1 + y2) == 1]


def is_cascading_cell(floorplan, cell, is_boundary=False, is_rooms_boundary = False):

    if cell == (4,7):
        print(cell)
    rows, cols = floorplan.shape
    room_value = floorplan[cell]

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    directions8 = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 상하좌우
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 대각선
    direction_names = ["north", "east", "south", "west"]
    direction8_names = ["north",'north_east', "east",'south_east',  "south",'south_west', "west",'north_west']

    # 셀이 유효한 범위에 있는지 확인하는 함수
    if not is_valid_cell(floorplan, cell): return False

    # 인접 셀들 중 현재 셀의 값과 다른 셀들의 위치와 방향을 저장

    diff_cell_dirs = []
    same_cell_dirs = []
    neighbors_direction = {}
    same_room_direction = {}


    invalid_neighbors, neighbor_cell_dir_diff_room, neighbor_cell_dir_same_room = count_two_continuous_directional_neighbors(floorplan, cell, directions, room_value)
    if is_boundary_cell(floorplan, cell) :
        if cell == (5,7):
            print(cell)
        if len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) >= 3 :
            print(f'{cell} True because boundary and len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')
            return True
        else:
            # todo (5, 7) 의 경우 왜 boundary인지를 먼저 확인하고, 왜냐하면 네 개의 모든 셀이 다 valid 하므로 이것을 boundary로 치면 안된다. boundary 로직을 살펴보자
            # todo invalid_neighbors 가 0이고  neighbors_diff가 2이므로 만일 이것을 boundary로 인지하면 return Fasle의 로직이 샌다.
            # 그러므로 boundary 로직을 고치던지, 아니면 여기서 다시 3보다 작더라도 즉 2가 되더라도 뭔가를 체크하는 로직을 만들던지 하자. 오늘은 그만 잠
            print(f'{cell} False because boundary and NOT len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')


            return False
    else: # not boundary_cell # no invalid_neighbors
        if len(neighbor_cell_dir_diff_room) >= 3 :
            return True
        if len(neighbor_cell_dir_diff_room) == 2:
            if len(neighbor_cell_dir_same_room) != 2:
                print(f'{cell} diff=2, same!= 2 how this can happens')
            else:
                same_room_neighbors_cells = [same for same in neighbor_cell_dir_same_room]
                diff_room_neighbors_cells = [diff for diff in neighbor_cell_dir_diff_room]

                for diff_nei, d in diff_room_neighbors_cells:
                    for d2 in directions:
                        # 옆방에서 대각선 방향으로 자신과 같은 룸이 있다면
                        diag_neighbor_me = [(diff_nei[0]+d2[0], diff_nei[1]+d2[1]) for d2 in directions \
                                         if (diff_nei[0]+d2[0], diff_nei[1]+d2[1]) != cell \
                                         and is_valid_cell(floorplan, (diff_nei[0]+d2[0], diff_nei[1]+d2[1])) \
                                         and floorplan[(diff_nei[0]+d2[0], diff_nei[1]+d2[1])] == floorplan[cell]]
                        if len(diag_neighbor_me) > 0 :
                            print(f'{cell} True ... has diagonal corner at {diag_neighbor_me}')
                            return True


                # todo 이건 디버깅용으로 확인만 하고 나중에 지울 것임
                two_step_away = [((nei[0] + di[0], nei[1] + di[1]), (nei[0], nei[1]), (di[0], di[1])) for nei, di in
                                 same_room_neighbors_cells if
                                 nei[0] + di[0] < rows and nei[1] + di[1] < cols]

                # 같은 방향으로 연속 같은 방일 때
                two_step_away_same_room = [((nei[0] + di[0], nei[1] + di[1]), (nei[0], nei[1]), (di[0], di[1])) for
                                           nei, di in same_room_neighbors_cells if
                                           nei[0] + di[0] < rows and nei[1] + di[1] < cols and floorplan[cell] ==
                                           floorplan[nei[0] + di[0], nei[1] + di[1]]]

                if len(two_step_away_same_room) >= 2:
                    dir_same_pairs = [di for double_nei, nei, di in two_step_away_same_room]
                    # dir_diff_pairs = [di for double_nei, nei, di in two_step_away_diff_room]
                    # todo 이중 인접셀이 같은 경우가 2 개의 경우만 생각함. 3 개 이상인 경우는 당연히 False 일 것 같은에?
                    diagonal = [(x1 + x2, y1 + y2) for (x1, y1), (x2, y2) in itertools.combinations(dir_same_pairs, 2)
                                if abs(x1 + x2) == 1 and abs(y1 + y2) == 1]
                    is_orthogonal = len(diagonal) > 0
                    if is_orthogonal:
                        diagonal_cell = (cell[0] + diagonal[0][0], cell[1] + diagonal[0][1])
                        diagonal_room = floorplan[diagonal_cell]
                        if diagonal_room == room_value:
                            print(f'{cell} False because {diagonal_cell} is the same room ? nop')
                            return False
                        else:
                            print(f'{cell} True becuase {diagonal_cell} is the different room')
                            return True


    # same_cell_dirs 는 현재 cell의 이웃 중 같은 방. 같은 방에 있는 직접 이웃과 그 이웃의 방향을 원소로 가진다.
    # C가 현재 셀일때 다음 형태로 같은 방이면 False
    # cxx
    # xx
    # x
    if len(neighbor_cell_dir_same_room) >= 2 :
        print(f'{cell}: len(neighbor_cell_dir_same_room) >= 2)')
        # 직접 이웃의 방향만큼 더해주고 그셀의 값이 같으면(같은 방이면) 추가


        # 직접 이웃의 두 개의 방향을 서로 더한다.
        # two_step_away_same_room의 원소들이 같은 바인지 체크 # todo  아래 for 문은 위에서 이미 했다. two_step_away_same_room에서  triple을 가진다.

        # # 연속 같은 방향으로 같은 방이면 cascading_cell이 아니다. => todo 여기서 same room direction을 체크하지 않음
        # if is_rooms_boundary and len(continuous_same_direction) >= 2 :
        #     return False
      # 인접 셀들의 방향이 시계방향 또는 반시계방향으로 연속되어 있는지 확인
    # todo boundary 를 따지지 말고 invalid_cell의 갯수와 diff_cell의 갯수를 합쳐서 세 개 가 넘으면 cascading ?  아냐 boundary를 따져야돼
    if len(neighbor_cell_dir_diff_room) + len(invalid_neighbors) >= 3: # 다른 게 두 개 이상 되면
        return True

    # if len(neighbor_cell_dir_diff_room) >= 2: # 다른 게 두 개 이상 되면
        # corner 인 경우 diff가  하나만 있어도 cascading?
        # todo boundary 를 따지지 말고 invalid_cell의 갯수와 diff_cell의 갯수를 합쳐서 세 개 가 넘으면 cascading ?
        # boundary가 아니면서, 네이버가 둘 인 경우만 다음 수행?


        # two_continous_neighbor_different = is_continuous_direction(directions, neighbor_cell_dir_diff_room)
        # two_continous_neighbor_same = is_continuous_direction(directions, neighbor_cell_dir_same_room)
        # if two_continous_neighbor_different and two_continous_neighbor_same:
        #     s1 = directions.index(same_room_direction[0%len(neighbor_cell_dir_same_room)][1])
        #     s2 = directions.index(same_room_direction[1%len(neighbor_cell_dir_same_room)][1])
        #     s3 = s1+s2
        #     cell3 = cell + s3
        #     if floorplan[cell3] == floorplan[cell]:
        #         print(f'this is corner')
        #         return True

    two_by_two = is_two_by_two_block_same_room(floorplan, cell, directions)
    if two_by_two:
        print(f'{cell} False becuase two_by_two_block_same_room ')
        return False

# TODO MOVED TO is_continous_direction(directions, neighbor_cell_dir_diff_room)
#        for i in range(len(neighbor_cell_dir_diff_room)-1):
#            d1 = directions.index(neighbor_cell_dir_diff_room[i%len(neighbor_cell_dir_diff_room)][1])
#            d2 = directions.index(neighbor_cell_dir_diff_room[i+1%len(neighbor_cell_dir_diff_room)][1])
#
#            if(d1 + 1) % 4 == d2 or (d1 -1) % 4 == d2:
#                return True
        # todo: 바운더리 셀이 커버되지 않으므로 당연히 에러가 나며, 나머지 모든 방향에 대해서 방향성을 체크해봐야 된다.
        #  (0,4)의 경우 boundary 라인에 있으므로 네이버의 개수는 3개고 3 개중 2개가 다르면 1개만 같고 이 경우 cascading cell
        #  (0,8)이 경우 하나의 위와 마찬가지
        #  (0,9)의 경우는 코너셀이다. 오루지 2 개의 neighbor cell만 존재하는데 그 중 1 개가 다르면 이 셀은 cascading ? 모든 경우가 그러한지 잘 모르겠음
        #  (1, 4)의 경우는 코너셀이 아닌데 연속 두 개가 다르므로  cascading_cell
    # todo (1, 7)의 경우 룸코너이다. 이 경우
    # else:  # boundary 인 경우
    #     if len(diffs) >= 2: return True
    #     return False

def valid_neighbors_in_room(floorplan, cell):
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0 and floorplan[cell] == floorplan[neighbor]:
            neighbors.append(neighbor)
    return neighbors

def all_active_neighbors(cell, floorplan):
    directions8 = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 상하좌우
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 대각선

    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0:
            neighbors.append(neighbor)
    return neighbors

def all_active_neighbors_directions(cell, floorplan):

    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction_names = ["north", "east", "south", "west"]
    neighbors = []
    neighbors_direction = {}
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0:
            neighbors.append(neighbor)
            neighbors_direction[neighbor] = directions4.index(d)
    return neighbors_direction


def all_neighbors(cell, floorplan):
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0:
            neighbors.append(neighbor)
    return neighbors

def cell_swap(floorplan, cell1, cell2):
    temp = floorplan[cell1[0], cell1[1]]
    floorplan[cell1[0], cell1[1]] = floorplan[cell2[0], cell2[1]]
    floorplan[cell2[0], cell2[1]] = temp
    return temp

def get_diff_ratio_neighbors(floorplan, cell):
    # cells = all_active_neighbors8(cell, floorplan)
    cells = all_neighbors(cell, floorplan)
    diff_ratio_dict = {}
    for neighbor_cell in cells:
        if floorplan[cell] != floorplan[neighbor_cell]:
            diff_ratio = get_diff_ratio_cell(floorplan, neighbor_cell)
            diff_ratio_dict[neighbor_cell] = diff_ratio
    return diff_ratio_dict


def get_diff_ratio_cell(floorplan, cell):
    # neighbors = all_active_neighbors8(cell, floorplan)
    # neighbors = all_active_neighbors(cell, floorplan)
    neighbors = all_neighbors(cell, floorplan)
    max_neighbors = len(neighbors)
    if max_neighbors == 0.:
        print(neighbors, cell)
    num_diff_neighbors = sum(
        1 for neighbor in neighbors if floorplan[neighbor[0], neighbor[1]] != floorplan[cell[0], cell[1]])
    diff_ratio = num_diff_neighbors / max_neighbors
    # print(f'max_neighbors = {max_neighbors}, num_diff_neighbors = {num_diff_neighbors}, diff_ratio = {diff_ratio} returning {diff_ratio}')
    # print(f'\t\t\t[get_diff_ratio_cell] ...when {cell}={floorplan[cell]}s diff_ratio = {diff_ratio} max_neighbors {max_neighbors}')
    return diff_ratio


def create_diff_ratio_array(floorplan):
    diff_ratios = np.zeros_like(floorplan, dtype=float)

    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cell = (i, j)
            if is_valid_cell(floorplan, cell):
                diff_ratios[cell] = get_diff_ratio_cell(floorplan, cell)
                # print(f'{cell} diff_ratio={diff_ratios[cell]}')
    return diff_ratios


def is_valid_cell(floorplan, cell):
    is_valid = 0 <= cell[0] < floorplan.shape[0] and 0 <= cell[1] < floorplan.shape[1] and floorplan[cell] > 0
    # print(f'{cell} is_valid={is_valid}')
    return is_valid


def get_neighbors_room_numbers(floorplan, cell):
    neighbors = all_active_neighbors(cell, floorplan)
    cell_neighbors_room_number = []
    for neighbor in neighbors:
        if floorplan[cell] != floorplan[neighbor]:
            cell_neighbors_room_number.append(neighbor)
    return cell_neighbors_room_number


def get_cell_neighbors_room_number(floorplan, cell):
    neighbors = all_active_neighbors(cell, floorplan)
    cell_neighbors_room_number = []
    for neighbor in neighbors:
        if is_valid_cell(floorplan, cell) and floorplan[cell] != floorplan[neighbor]:
            cell_neighbors_room_number.append(floorplan[neighbor])
    return cell_neighbors_room_number



# 지정한 동서남북 중 하나의 방향으로 연속으로 세 쌍의 셀이 셀 자신과 그 값이 같으면(즉 같은 방이면) 테스티드 o.k.
# todo 코너에 위치한 셀들은 모두 False를 리턴하므로 이것에 대해서 숙고해서 처리해주어야 함
def is_consecutive_neighbors_same_room(floorplan, cell, direction):
    directions = {'north': [(-1, 0), (-1, -1), (-1, 1)],
                  'south': [(1, 0), (1, -1), (1, 1)],
                  'west': [(0, -1), (-1, -1), (1, -1)],
                  'east': [(0, 1), (-1, 1), (1, 1)]
                  }
    consecutive3 = []
    rows, cols = floorplan.shape
    for d in directions[direction]:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        # 경계를 벗어났는지 확인
        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
            consecutive3.append(neighbor)

    if len(consecutive3) < 3:
        return False
    is_consecutive = all([floorplan[n] == floorplan[cell] for n in consecutive3])  # 현재셀과 세 개의 연속셀이 모두 같으면
    # print(f'[is_consecutive_neighbors_same_room]{cell} to  is_consecutive = {is_consecutive}')
    return is_consecutive

def change_cell_new_room_number(floorplan, cell, new_room_number):
    temp = floorplan[cell]
    floorplan[cell] = new_room_number
    neighbors = list(set(all_active_neighbors(cell, floorplan)))
    valid = any(floorplan[n[0], n[1]] == floorplan[cell[0], cell[1]] for n in neighbors)
    # valid = valid and (not is_cascading_cell(floorplan, cell))
    if valid:
        return True
    else:  # revert
        floorplan[cell] = temp
        return False


def change_room_cell(floorplan, cell):
    cell_neighbors_room_numbers = get_cell_neighbors_room_number(floorplan, cell)

    changed = False
    while cell_neighbors_room_numbers:
        new_room_number = random.choice(cell_neighbors_room_numbers)
        changed = change_cell_new_room_number(floorplan, cell, new_room_number)
        if changed: return True
        cell_neighbors_room_numbers.remove(new_room_number)
    if not changed: return False

def is_boundary_cell(floorplan, cell):
    rows, cols = floorplan.shape
    if floorplan[cell] == -1: return False
    if cell[0] == 0 or cell[1] == 0 or cell[0] == (rows -1)  or cell[1] == (cols -1): return True
    # 방향: 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 이웃 셀들 중 경계 바깥에 있거나 -1이 있는지 확인
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if not is_valid_cell(floorplan, neighbor) or floorplan[neighbor] == -1:
            return True

    return False

def create_cell_room_corners(floorplan):
    corners = np.zeros_like(floorplan, dtype=int)

    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cell = (i, j)
            if cell == (0, 5):
                print(f'create_cell_room_corners:{cell}')
            if is_valid_cell(floorplan, cell):
                corners[cell] = is_cell_room_corner(floorplan, cell)
    return corners



# 방의 코너를 모두 찾는다. 코너와 한줄 짜리 셀을 모두 찾는다.
def is_cell_room_corner(floorplan, cell):
    rows, cols = floorplan.shape

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    # 결과 배열 초기화 (-1은 그대로 유지)
    boundary_cells = np.where(floorplan == -1, -1, 0)
    invalid_different_neighbor_count = 0
    current_value = floorplan[cell]
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        # 인접셀이 다른 방이거나 invalid 셀인 경우를 모두 count
        if not is_valid_cell(floorplan, neighbor) or floorplan[neighbor] != current_value:
            invalid_different_neighbor_count += 1
    if invalid_different_neighbor_count >= 2:
        return True
    else:
        return False


def create_room_boundary_cells_array(floorplan):
    rows, cols = floorplan.shape

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]


    # 결과 배열 초기화 (-1은 그대로 유지)
    boundary_cells = np.where(floorplan == -1, -1, 0)
    for i in range(rows):
        for j in range(cols):
            if floorplan[i, j] == -1:
                continue
            current_value = floorplan[i, j]
            for d in directions:
                neighbor = (i + d[0], j + d[1])
                # 경계 셀은 배열의 경계를 벗어나는 셀과 인접한 셀로 정의
                if not is_valid_cell(floorplan, neighbor) or floorplan[neighbor] != current_value:
                    boundary_cells[i, j] = 1
                    break

    return boundary_cells


def create_boundary_cells_array(floorplan):
    # 경계 셀 여부를 저장할 배열 초기화
    # boundary_cells = np.full(floorplan.shape, False, dtype=int)
    boundary_cells = np.where(floorplan == -1, -1,  0)
    # 모든 셀에 대해 경계 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            if floorplan[i, j] == -1:
                continue
                # boundary_cells[i, j] = -1
            else:
                boundary_cells[i, j] = is_boundary_cell(floorplan, (i, j))

    # 결과 출력
    print('boundary_cells')
    for row in boundary_cells:
        print("[", " ".join(f"{int(x):2}" if x != -1 else f"{x:2}" for x in row), "]")

    return boundary_cells

def random_cascading_cell(cascading_cells):
    indices = np.argwhere(cascading_cells == 1)
    if len(indices) == 0:
        return None
    return tuple(indices[np.random.choice(len(indices))])


def update_cascading_cells_array(floorplan, cascading_cells_array, cell):
    # cell의 cascading 여부를 먼저 확인후
    cascading_cells_array[cell] = is_cascading_cell(floorplan, cell)

    # 이웃 셀들의 cascading 여부를 확인
    neighbors = all_active_neighbors(cell, floorplan)
    for neighbor_cell in neighbors:
        cascading_cells_array[neighbor_cell] = is_cascading_cell(floorplan, neighbor_cell)
    return cascading_cells_array


def exchange_protruding_cells(floorplan, iteration=1):
    cascading_cells = create_cascading_cells(floorplan)
    room_boundary_cells = create_room_boundary_cells_array(floorplan)
    print(f'room_boundary_cells=\n{room_boundary_cells}')
    if not np.any(cascading_cells == 1):
        return
    for _ in range(iteration):
        if not np.any(cascading_cells == 1):
            return
        cell = random_cascading_cell(cascading_cells)
        print(f'random_cell = {cell}')
        room_number = floorplan[cell]
        print(f'For Room {room_number} : cell {cell} ')
        if change_room_cell(floorplan, cell):
            print(f'floorplan after change_room_cell {cell} \n{floorplan}')
            update_cascading_cells_array(floorplan, cascading_cells, cell)
            grid_print(floorplan)


def create_cascading_cells(floorplan):
    # 코너 셀 여부를 저장할 배열 초기화
    cascading_cells = np.zeros_like(floorplan, dtype=bool)
    boundary_cells = create_boundary_cells_array(floorplan)
    rooms_boundary_cells = create_room_boundary_cells_array(floorplan)
    room_cell_corners = create_cell_room_corners(floorplan)
    print(f'cascading_cells = \n{cascading_cells}')
    print(f'boundary_cells = \n{boundary_cells}')
    print(f'rooms_boundary_cells = \n{rooms_boundary_cells}')
    print(f'room_cell_corners = \n{room_cell_corners}')
    # 모든 셀에 대해 코너 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            is_boundary = boundary_cells[i, j]
            is_rooms_boundary = rooms_boundary_cells[i, j]
            cascading_cells[i, j] = is_cascading_cell(floorplan, (i, j), is_boundary, is_rooms_boundary)
    print('cascading_cells')
    # 결과 출력 (칸을 맞춰서)
    for row in cascading_cells:
        print("[", " ".join(f"{int(x):2}" for x in row), "]")

    return cascading_cells

if __name__ == '__main__':
    floorplan = np.array([
        [5, 4, 4, 4, 4, 3, 3, 3, 3, 2, -1, -1, -1, -1],
        [5, 4, 4, 4, 3, 3, 3, 2, 2, 2, -1, -1, -1, -1],
        [5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
        [5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1],
        [5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1, 1, -1, -1],
        [5, 5, 5, 4, 3, 3, 3, 1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, 3, 3, 1, 1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 3, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    ])
    grid_print(floorplan)
    # test_module(floorplan)
    exchange_protruding_cells(floorplan, 10)
    # grid_print(floorplan)

