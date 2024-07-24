import numpy as np
from GridDrawer import GridDrawer
import trivial_utils
import random

threshold_diff_ratio = .49
def grid_print(sample_grid, no=0, format='png', prefix = 'test',):
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

def consecutive_neighbor_in_orientation(floorplan, cell):
    orientation = ['north', 'south', 'west', 'east']
    count_true = 0
    triple_match_directions = []
    for o in orientation:
        if is_consecutive_neighbors_same_room(floorplan, cell, o):
            triple_match_directions.append(o)
    # print(f'[{cell}]: triple_match_directions{triple_match_directions}')
    return triple_match_directions

# todo is_boundary 추가됨. 다른 경우도 고려
def is_cascading_cell(floorplan, cell, is_boundary = False):
    def is_valid(cell):
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols and floorplan[cell] > 0


    rows, cols = floorplan.shape
    room_value = floorplan[cell]

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_names = ["north", "east", "south", "west"]

    # 셀이 유효한 범위에 있는지 확인하는 함수
    if not is_valid(cell) : return False

    # 인접 셀들 중 현재 셀의 값과 다른 셀들의 위치와 방향을 저장
    different_neighbors = []
    same_neighbors = []
    for idx, d in enumerate(directions):
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if (is_valid(neighbor)) :
            if floorplan[neighbor] != room_value:
                different_neighbors.append((neighbor, direction_names[idx]))
            else:
                same_neighbors.append((neighbor, direction_names[idx]))

    # 인접 셀들의 방향이 시계방향 또는 반시계방향으로 연속되어 있는지 확인
    if not is_boundary:
        if len(different_neighbors) != 2:
            return False
        # boundary가 아니면서, 네이버가 둘 인 경우만 다음 수행
        diff_dir1 = direction_names.index(different_neighbors[0][1])
        diff_dir2 = direction_names.index(different_neighbors[1][1])
        if (diff_dir1 + 1) % 4 == diff_dir2 or (diff_dir1 - 1) % 4 == diff_dir2:
            return True
        return False
        # todo: 바운더리 셀이 커버되지 않으므로 당연히 에러가 나며, 나머지 모든 방향에 대해서 방향성을 체크해봐야 된다.
        #  (0,4)의 경우 boundary 라인에 있으므로 네이버의 개수는 3개고 3 개중 2개가 다르면 1개만 같고 이 경우 cascading cell
        #  (0,8)이 경우 하나의 위와 마찬가지
        #  (0,9)의 경우는 코너셀이다. 오루지 2 개의 neighbor cell만 존재하는데 그 중 1 개가 다르면 이 셀은 cascading ? 모든 경우가 그러한지 잘 모르겠음
        #  (1, 4)의 경우는 코너셀이 아닌데 연속 두 개가 다르므로  cascading_cell
    else: # boundary 인 경우
        if len(different_neighbors) >= 2 : return True
        return False

def is_cascading_corner(triple_match_directions): # consecutive3_direction 세 셀이 같은 방향

    if len(triple_match_directions) != 2: return False
    if 'north' in triple_match_directions  or 'south' in triple_match_directions :
        if 'east'  in triple_match_directions or 'west' in triple_match_directions :
            return True
    if 'east' in  triple_match_directions or  'west' in triple_match_directions:
        if ('north' in triple_match_directions  or 'south' in triple_match_directions) :
            return True

    # else:
    #     # triple_match_directions와 별개로 코너나 에지에 있는지를 확인해서
    #
    #     # 상하좌우 인접노드들의 셀 수를 세자
    #     pass

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


def all_neighbors(cell, floorplan):
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0:
            neighbors.append(neighbor)
    return neighbors


def all_different_room_neighbors(cell, floorplan):
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0 and floorplan[cell] != floorplan[neighbor]:  # 방이 다른 조건 추가
            neighbors.append(neighbor)
    return neighbors


def all_active_neighbors8(cell, floorplan):
    directions8 = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 상하좌우
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 대각선

    neighbors = []
    for d in directions8:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[neighbor] > 0:
            neighbors.append(neighbor)
    return neighbors


def get_protruding_cells(floorplan):
    protruding_cells_dict = {}
    diff_ratio_dict = {}
    for row in range(floorplan.shape[0]):
        for col in range(floorplan.shape[1]):
            room_number = floorplan[row, col]
            if room_number > 0:
                if room_number not in protruding_cells_dict:
                    protruding_cells_dict[room_number] = []
                cell = (row, col)
                # neighbors = all_active_neighbors8(cell, floorplan)
                neighbors = all_neighbors(cell, floorplan)
                different_neighbors = sum(floorplan[neighbor] != room_number for neighbor in neighbors)
                diff_ratio = different_neighbors / len(neighbors)
                diff_ratio_dict[cell] = diff_ratio
                # 여기에다가 만일 0.5일 경우 삐죽삐죽한 corner인지 결정한다.
                if diff_ratio > threshold_diff_ratio:
                    protruding_cells_dict[room_number].append(cell)
                    # print(f'diff_ratio of {cell} = {diff_ratio}')
    # 빈 리스트는 제거하고 값이 있는 헬만 가져온다.
    protruding_cells_dict = {k: v for k, v in protruding_cells_dict.items() if v}

    return protruding_cells_dict


def rooms_cells(floorplan):
    cell_dict = {}
    for row in range(floorplan.shape[0]):
        for col in range(floorplan.shape[1]):
            room_number = floorplan[row, col]
            if room_number > 0:
                if room_number not in cell_dict:
                    cell_dict[room_number] = []
                cell_dict[room_number].append((row, col))
    return cell_dict


def cell_swap(floorplan, cell1, cell2):
    temp = floorplan[cell1[0], cell1[1]]
    floorplan[cell1[0], cell1[1]] = floorplan[cell2[0], cell2[1]]
    floorplan[cell2[0], cell2[1]] = temp
    return temp


def find_neighbor_cell_to_swap(floorplan, extreme_cell, extreme_diff_ratio):
    neighbors = all_active_neighbors(extreme_cell, floorplan)
    max_neighbors = len(neighbors)
    max_diff_ratio = -1
    concave_cell = None

    diff_ratio_dict = {}
    for cell in neighbors:
        if floorplan[cell[0], cell[1]] != floorplan[
            extreme_cell[0], extreme_cell[1]]:  # 현재 셀은 extreme_cel의 인접셀이며, 모든 인접셀 비교해서 다른 방이면 다음 실행
            neighbors_of_neighbor = all_active_neighbors(cell, floorplan)  # 그 인접셀의 모든 유효 인접셀을 구한 후
            max_neighbors = len(neighbors_of_neighbor)
            num_different_neighbors = sum(1 for neighbor in neighbors_of_neighbor if
                                          floorplan[neighbor[0], neighbor[1]] != floorplan[
                                              cell[0], cell[1]])  # 그 인접셀의 유효인접셀들이 extreme_cell의 인접셀과 다른 개수를 구한다.

            diff_ratio = num_different_neighbors / max_neighbors
            diff_ratio_dict[cell] = diff_ratio  # 여기서 제일 큰 값을 구하기 위해서
            print(
                f'[find_neighbor_cell_to_swap] {cell} diff_ratio = {diff_ratio} > max_diff_ratio = {max_diff_ratio} and diff_ratio={diff_ratio} > extreme_diff_ratio {extreme_diff_ratio}')
    cell = max(diff_ratio_dict, key=diff_ratio_dict.get)
    diff_ratio = diff_ratio_dict[cell]
    print(f'[find_neighbor_cell_to_swap] cell={cell}, diff_ratio={diff_ratio}')
    return cell

    # if diff_ratio > max_diff_ratio and diff_ratio > extreme_diff_ratio:
    #             max_diff_ratio = diff_ratio
    #             concave_cell = cell

    return concave_cell


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
    if max_neighbors == 0. :
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


def change_room_cells(floorplan, room_number, cells):
    for cell in cells:
        cell_neighbors_room_numbers = get_neighbors_room_numbers(floorplan, cell)  # 같은 건 이미 필터링됨
        if cell_neighbors_room_numbers:
            print(f'cell_neighbors_room_numbers={cell_neighbors_room_numbers}')
            selected_neighbor_cell = random.choice(cell_neighbors_room_numbers)
            selected_neighbor_cell_room = floorplan[selected_neighbor_cell]
            floorplan[cell] = selected_neighbor_cell_room
            print(f'cell{cell} changed from {room_number}to{floorplan[cell]}')


def is_valid_swap(floorplan, cell1, cell2):
    # Perform the swap
    # diff_ratio_cell1 = get_diff_ratio_cell(floorplan, cell1)
    # diff_ratio_cell1_neighbors = get_diff_ratio_neighbors(floorplan, cell1)
    # diff_ratio_cell2 = get_diff_ratio_cell(floorplan, cell2)
    # diff_ratio_cell2_neighbors = get_diff_ratio_neighbors(floorplan, cell2)
    # print(f'{cell1}{diff_ratio_cell1:.2f}:{diff_ratio_cell1_neighbors}')
    # print(f'{cell2}{diff_ratio_cell2:.2f}:{diff_ratio_cell2_neighbors}')

    temp = cell_swap(floorplan, cell1, cell2)
    # print(f'after {cell1} swap {cell2} : \n{floorplan }')

    # diff_ratio_cell1 = get_diff_ratio_cell(floorplan, cell1)
    # diff_ratio_cell1_neighbors = get_diff_ratio_neighbors(floorplan, cell1)
    # diff_ratio_cell2 = get_diff_ratio_cell(floorplan, cell2)
    # diff_ratio_cell2_neighbors = get_diff_ratio_neighbors(floorplan, cell2)
    # print(f'{cell1}{diff_ratio_cell1:.2f}:{diff_ratio_cell1_neighbors}')
    # print(f'{cell2}{diff_ratio_cell2:.2f}:{diff_ratio_cell2_neighbors}')

    # Check if the swap maintains the rule
    neighbors1 = all_active_neighbors(cell1, floorplan)
    neighbors2 = all_active_neighbors(cell2, floorplan)

    # valid 변수는 neighbors1 중 하나가 cell1과 같은 값을 가지는지,
    # neighbors2 중 하나가 cell2와 같은 값을 가지는지를 모두 확인하여 결정됩니다.
    # 두 조건이 모두 만족되면 valid는 True가 되고, 하나라도 만족되지 않으면 False가 됩니다.

    valid = any(floorplan[n[0], n[1]] == floorplan[cell1[0], cell1[1]] for n in neighbors1) and \
            any(floorplan[n[0], n[1]] == floorplan[cell2[0], cell2[1]] for n in neighbors2)

    # print(f'\t[is_valid_swap] neighbors of {cell1}{floorplan[cell1[0], cell1[1]]} = {[((n[0], n[1]), floorplan[n[0], n[1]]) for n in neighbors1]}={[floorplan[n[0], n[1]] for n in neighbors1]}')
    # print(f'\t[is_valid_swap] neighbors of {cell2}{floorplan[cell2[0], cell2[1]]} = {[((n[0], n[1]), floorplan[n[0], n[1]]) for n in neighbors2]}={[floorplan[n[0], n[1]] for n in neighbors2]}')
    print(f'\t[is_valid_swap] cell1{cell1},cell2{cell2} valid={valid}')

    # Revert the swap
    floorplan[cell2[0], cell2[1]] = floorplan[cell1[0], cell1[1]]
    floorplan[cell1[0], cell1[1]] = temp

    return valid


# 특정 룸 넘버에 해당하는 좌표를 찾는 함수
def find_coordinates(floorplan, room_number):
    coordinates = np.argwhere(floorplan == room_number)
    return [tuple(coord) for coord in coordinates]

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
def consecutive_neighbor_in_orientation(floorplan, cell):
    orientation = ['north', 'south', 'west', 'east']
    count_true = 0
    triple_match_directions = []
    for o in orientation:
        if is_consecutive_neighbors_same_room(floorplan, cell, o):
            triple_match_directions.append(o)
    # print(f'[consecutive_neighbor_in_orientation]: triple_match_directions{triple_match_directions}')
    return triple_match_directions

# cell을 room_number로 바꾸는 것이 과연 타당한가
def is_valid_change(floorplan, cell, room_number, current_diff_ratio):
    temp = floorplan[cell]  # 현재 셀의 룸번호를 템프에 저장
    floorplan[cell] = room_number  # 일단 바꾸고
    new_diff_ratio = get_diff_ratio_cell(floorplan, cell)
    print(f'\t\t[is_valid_change] changed {cell} to {floorplan[cell]}')

    neighbors = all_active_neighbors(cell, floorplan)  # 모든 인접셀들을 구한후
    # 바뀐 값이 각각의 액티브 네이버 중 어느 하나라도 현재셀의 값이 같으면 valid
    valid = any(floorplan[n[0], n[1]] == floorplan[cell[0], cell[1]] for n in neighbors)
    # valid = valid and is_corner_protruding(floorplan, cell, new_diff_ratio) => todo protruding_cell 구할 때 케어해보자
    valid = valid and (current_diff_ratio > new_diff_ratio) # 바뀐 값이 더 작아야 valid
    print(f'\t\t[is_valid_change](current_diff_ratio {current_diff_ratio} > new_diff_ratio {new_diff_ratio})={(current_diff_ratio > new_diff_ratio)}')
    print(
        f'\t\t[is_valid_change] neighbors: {neighbors}=>{[floorplan[n[0], n[1]] for n in neighbors if floorplan[n[0], n[1]] == floorplan[cell[0], cell[1]]]} isequal {cell}=>{floorplan[cell[0], cell[1]]}')
    if valid:
        return True
    else:
        floorplan[cell] = temp
        print(f'\t\t\t[is_valid_change] reverted {cell} to {floorplan[cell]}')
        return False


# 선택된 셀을 해당 룸번호에서 다른 룸번호로 변경한다.
def change_room_cell_old(floorplan, room_number, cell):
    # 먼저 이웃셀들의 방 번호들을 가져온다.
    cell_neighbors_room_numbers = get_neighbors_room_numbers(floorplan, cell)  # 같은 건 이미 필터링됨
    current_diff_ratio = get_diff_ratio_cell(floorplan, cell)
    valid_change_success = False
    while cell_neighbors_room_numbers:
        selected_neighbor_cell = random.choice(cell_neighbors_room_numbers)
        cell_neighbors_room_numbers.remove(selected_neighbor_cell)
        selected_neighbor_cell_room = floorplan[selected_neighbor_cell]
        valid_change_success = is_valid_change(floorplan, cell, selected_neighbor_cell_room, current_diff_ratio)
        # 만일 변경이 성공했다면 while 문을 빠져나온다.
        if valid_change_success:
            return selected_neighbor_cell_room
    # 만일 변경이 성공하지 못했다면 while문을 반복하면서 새로운 이웃셀을 선택한다.
    # 이웃 셀이 없으면 while 문을 빠져나오고 셀을 변경하지 못했으므로 원래의 room_number를 리턴한다. .
    return room_number

def change_cell_new_room_number(floorplan, cell, new_room_number):
    temp = floorplan[cell]
    floorplan[cell] = new_room_number
    neighbors = list(set(all_active_neighbors(cell, floorplan)))
    valid = any(floorplan[n[0], n[1]] == floorplan[cell[0], cell[1]] for n in neighbors)
    # valid = valid and (not is_cascading_cell(floorplan, cell))
    if valid: return True
    else: # revert
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

def change_room_cell2(floorplan, room_number, cell):
    cell_neighbors_room_numbers = get_neighbors_room_numbers(floorplan, cell)  # 같은 건 이미 필터링됨
    valid_change = False
    current_diff_ratio = get_diff_ratio_cell(floorplan, cell)
    while not valid_change:
        if cell_neighbors_room_numbers:
            print(f'cell_neighbors_room_numbers={cell_neighbors_room_numbers}')
            selected_neighbor_cell = random.choice(cell_neighbors_room_numbers)  # 네이버 중 하나를 골라
            selected_neighbor_cell_room = floorplan[selected_neighbor_cell]
            valid_change = is_valid_change(floorplan, cell, selected_neighbor_cell_room, current_diff_ratio)
            print(f'is_valid_change returned: room {cell} to  {selected_neighbor_cell_room} {valid_change}')
            # floorplan[cell] = selected_neighbor_cell_room # is_valid_change에서 이미 변경함

            if valid_change:
                print(f'cell{cell} changed from [{room_number}] to [{floorplan[cell]}]')
                return selected_neighbor_cell_room # 변경된 룸 번호
            # else: not valid_change go throguh valid_change
        else:
            print(
                f'change_room_cell({room_number},{cell} ) not changed anything becuase cell_neighbors_room_numbers={cell_neighbors_room_numbers} not empty')
            return room_number


def remove_cell_from_protruding_cells_dict_list(protruding_cells_dict, key_to_modify, item_to_remove):
    if key_to_modify in protruding_cells_dict:
        if item_to_remove in protruding_cells_dict[key_to_modify]:
            protruding_cells_dict[key_to_modify].remove(item_to_remove)
            if not protruding_cells_dict[key_to_modify]:
                del protruding_cells_dict[key_to_modify]


def add_cell_to_protruding_cells_dict_list(protruding_cells_dict, key_to_modify, cell):
    if key_to_modify not in protruding_cells_dict:
        protruding_cells_dict[key_to_modify] = [cell]
    else:
        protruding_cells_dict[key_to_modify].append(cell)


# 방번호를 변경
def change_room_number_from_protruding_cells_dict_list(protruding_cells_dict, item_to_move, old_key, new_key):
    print(f'\t\t[change_room_number_from_protruding_cells_dict_list] {item_to_move} from {old_key} to {new_key}')
    if old_key in protruding_cells_dict and item_to_move in protruding_cells_dict[old_key]:
        # when (5,7) from 1 to 3
        # 1 in protruding_cells_dict and (5,7) in protruding_cells_dict
        print(f'\t\t{old_key} in {protruding_cells_dict} and {item_to_move} in {protruding_cells_dict[old_key]}')
        protruding_cells_dict[old_key].remove(item_to_move)

        # old_key의 리스트가 비어있으면 old_key 제거
        if not protruding_cells_dict[old_key]:
            del protruding_cells_dict[old_key]

        # new_key가 없으면 생성하고 추가
        if new_key not in protruding_cells_dict:
            protruding_cells_dict[new_key] = []
        protruding_cells_dict[new_key].append(item_to_move)


def update_protruding_cells_dict(floorplan, room_number, new_room_number, selected_cell, protruding_cells_dict):
    print(f'\tcurrent protruding_cells_dict...{protruding_cells_dict}')
    print(f'new_room_number={new_room_number}, selected_cell = {selected_cell}')
    current_diff_ratio = get_diff_ratio_cell(floorplan, selected_cell)
    # 방번호 변경
    if current_diff_ratio > threshold_diff_ratio:
        change_room_number_from_protruding_cells_dict_list(protruding_cells_dict, selected_cell, room_number,
                                                           new_room_number)
        print(f'\tchanged_room_number from {room_number} to {new_room_number}')
        print(f'\tresult protruding_cells_dict...{protruding_cells_dict}')
    if current_diff_ratio <= threshold_diff_ratio:
        remove_cell_from_protruding_cells_dict_list(protruding_cells_dict, room_number, selected_cell)
        print(f'\tremove_room_number from {room_number} from {selected_cell}')
        print(f'\tresult protruding_cells_dict...{protruding_cells_dict}')

    # neighbors_diff_ratio를 차례로 방문해서 값을 비교해서 있어야 하는데 없으면 protruding_cells_dict[selected_cell]에 neighbors_diff_ratio의 키값을 추가 없어야 하는데 있으면 삭제
    print(f'\tupdate_protruding_cells_dict current_diff_ratio of {selected_cell} = {current_diff_ratio}')
    neighbors_diff_ratio = get_diff_ratio_neighbors(floorplan, selected_cell)  # room_number가 바뀌지 않았음
    print(f'\tneighbors_diff_ratio of {new_room_number}:{selected_cell} = {neighbors_diff_ratio}')

    for neighbor_cell, neighbor_new_diff in neighbors_diff_ratio.items():
        neighbor_cell_room_number = floorplan[neighbor_cell]
        if neighbor_new_diff > threshold_diff_ratio:  # 셀의 변경으로 인해 네이버가 변경되었으므로 protruding_cells_dict에 추가
            add_cell_to_protruding_cells_dict_list(protruding_cells_dict, neighbor_cell_room_number, neighbor_cell)
            print(f'\tneighbor_cell {neighbor_cell} added to protruding_cells_dict => {protruding_cells_dict}')
        else:  # 해당 셀은 이동으로 인해  더이상 돌출되지 않았다. 그러므로 제거
            neighbor_room_number = floorplan[neighbor_cell]
            remove_cell_from_protruding_cells_dict_list(protruding_cells_dict, neighbor_room_number, neighbor_cell)
            print(
                f'\tdebugging...neighbor_cell {neighbor_cell} removed from protruding_cells_dict => {protruding_cells_dict}')

    diff_ratio_dict = protruding_cells_dict[selected_cell] = neighbors_diff_ratio


def exchange_protruding_cells2(floorplan, iterations=10):
    for _ in range(iterations):
        protruding_cells_dict = get_protruding_cells(floorplan)
        if not protruding_cells_dict: break
        print(f'protruding_cells_dict = {protruding_cells_dict} ')

        swap_neighbor_success = False

        # 순차적으로 하면 업데이트된 것이 반영이 안되니까 랜덤하게 합시다.

        room_number = random.choice(list(protruding_cells_dict.keys()))
        cells_in_room_number = protruding_cells_dict[room_number]
        selected_cell = random.choice(cells_in_room_number)
        print(f'For Room({room_number}):  cell {selected_cell}')
        new_room_number = change_room_cell2(floorplan, room_number, selected_cell)
        grid_print(floorplan)
        print(f'floorplan after change_room_cell {selected_cell}\n{floorplan}')
        update_protruding_cells_dict(floorplan, room_number, new_room_number, selected_cell,
                                     protruding_cells_dict)  # 해당 셀과 그 모든 이웃 셀의 diff_ratio를 다시 구해서  protruding_cells_dict를 업데이트한다.


import numpy as np


def is_boundary_cell(floorplan, cell):
    rows, cols = floorplan.shape

    # 방향: 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 셀이 유효한 범위에 있는지 확인하는 함수
    def is_valid(cell):
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    # 이웃 셀들 중 경계 바깥에 있거나 -1이 있는지 확인
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if not is_valid(neighbor) or floorplan[neighbor] == -1:
            return True

    return False

def creating_boundary_cell_array(floorplan):
    # 경계 셀 여부를 저장할 배열 초기화
    boundary_cells = np.full(floorplan.shape, False, dtype=int)

    # 모든 셀에 대해 경계 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            if floorplan[i, j] == -1:
                boundary_cells[i, j] = -1
            else:
                boundary_cells[i, j] = is_boundary_cell(floorplan, (i, j))

    # 결과 출력
    for row in boundary_cells:
        print("[", " ".join(f"{int(x):2}" if x != -1 else f"{x:2}" for x in row), "]")
    return boundary_cells


def is_boundary_cell(floorplan, cell):
    rows, cols = floorplan.shape

    # 방향: 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 셀이 유효한 범위에 있는지 확인하는 함수
    def is_valid(cell):
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    # 이웃 셀들 중 경계 바깥에 있거나 -1이 있는지 확인
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if not is_valid(neighbor) or floorplan[neighbor] == -1:
            return True

    return False

def create_boundary_cells_array(floorplan):
    # 경계 셀 여부를 저장할 배열 초기화
    boundary_cells = np.full(floorplan.shape, False, dtype=int)

    # 모든 셀에 대해 경계 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            if floorplan[i, j] == -1:
                boundary_cells[i, j] = -1
            else:
                boundary_cells[i, j] = is_boundary_cell(floorplan, (i, j))

    # 결과 출력
    for row in boundary_cells:
        print("[", " ".join(f"{int(x):2}" if x != -1 else f"{x:2}" for x in row), "]")

    return boundary_cells

def test_module(floorplan):
    np.set_printoptions(precision=1, suppress=True)
    diff_ratios = create_diff_ratio_array(floorplan)
    print(f'diff_ratios=\n{diff_ratios}')
    aligned_corners = np.zeros_like(floorplan)
    directions = ['west', 'east', 'south', 'north']
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cell = (i, j)
            if is_valid_cell(floorplan, cell):
                # for direction in directions:
                #     aligned3 = is_consecutive_neighbors_same_room(floorplan, cell, direction)
                #     print(f'{cell}:{direction} = {aligned3}')

                triple_match_directions = consecutive_neighbor_in_orientation(floorplan, cell)
                
                # print(f'triple_match_directions {cell} = {triple_match_directions}')
                aligned_corners[i, j] = is_cascading_corner(triple_match_directions)

    print(f'aligned_corners=\n{aligned_corners}')
    boundary_cells_array = creating_boundary_cell_array(floorplan)
    # boundary_cell인 경우는
    protruding_cells_array = ((diff_ratios > 0.5) | ((diff_ratios == 0.5) & (aligned_corners == 1)) | ((boundary_cells_array==True) & (diff_ratios >= .5))).astype(int)

    print(f'protruding_cells_array=\n{protruding_cells_array}')

    # for idx in np.ndindex(floorplan.shape):
    #     print(f"{idx}: {floorplan[idx]}")

    # consecutive_neighbor_in_orientation(floorplan, cell)
    # smoothed_grid = exchange_protruding_cells(floorplan, iterations=5)

def create_aligned_corners(floorplan):
    aligned_corners = np.zeros_like(floorplan)
    directions = ['west', 'east', 'south', 'north']
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cell = (i, j)
            if is_valid_cell(floorplan, cell):
                # for direction in directions:
                #     aligned3 = is_consecutive_neighbors_same_room(floorplan, cell, direction)
                #     print(f'{cell}:{direction} = {aligned3}')

                triple_match_directions = consecutive_neighbor_in_orientation(floorplan, cell)

                # print(f'triple_match_directions {cell} = {triple_match_directions}')
                aligned_corners[i, j] = is_cascading_corner(triple_match_directions)
def create_protruding_cells_array(floorplan):
    diff_ratios = create_diff_ratio_array(floorplan)
    boundary_cells_array = create_boundary_cells_array(floorplan)
    aligned_corners = create_aligned_corners(floorplan)
    protruding_cells_array = ((diff_ratios > 0.5) | ((diff_ratios == 0.5) & (aligned_corners == 1)) | ((boundary_cells_array==True) & (diff_ratios >= .5))).astype(int)
    return protruding_cells_array

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
    # 모든 셀에 대해 코너 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cell = [i,j]
            is_boundary = boundary_cells[i,j]
            cascading_cells[i, j] = is_cascading_cell(floorplan, (i, j), is_boundary)

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

########################## move saved module to backup

# todo is_boundary 추가됨. 다른 경우도 고려
def is_cascading_cell_origin(floorplan, cell, is_boundary=False, is_rooms_boundary = False):
    # if cell == (0,1):
    print(f'cell={cell}')
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
    different_neighbors=[]
    same_neighbors=[]
    neighbors_direction = {}
    same_room_direction = {}
    for idx, d in enumerate(directions):
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if is_valid_cell(floorplan, neighbor):
            if floorplan[neighbor] != room_value: # 현재 인접셀이 다른 방일때
                different_neighbors.append((neighbor, d))
            else:
                same_neighbors.append((neighbor, d))

                # 현재 인접셀이 같은 방일 경우 todo 같은 방의 인접셀이 2 개 이상일 때만 이 작업이 필요함 모든 경우에 대해서 핲 필요없음
                neighbors_neighbor_in_room = valid_neighbors_in_room(floorplan, neighbor)
                neighbors_direction[neighbor] = d
                for nn in neighbors_neighbor_in_room:
                    if not nn == cell and is_same_direction(floorplan, neighbor, nn, d) : # 이웃과 같은 방에 있는 셀 중 현재셀과 같은 방향의 셀이 있는가
                        same_room_direction[nn] = d
                # 모든 neighbors neigbors에 대해 for 문 끝난 후 same_room_direction을 보자.
    print(f'cell={cell}, same_room_direction = {same_room_direction}')
    continuous_same_direction = [item[1] for item in same_room_direction.items()]
    if is_cell_room_corner(floorplan,cell):
        pass
    # 연속 같은 방향으로 같은 방이면 cascading_cell이 아니다.
    if is_rooms_boundary and len(continuous_same_direction) >= 2 :
        return False
      # 인접 셀들의 방향이 시계방향 또는 반시계방향으로 연속되어 있는지 확인
#    if len(different_neighbors) >= 2: # 다른 게 두 개 이상 되면
    if len(different_neighbors) >= 2: # 다른 게 두 개 이상 되면
        # boundary가 아니면서, 네이버가 둘 인 경우만 다음 수행
        for i in range(len(different_neighbors)-1):
            d1 = directions.index(different_neighbors[i%len(different_neighbors)][1])
            d2 = directions.index(different_neighbors[i+1%len(different_neighbors)][1])

            if(d1 + 1) % 4 == d2 or (d1 -1) % 4 == d2:
                return True
#        diff_dir1 = direction_names.index(different_neighbors[0][1])
#        diff_dir2 = direction_names.index(different_neighbors[1][1])
#        if (diff_dir1 + 1) % 4 == diff_dir2 or (diff_dir1 - 1) % 4 == diff_dir2:
#            return True
        return False
