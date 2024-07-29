import numpy as np
from GridDrawer import GridDrawer
import trivial_utils
import random
import itertools
import numpy as np

# working version as of 2024-07-19

def get_position(arr2d, condition):
    np.argwhere(arr2d == condition)
def grid_to_image(sample_grid, no=0, format='png', prefix='test', text = None):
    filename, current_step = trivial_utils.create_filename_in_order(format, prefix, no)
    GridDrawer.color_cells_by_value(sample_grid, filename, text)

def grid_to_screen_image(sample_grid, no=0, format='png', prefix='test', text = None):
    GridDrawer.color_cells_by_value(sample_grid, filename='test_text', text=text)

def grid_print_as_int(arrays2d):
    for row in arrays2d:
        print("[", " ".join(f"{int(x):2}" for x in row), "]")

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
        #if len(diagonals) > 1:
        #    print(f'diagonals={len(diagonals)} > 1')
        return True
    else :
        return False


def get_two_neighbors_direction(added_dir_pair):
    diag_d = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    if added_dir_pair in diag_d:
        return 'orthogonal'
    elif added_dir_pair == (0,0):
        return 'parallel'
    # todo 방향도 가지고 있어야 됨
# neighbor의 위치와 방향을 쌍으로 가지고 있는 리스트를 리턴
def count_neighbors_dirs(floorplan, cell):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    active_neighbors = all_active_neighbors(cell, floorplan)
    invalids, same_room_neighbors, diff_room_neighbors = [],[],[]
    for d in directions:
        neighbor = tuple(np.add(cell, d))
        if not is_valid_cell(floorplan, neighbor):
            invalids.append((neighbor, d))
        elif floorplan[cell] == floorplan[neighbor]: # same room
            same_room_neighbors.append((neighbor, d))
        elif floorplan[cell] != floorplan[neighbor]: # diff_room
            diff_room_neighbors.append((neighbor, d))
    return invalids, same_room_neighbors, diff_room_neighbors
def is_cascading_cell(floorplan, cell, is_boundary=False, is_rooms_boundary = False):

    rows, cols = floorplan.shape
    room_value = floorplan[cell]

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # 셀이 유효한 범위에 있는지 확인하는 함수
    if not is_valid_cell(floorplan, cell): return False

    # todo 이건 두 칸 떨어진 셀도 같거나 다르거나 한 걸 구하는 것임
    invalid_neighbors, neighbor_cell_dir_diff_room, neighbor_cell_dir_same_room = count_two_continuous_directional_neighbors(floorplan, cell, directions, room_value)
    # todo  위의 변수들은 두칸 넘어 셀들에 대한 값임, 아래는 바로 네이버에 대한 값임
    invalid_neighbors, same_room_neighbors, diff_room_neighbors = count_neighbors_dirs(floorplan, cell)

    if is_boundary_cell(floorplan, cell) :
        if len(invalid_neighbors) + len(diff_room_neighbors) >= 3 :
            # print(f'{cell} True because boundary and len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')
            return True
        else: # boundary_cell인데 이1셀 이하로만 다르다면 definitely 그냥 boundary todo 모든 boundary에 대해서 다 체크해보자.
            # print(f'{cell} False because boundary and NOT len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')
            return False
    else: # not boundary_cell # no invalid_neighbors
        if len(diff_room_neighbors) >= 3 : # boundary cell이 아니더라도 세 개 이상이 다르면 무조건 cascading
            return True

        if len(diff_room_neighbors) == 2 and len(same_room_neighbors) == 2: #  두개 두개씩 같을 때는 다른 방에서 대각선을 본다.
            diag_d = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
            for diff_cell, diff_dr in diff_room_neighbors:
                for same_cell, same_dr in same_room_neighbors:
                    added_dir_pair = tuple(np.add(diff_dr, same_dr))
                    if get_two_neighbors_direction(added_dir_pair) == 'orthogonal':
                        diag_cell = tuple(np.add(cell, added_dir_pair))
                        if (floorplan[diag_cell] == floorplan[cell]):
                            return True
        if len(same_room_neighbors) >=3:
            return False
        #  todo 아래 블럭 복잡해서 위에다가 다시 짠다
#        if len(neighbor_cell_dir_diff_room) == 2:
#            if len(neighbor_cell_dir_same_room) != 2:
#                print(f'{cell} diff=2, same!= 2 how this can happens')
#            else: #  다른 방 같은 방 각각 2개씩일 때
#                # 2 step away
#                diff2step = []
#                diff_2step_diag_dir  = tuple(np.sum([d for _, d in neighbor_cell_dir_diff_room], axis=0)) # 다른 색깔의 인접한 두 셀의 방향이 orthorgonal L 자 shape인지 확인
#
#                for diff_nei, d in neighbor_cell_dir_diff_room:
#                    diff2 = tuple(np.add(diff_nei, d))
#                    if is_valid_cell(floorplan, diff2) and floorplan[diff2] != floorplan[cell]:
#                        diff2step.append(diff2)
#                # for 문이 끝나면 2 개의 다른 방의 셀들이 들어있다.
#                sames2step  = [tuple(np.add(same_nei, d)) for same_nei, d in neighbor_cell_dir_same_room if is_valid_cell(floorplan, (same_nei[0]+d[0], same_nei[1]+d[1]))]
#                if len(sames2step) < 2:
#                    print(f'lets see what kind of cell makes this happen {cell}')
#                same_2step_diag_dir = tuple(np.sum([d for _, d in neighbor_cell_dir_same_room], axis=0)) # 대각선 방향
#                is_continous_same_dir = same_2step_diag_dir in diag_d # 두개의 인접셀이 같은 색이고, 이것이 L 방향 방향이다.
#                diag_same_cell = tuple(np.add(cell, same_2step_diag_dir)) # at least twoxtwo block
#                if floorplan[diag_same_cell] == floorplan[cell]:
#                    return False
    if is_two_by_two_block_same_room(floorplan, cell, directions):
        # print(f'{cell} False becuase two_by_two_block_same_room ')
        return False
def is_cascading_cell_save(floorplan, cell, is_boundary=False, is_rooms_boundary = False):

    rows, cols = floorplan.shape
    room_value = floorplan[cell]

    # 방향: 북, 동, 남, 서
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # 셀이 유효한 범위에 있는지 확인하는 함수
    if not is_valid_cell(floorplan, cell): return False


    invalid_neighbors, neighbor_cell_dir_diff_room, neighbor_cell_dir_same_room = count_two_continuous_directional_neighbors(floorplan, cell, directions, room_value)
    if is_boundary_cell(floorplan, cell) :
        if len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) >= 3 :
            # print(f'{cell} True because boundary and len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')
            return True
        else:
            # print(f'{cell} False because boundary and NOT len(invalid_neighbors) + len(neighbor_cell_dir_diff_room) {len(invalid_neighbors) + len(neighbor_cell_dir_diff_room)} >= 3')
            return False
    else: # not boundary_cell # no invalid_neighbors
        if len(neighbor_cell_dir_diff_room) >= 3 :
            return True
        if len(neighbor_cell_dir_diff_room) == 2:
            if len(neighbor_cell_dir_same_room) != 2:
                print(f'{cell} diff=2, same!= 2 how this can happens')
            else: #  다른 방 같은 방 각각 2개씩일 때 todo  아래 두 줄 제거, same_room_...와 neigghbor_cell...이 같다.
                same_room_neighbors_cells = [same for same in neighbor_cell_dir_same_room]
                diff_room_neighbors_cells = [diff for diff in neighbor_cell_dir_diff_room]

                for diff_nei, d in diff_room_neighbors_cells:
                    for d2 in directions:
                        # 옆방에서 대각선 방향으로 자신과 같은 룸이 있다면
                        # todo 로직이 잘못된 듯 하여 변경해본다. floorplan[cell]과 비교하면 안되고 diff_nei 와 비교해야 됨
                        # diag_neighbor_me = [(diff_nei[0]+d2[0], diff_nei[1]+d2[1]) for d2 in directions \
                        #                  if (diff_nei[0]+d2[0], diff_nei[1]+d2[1]) != cell \
                        #                  and is_valid_cell(floorplan, (diff_nei[0]+d2[0], diff_nei[1]+d2[1])) \
                        #                  and floorplan[(diff_nei[0]+d2[0], diff_nei[1]+d2[1])] == floorplan[cell]]
                        # For readibility
                        diff_double_nei =  (diff_nei[0]+d2[0], diff_nei[1]+d2[1]) # cell에서 2칸 떨어져있음
                        diag_neighbor_me = [diff_double_nei for d2 in directions \
                                         if diff_double_nei != cell \
                                         and is_valid_cell(floorplan, diff_double_nei) \
                                         and floorplan[diff_double_nei] == floorplan[diff_nei]]
                        if len(diag_neighbor_me) > 0 :
                            # print(f'{cell} True ... has diagonal corner at {diag_neighbor_me}')
                            return True

                # 같은 방향으로 연속 같은 방일 때
                two_step_away_same_room = [((nei[0] + di[0], nei[1] + di[1]), (nei[0], nei[1]), (di[0], di[1])) for
                                           nei, di in same_room_neighbors_cells if
                                           nei[0] + di[0] < rows and nei[1] + di[1] < cols and floorplan[cell] ==
                                           floorplan[nei[0] + di[0], nei[1] + di[1]]]

                if len(two_step_away_same_room) >= 2:
                    dir_same_pairs = [di for double_nei, nei, di in two_step_away_same_room]
                    # todo 이중 인접셀이 같은 경우가 2 개의 경우만 생각함. 3 개 이상인 경우는 당연히 False 일 것 같은에?
                    diagonal = [(x1 + x2, y1 + y2) for (x1, y1), (x2, y2) in itertools.combinations(dir_same_pairs, 2)
                                if abs(x1 + x2) == 1 and abs(y1 + y2) == 1]
                    is_orthogonal = len(diagonal) > 0
                    if is_orthogonal:
                        diagonal_cell = (cell[0] + diagonal[0][0], cell[1] + diagonal[0][1])
                        diagonal_room = floorplan[diagonal_cell]
                        if diagonal_room == room_value:
                            # print(f'{cell} False because {diagonal_cell} is the same room ? nop')
                            return False
                        else:
                            # print(f'{cell} True becuase {diagonal_cell} is the different room')
                            return True


    if len(neighbor_cell_dir_diff_room) + len(invalid_neighbors) >= 3: # 다른 게 두 개 이상 되면
        print(f'{cell} True because diff=>3')
        return True
    two_by_two = is_two_by_two_block_same_room(floorplan, cell, directions)
    if two_by_two:
        # print(f'{cell} False becuase two_by_two_block_same_room ')
        return False


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

#  floorplan 내의 유효 이웃을 모두 리턴
# todo duplcated (all_active_neigghbors로 통일)
def all_neighbors(cell, floorplan):
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for d in directions4:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < floorplan.shape[0] and 0 <= neighbor[1] < floorplan.shape[1] and floorplan[
            neighbor] > 0:
            neighbors.append(neighbor)
    return neighbors


def is_valid_cell(floorplan, cell):
    is_valid = 0 <= cell[0] < floorplan.shape[0] and 0 <= cell[1] < floorplan.shape[1] and floorplan[cell] > 0
    # print(f'{cell} is_valid={is_valid}')
    return is_valid


def get_cell_neighbors_room_number(floorplan, cell):
    neighbors = all_active_neighbors(cell, floorplan)
    cell_neighbors_room_number = []
    for neighbor in neighbors:
        if is_valid_cell(floorplan, cell) and floorplan[cell] != floorplan[neighbor]:
            cell_neighbors_room_number.append(floorplan[neighbor])
    return cell_neighbors_room_number

def change_room_cell(floorplan, cell, cascading_neighbors):

    new_room = random.choice(cascading_neighbors)
    new_room_number = floorplan[new_room]
    original_room_number = floorplan[cell]

    floorplan[cell] = new_room_number

    # 바꾸어도 valid한가
    neighbors = (all_active_neighbors(cell, floorplan))
    valid = any(floorplan[n[0], n[1]] == floorplan[cell[0], cell[1]] for n in neighbors)
    for nei in neighbors:
        neighbor_neighbors = all_active_neighbors(nei, floorplan)
        valid = valid and any(floorplan[nn] == floorplan[nei] for nn in neighbor_neighbors) # 해당 네이버가 고립되지 않아야 하므로 하나라도 같은 네이버가 있어야 한다.
        if not valid:
            floorplan[cell] = original_room_number
            return None
    if valid:
        return original_room_number
    else:
        print(f'exchange{cell} from {original_room_number} to {floorplan[cell]} is not valid how can this happen!!!!')
        return None
    #  바뀌기 전의 cell 방을 기억해야 하므로


def is_boundary_cell(floorplan, cell):
    rows, cols = floorplan.shape
    if floorplan[cell] == -1: return False
    if cell[0] == 0 or cell[1] == 0 or cell[0] == (rows -1)  or cell[1] == (cols -1): return True
    # 방향: 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    # directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 이웃 셀들 중 경계 바깥에 있거나 -1이 있는지 확인
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if not is_valid_cell(floorplan, neighbor) or floorplan[neighbor] == -1:
            return True

    return False

def is_isolated_cell(floorplan, cell):
    neighbors = all_neighbors(cell, floorplan)
    outside_cells = [nei for nei in neighbors if not is_valid_cell(floorplan, cell)]
    diff_neighbors = [nei for nei in  neighbors if floorplan[nei] != floorplan[cell]]
    if len(outside_cells) + len(diff_neighbors) >= 4:
        return True
    else:
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
    # print('boundary_cells')
    # grid_print_as_int(boundary_cells)
    # for row in boundary_cells:
    #     print("[", " ".join(f"{int(x):2}" if x != -1 else f"{x:2}" for x in row), "]")

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
    print(f'number of cascading cells = {np.count_nonzero(cascading_cells_array==1)}')
    return cascading_cells_array

# todo (2,13), (3,13) 이 cascading인 이유 = > 3개 이상이 다르니까
# def exchange_protruding_cells_save2(floorplan, iteration=1):
#     cascading_cells = create_cascading_cells(floorplan)
#     boundary_cells = create_room_boundary_cells_array(floorplan)
#     changed_cell_pairs = {}
#     changed_cell_history={}
#     if not np.any(cascading_cells == 1):
#         return
#
#     new_candid = np.argwhere(cascading_cells == 1)
#     i = 0
#     # todo 무한루프 해결
#
#     while len(new_candid) > 0:
#         i+=1
#         if not np.any(cascading_cells == 1):
#             return
#         # candidates = np.argwhere(cascading_cells == 1)
#         # np.argwhere는 리스트의 리스트를 리턴하기 때문에 map을 이용해서 tuple로 바꾸어준다.
#         # 1로 된 셀의 위치를 찾기
#         cascading_candidates = list(map(tuple,np.argwhere(cascading_cells == 1)))
#         # 이미 바꾼 것은 제거 todo 한 번 바뀌었다고 해서 계속 못바꿀 수는 없지 않나? 색깔이 제한되어 있으므로 그럴 수도 있지만 네 개의 이웃이 있으니 네 개의 색깔이 가능해
#         # cascading_candidates  = [tuple(pos) for pos in cascading_candidates if tuple(pos) not in set(changed_cell_pairs.keys())]
#         new_candid = {}
#         new_candid2={}
#         for cell in cascading_candidates:
#             neighbors_to_exchange = all_active_neighbors(cell, floorplan)
#
#             # neighbor가 cascading이면서 같은 방이 아닐때
#             cascading_nei = [nei for nei in neighbors_to_exchange if cascading_candidates
#                              and floorplan[cell] != floorplan[nei] and nei not in changed_cell_history]
#             new_candid[cell] = [(k, v) for k, v in cascading_nei if v]
#
#             #            # filter more
# #            # 교환을 위해 선택된 셀이 선택 셀과 이미 바뀐 전적이 있을 때를 필터링
# #            if cell in changed_cell_history: # 바뀐적이 있어
# #                # todo for debugging
# #                print(f'changed_cell_history:{changed_cell_history}')
# #                # cascading_neighbor 중에서 history에 없는 걸 찾자
# #                if cascading_nei : # cascading_nei 중에서 바뀐적이 없는 것을 선택
# #                    cascading_nei = [n for n in cascading_nei if floorplan[n] not in changed_cell_history[cell]] #!! respect me neighbor 중에서 과거의 색이 현재의 색이었던 것을 거른다. 중복적으로 바뀌는 걸 ㅂ피하기 위해서
# #                    print(f'current cell{cell}: cascading_nei {cascading_nei} not in history {cascading_nei}')
# #            # cascading_nei가 있을 때만 new_candid에 추가한다.
# #            if len(cascading_nei) <= 0:
# #                continue
#             new_candid[cell] = cascading_nei
#         new_candid = {k:v for k,v in new_candid.items() if len(v) > 0 }
#         print(f'Step[{i}] new_candid{new_candid}')
#         if len(new_candid) <= 0 :
#             print(f'new_candid < =0 returns')
#             return
#         # new_candid의 cascading_neighbor를 먼저 구해보자.
#         elif len(new_candid) == 1: # 하나밖에 없으면 선택의 여지가 없으므로 그 값이 그값이 됨
#             cell, new_cells = next(iter(new_candid.items()))
#             print(f'cell = {cell},new_cells= {new_cells}')
#             if cell != new_cells :
#                 print(f'cell{cell} = new_cells{new_cells} 원치 않은 일이 일어났다. filter 모듈을 다시 점검해') # 무슨 헛소리냐, cell과 new_cell은 다르다.
#         else:
#             selected_cell, new_cells = random.choice(list(new_candid.items()))
#             print(f'selected_cell = {selected_cell},new_cells= {new_cells}')
#             # for debugging to see cascading of (4,5) when (4,5) becomes yellow  should be not cascading. todo to get rid of it after test if this cell is cascading
#             # cell, new_cells = (4,5), new_candid[(4,5)]
#         ####
#         old_room_number = change_room_cell(floorplan, selected_cell, new_cells)
#         if old_room_number != None: # 바뀌었다면 history에 기록
#             if selected_cell not in changed_cell_history:
#                 changed_cell_history[selected_cell] = [old_room_number]
#             else:
#                 changed_cell_history[selected_cell].append(old_room_number)
#             # todo isolated cell 처리
#             # todo is_cascading_cell에서 is_boundary_cell을 체크하므로 is_boundary_cell array를 update해야됨
#             # update_boundary_cells_array(floorplan, cascading_cells, cell)
#             update_cascading_cells_array(floorplan, cascading_cells, selected_cell)
#
#             grid_print_as_int(cascading_cells)
#             grid_to_image(floorplan)


def is_valid_change(floorplan, cell, new_room, changed_cell_history):
    old_room  = floorplan[cell]
    floorplan[cell] = new_room

    neighbors = all_active_neighbors(cell, floorplan)
    # 교환으로 인해 인접셀들도 영향을 받았다. 인접셀들도 모두 유효한지 보자.
    valid = any(floorplan[n] == floorplan[cell] for n in neighbors)  # 바꾼 후의 현재 셀의 인접셀에 최소 1개 이상 현재 셀의 바뀐 방과 일치할 때만 valid 함
    for nei in neighbors:
        neighbor_neighbors = all_active_neighbors(nei, floorplan)
        valid = valid and any(floorplan[nn] == floorplan[nei] for nn in
                                      neighbor_neighbors)

    if valid:
        if cell not in changed_cell_history:
            changed_cell_history[cell] = [old_room]
        else:
            changed_cell_history[cell].append(old_room)
        return True
    else:
        floorplan[cell] = old_room
        return False

def exchange_protruding_cells(floorplan, iteration=1):
    cascading_cells = create_cascading_cells(floorplan)
    changed_cell_history={}
    if not np.any(cascading_cells == 1):
        return

    cascading_cells_list = list(map(tuple, np.argwhere(cascading_cells == 1)))
    i = 0
    # todo 무한루프 해결

    while len(cascading_cells_list) > 0:
        i += 1
        cell = random.choice(cascading_cells_list)
        current_room_number = floorplan[cell]
        neighbors = all_active_neighbors(cell, floorplan) # 선택된 셀의 neighbors들 중에서 선택
        neighbors_room_to_exchange = list(set([ floorplan[n] for n in neighbors if floorplan[n] != current_room_number])) # 같은 방 제외
        # filtering1: 네이버 room들 중에서 다른 방 선택
        candidate_rooms = [room for room in neighbors_room_to_exchange]
        print(f'Step [{i} ]Cell [{cell}]: candidate_room before filtering changed history: {candidate_rooms}')
        if cell in changed_cell_history: # filtering more
            candidate_rooms = [room for room in candidate_rooms if room not in changed_cell_history[cell]]
        print(f'\tCell [{cell}]: candidate_room after filtering changed history: {candidate_rooms}')

        # 교환하자.
        # candidate_rooms가 success하거나 없어질 때까지 새 방을 고른다. todo candidate_rooms는 never 몽땅 없어지지 않는다. 그러므로 while에서 무한루프를 돈다. 무엇이 stop시킬 수 있는지를 고민해보자. 아마도 changed_cell_history에서 더이상 가져올 게 없을 때를 보는 게 맞는 거 같다.
        w = 0
        while len(candidate_rooms) > 0 :
            w+=1
            print(f'[debugging] while loop iteration {w}')
            new_room_number = random.choice(candidate_rooms)
            if is_valid_change(floorplan, cell, new_room_number, changed_cell_history): # 교환했으므로  while문을 빠져나온다.
                update_cascading_cells_array(floorplan, cascading_cells, cell)
                cascading_cells_list = list(map(tuple, np.argwhere(cascading_cells == 1)))
                text = f'{cell}: {current_room_number} => {new_room_number}\nhistory={changed_cell_history}\ncascading={cascading_cells_list}'
                print(f'Step {i} exchange {cell} to Room{new_room_number}...\ncascading_cell_list={cascading_cells_list}\nchanged_cell_history({len(changed_cell_history)}) = {changed_cell_history}')
                break
            else:
                candidate_rooms.remove(new_room_number) # valid 하지 않으므로 지우고 다른 걸 선택한다.
                text = 'no valid exchange rrom'
        grid_print_as_int(cascading_cells)
        grid_to_screen_image(floorplan,text = text)
        if i > 3:
            break

def create_cascading_cells(floorplan):
    # 코너 셀 여부를 저장할 배열 초기화
    cascading_cells = np.zeros_like(floorplan, dtype=bool)
    boundary_cells = create_boundary_cells_array(floorplan)
    rooms_boundary_cells = create_room_boundary_cells_array(floorplan)
    # print(f'cascading_cells = \n{cascading_cells}')
    # print(f'boundary_cells = \n{boundary_cells}')
    # print(f'rooms_boundary_cells = \n{rooms_boundary_cells}')
    # print(f'room_cell_corners = \n{room_cell_corners}')
    # 모든 셀에 대해 코너 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            is_boundary = boundary_cells[i, j]
            is_rooms_boundary = rooms_boundary_cells[i, j]
            cascading_cells[i][j] = is_cascading_cell(floorplan, (i, j), is_boundary, is_rooms_boundary)
    print(f'number of cascading_cells = {np.count_nonzero(cascading_cells==1)}')
    # 결과 출력 (칸을 맞춰서)
    grid_print_as_int(cascading_cells)

    return cascading_cells


def test_main():
    # 주어진 floorplan 배열
    floorplan = np.array( [
    [2, 2, 2, 2, 4, 4, 1, 1, 1, 1, -1, -1, -1, -1],
    [2, 2, 2, 4, 4, 4, 4, 1, 1, 1, -1, -1, -1, -1],
    [2, 2, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, -1, -1],
    [2, 2, 4, 4, 5, 5, 3, 5, 5, 3, 3, 3, -1, -1],
    [-1, -1, -1, -1, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1]
    ])
    grid_to_image(floorplan)
    # test_module(floorplan)
    exchange_protruding_cells(floorplan, 100)
    # grid_print(floorplan)

if __name__ == '__main__':
    test_main()