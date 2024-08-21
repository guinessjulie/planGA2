import numpy as np
from GridDrawer import GridDrawer
import trivial_utils
import random
import itertools
import numpy as np
from config_reader import read_config_int
from cells_utils import is_valid_cell
# working version as of 2024-07-19

def grid_to_screen_image(sample_grid, no=0, format='png', prefix='test', text = None):
    filename, current_step = trivial_utils.create_filename_in_order(format, prefix, no)
    GridDrawer.color_cells_by_value(sample_grid, filename=filename, text=text)

# for debugging information
def grid_print_as_int(arrays2d):
    for row in arrays2d:
        print("[", " ".join(f"{int(x):2}" for x in row), "]")


# 해당 셀의 4방향 인접셀이 현재 방에 위치하는지 다른 방에 위치하는지, 혹은 invalid 셀인지를 판단함
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



def get_same_room_cells(floorplan, cell):
    return

# cascading_cell인지를 판단
def is_cascading_cell_simple(floorplan, cell): # todo change name
    current_room = floorplan[cell]
    invalid_neighbors, same_room_neighbors, diff_room_neighbors = count_neighbors_dirs(floorplan,cell)
    num_diff_invalid = len(invalid_neighbors) + len(diff_room_neighbors)

    same_room_cells = np.argwhere(floorplan == floorplan[cell])
    if len(same_room_cells) <= 2: # 방의 크기가 2유닛 이하면 cascading이 아니다.
        return False

    if is_valid_cell(floorplan, cell) and num_diff_invalid >=3:
        return True
    else:
        return False

def count_cascading_cells(floorplan):
    rows, cols = floorplan.shape
    cascading_cells_count = 0
    cascading_cells_list = []

    for i in range(rows):
        for j in range(cols):
            cell = (i, j)
            if is_cascading_cell_simple(floorplan, cell):
                cascading_cells_count += 1
                cascading_cells_list.append(cell)

    return cascading_cells_count, cascading_cells_list

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


def count_cascading_cells_neighbors(floorplan, cell, old_room, new_room_number):
    # 변경전 당연히 cascading
    neighbors = all_active_neighbors(cell, floorplan)
    cascading_neighbors = [n for n in neighbors if is_cascading_cell_simple(floorplan, n)]
    old_cascading_count = len(cascading_neighbors) + 1

    floorplan[cell] = new_room_number # 변경 후
    new_cell_cascading = is_cascading_cell_simple(floorplan,cell)
    new_cascading_neighbors = [n for n in neighbors if is_cascading_cell_simple(floorplan, n)]
    new_cascading_count = len(new_cascading_neighbors) + 1 if new_cell_cascading else len(new_cascading_neighbors)

    # revert
    floorplan[cell] = old_room
    return old_cascading_count - new_cascading_count

def update_cascading(floorplan,cell,neighbors, cascading_cells_list, cascading_cells):
    if not is_cascading_cell_simple(floorplan, cell):
        if cell in cascading_cells_list:
            cascading_cells_list.remove(cell)
            cascading_cells[cell] = 0

    for neighbor in neighbors:
        if is_cascading_cell_simple(floorplan, neighbor):
            if neighbor not in cascading_cells_list:
                cascading_cells_list.append(neighbor)
                cascading_cells[neighbor] = 1
        else:
            if neighbor in cascading_cells_list:
                cascading_cells_list.remove(neighbor)
                cascading_cells[neighbor] = 0

# todo separate floorplan and resulting_floorplan
def exchange_protruding_cells_debug(floorplan_origin, iteration=1, display=False, save=True):
    floorplan = floorplan_origin.copy()
    cascading_cells = create_cascading_cells(floorplan)
    changed_cell_history={}
    if not np.any(cascading_cells == 1):
        return

    cascading_cells_list = list(map(tuple, np.argwhere(cascading_cells == 1)))
    candidate_cascading_cells_list = cascading_cells_list.copy()
    save_path = trivial_utils.create_folder_by_datetime()
    i = 0
    while len(cascading_cells_list) > 0:
        i += 1
        cell = random.choice(cascading_cells_list) #
        current_room_number = floorplan[cell]
        neighbors = all_active_neighbors(cell, floorplan) # 선택된 셀의 neighbors들 중에서 선택
        neighbors_room_to_exchange = list(set([ floorplan[n] for n in neighbors if floorplan[n] != current_room_number])) # 같은 방 제외
        # filtering1: 네이버 room들 중에서 다른 방 선택
        candidate_rooms = [room for room in neighbors_room_to_exchange]
        print(f'Step [{i}] Cell [{cell}]: from {candidate_cascading_cells_list} before filtering changed history: candidate_rooms= {candidate_rooms}')
        if cell in changed_cell_history: # filtering more
            candidate_rooms = [room for room in candidate_rooms if room not in changed_cell_history[cell]]
        print(f'\tCell [{cell}]: candidate_room after filtering changed history: {candidate_rooms}')

        # 교환하자.
        # candidate_rooms가 success하거나 없어질 때까지 새 방을 고른다. todo candidate_rooms는 never 몽땅 없어지지 않는다. 그러므로 while에서 무한루프를 돈다. 무엇이 stop시킬 수 있는지를 고민해보자. 아마도 changed_cell_history에서 더이상 가져올 게 없을 때를 보는 게 맞는 거 같다.
        w = 0
        if len(candidate_rooms) == 0:
            # candidate_cascading_cells_list.remove(cell)
            cascading_cells_list.remove(cell) ## todo 3. 0731 to see candidate_cascading_cell_list => cascading_cell_list
            continue # []빈 candidate_room을 가졌으므로 candidatae_cascading_cell_list도 업데이트해야 됨
        elif len(candidate_rooms) > 1:
            best_room = candidate_rooms[0]
            gap = count_cascading_cells_neighbors(floorplan, cell, current_room_number, best_room)

            for it in range(len(candidate_rooms)-1):
                gap2 = count_cascading_cells_neighbors(floorplan, cell, current_room_number, candidate_rooms[it+1])
                if gap < gap2:
                    best_room = candidate_rooms[it+1]
        elif len(candidate_rooms) == 1:
            best_room = candidate_rooms[0]


        if is_valid_change(floorplan, cell, best_room, changed_cell_history): # 성공했다면
            update_cascading(floorplan,cell, neighbors, cascading_cells_list, cascading_cells)
            # candidate_cascading_cells_list.remove(cell)
            text = (f'Step {i}  {cell}: {current_room_number} => {best_room}\nhistory={changed_cell_history}\n'
                    f'cascading({len(cascading_cells_list)}) = {cascading_cells_list}')
        else:
            candidate_rooms.remove(best_room)  # valid 하지 않으므로 지우고 다른 걸 선택한다.
            # todo to avoid infinite loop 현재 안되는 것을 history에 추가
            if cell not in changed_cell_history:
                changed_cell_history[cell] = [best_room]
            else:
                changed_cell_history[cell].append(best_room)

            text = (f'Step {i} Failed to change {cell} from {current_room_number} to {best_room}\n'
                    f'becuase ( {cell} to {best_room}) is not valid change')

        grid_print_as_int(cascading_cells)

        full_path = trivial_utils.create_file_name_in_path(path = save_path,  prefix = 'Simplifying', postfix_number=i )
        num_rooms = read_config_int('constraints.ini', 'Metrics', 'num_rooms')

        GridDrawer.color_cells_by_value(floorplan, filename=full_path, text=text, display=display, save=save, num_rooms = num_rooms)
        # grid_to_screen_image(floorplan, no=i, format='png', prefix = filename , text = text)
    return floorplan

def exchange_protruding_cells(floorplan_origin, iteration=1, display=False, save=True):
    floorplan = floorplan_origin.copy()
    cascading_cells = create_cascading_cells(floorplan)
    changed_cell_history={}
    if not np.any(cascading_cells == 1):
        return

    cascading_cells_list = list(map(tuple, np.argwhere(cascading_cells == 1)))
    candidate_cascading_cells_list = cascading_cells_list.copy()
    save_path = trivial_utils.create_folder_by_datetime()
    i = 0
    while len(cascading_cells_list) > 0:
        i += 1
        cell = random.choice(cascading_cells_list) #
        current_room_number = floorplan[cell]
        neighbors = all_active_neighbors(cell, floorplan) # 선택된 셀의 neighbors들 중에서 선택
        neighbors_room_to_exchange = list(set([ floorplan[n] for n in neighbors if floorplan[n] != current_room_number])) # 같은 방 제외
        # filtering1: 네이버 room들 중에서 다른 방 선택
        candidate_rooms = [room for room in neighbors_room_to_exchange]
        if cell in changed_cell_history: # filtering more
            candidate_rooms = [room for room in candidate_rooms if room not in changed_cell_history[cell]]

        # 교환하자.
        # candidate_rooms가 success하거나 없어질 때까지 새 방을 고른다. todo candidate_rooms는 never 몽땅 없어지지 않는다. 그러므로 while에서 무한루프를 돈다. 무엇이 stop시킬 수 있는지를 고민해보자. 아마도 changed_cell_history에서 더이상 가져올 게 없을 때를 보는 게 맞는 거 같다.
        w = 0
        if len(candidate_rooms) == 0:
            # candidate_cascading_cells_list.remove(cell)
            cascading_cells_list.remove(cell) ## todo 3. 0731 to see candidate_cascading_cell_list => cascading_cell_list
            continue # []빈 candidate_room을 가졌으므로 candidatae_cascading_cell_list도 업데이트해야 됨
        elif len(candidate_rooms) > 1:
            best_room = candidate_rooms[0]
            gap = count_cascading_cells_neighbors(floorplan, cell, current_room_number, best_room)

            for it in range(len(candidate_rooms)-1):
                gap2 = count_cascading_cells_neighbors(floorplan, cell, current_room_number, candidate_rooms[it+1])
                if gap < gap2:
                    best_room = candidate_rooms[it+1]
        elif len(candidate_rooms) == 1:
            best_room = candidate_rooms[0]


        if is_valid_change(floorplan, cell, best_room, changed_cell_history): # 성공했다면
            update_cascading(floorplan,cell, neighbors, cascading_cells_list, cascading_cells)
            # info: uncomment to save processing steps png file
            # text = (f'Step {i}  {cell}: {current_room_number} => {best_room}\nhistory={changed_cell_history}\n'
            #    f'cascading({len(cascading_cells_list)}) = {cascading_cells_list}')

        else:
            candidate_rooms.remove(best_room)  # valid 하지 않으므로 지우고 다른 걸 선택한다.
            # todo to avoid infinite loop 현재 안되는 것을 history에 추가
            if cell not in changed_cell_history:
                changed_cell_history[cell] = [best_room]
            else:
                changed_cell_history[cell].append(best_room)
            # info: uncomment following three lines to save processing steps png file
            # text = (f'Step {i} Failed to change {cell} from {current_room_number} to {best_room}\n'
            #        f'becuase ( {cell} to {best_room}) is not valid change')

        # info: uncomment following four lines to save processing steps png file
        # grid_print_as_int(cascading_cells)
        # full_path = trivial_utils.create_file_name_in_path(path = save_path,  prefix = 'Simplifying', postfix_number=i )
        # num_rooms = read_config_int('constraints.ini', 'Metrics', 'num_rooms')
        # GridDrawer.color_cells_by_value(floorplan, filename=full_path, text=text, display=display, save=save, num_rooms = num_rooms)
        # # grid_to_screen_image(floorplan, no=i, format='png', prefix = filename , text = text)
    return floorplan

def create_cascading_cells(floorplan):
    cascading_cells = np.zeros_like(floorplan, dtype=bool)
    # 모든 셀에 대해 뾰족 셀 여부를 판단
    for i in range(floorplan.shape[0]):
        for j in range(floorplan.shape[1]):
            cascading_cells[i][j] = is_cascading_cell_simple(floorplan, (i, j))
    # print(f'number of cascading_cells = {np.count_nonzero(cascading_cells==1)}')
    # 결과 출력 (칸을 맞춰서) todo for debug uncomment grid_print_as_int
    # grid_print_as_int(cascading_cells)

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
    grid_to_screen_image(floorplan, prefix = 'Initial_floorplan', no=0, format='png', text= ' ')
    # test_module(floorplan)
    exchange_protruding_cells(floorplan)
    # grid_print(floorplan)

if __name__ == '__main__':
    # test_main()
    pass