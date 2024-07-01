import networkx as nx
import matplotlib.pyplot as plt
from constants import DIRECTIONS, test_grid


def build_graph(grid):
    """
    각 색상 그룹을 노드로 표현
    두 색상 그룹이 인접한 경우, 에지 생성
    :param grid: two-dimensional array. each element represent colors(plan)
    :return: dictionary of set
    """

    nrows, ncols = len(grid), len(grid[0])
    graph = {}


    for r in range(nrows):
        for c in range(ncols):
            color = grid[r][c]
            if color not in graph:
                graph[color] = set()

            # 현재 셀의 색상과 인접한 다른 색상들 찾기
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    neighbor_color = grid[nr][nc]
                    if neighbor_color != color:
                        graph[color].add(neighbor_color)

    return graph

def build_weighted_graph(grid):
    """
    각 색상 그룹을 노드로 표현하고, 노드의 가중치로 색상별 셀의 개수를 지정
    두 색상 그룹이 인접한 경우, 에지 생성
    :param grid: two-dimensional array. each element represents colors(plan)
    :return: graph as a NetworkX object with weights representing the count of cells per color
    """

    nrows, ncols = len(grid), len(grid[0])
    graph = nx.Graph()

    # 색상별 셀의 개수를 저장하기 위한 딕셔너리
    color_counts = {}

    for r in range(nrows):
        for c in range(ncols):
            color = grid[r][c]
            # 색상별 셀의 개수 업데이트
            if color in color_counts:
                color_counts[color] += 1
            else:
                color_counts[color] = 1

            # 노드가 아직 그래프에 없으면 추가
            if not graph.has_node(color):
                graph.add_node(color, weight=color_counts[color])
            else:
                # 노드의 가중치(색상별 셀의 개수) 업데이트
                nx.set_node_attributes(graph, {color: {'weight': color_counts[color]}})

            # 현재 셀의 색상과 인접한 다른 색상들 찾기
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    neighbor_color = grid[nr][nc]
                    if neighbor_color != color:
                        graph.add_edge(color, neighbor_color)

    return graph


def build_graph_inside_only(grid):
    """
    동서남북 방향에 outer space 로 boundary cell이 추가된 경우 내부 공간만을 대상으로 그래프 생성
    :param grid:
    :return: graph
    """
    nrows, ncols = len(grid), len(grid[0])
    graph = {}

    for r in range(nrows)[1:nrows-1]:
        for c in range(ncols)[1:ncols-1]:
            color = grid[r][c]
            if color not in graph and color > 0:
                graph[color] = set()

            # 현재 셀의 색상과 인접한 다른 색상들 찾기
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    neighbor_color = grid[nr][nc]
                    if color < 0 or neighbor_color < 0:
                        break
                    elif neighbor_color != color:
                        graph[color].add(neighbor_color)
    return graph

def build_graph_not_connect_outside(grid):
    """
    동서남북 방향의 boundary cell이 존재하는 경우, graph 생성시 outter 공간과 연결하지 않음
    :param grid:
    :return:
    """
    nrows, ncols = len(grid), len(grid[0])
    graph = {}

    for r in range(nrows)[1:nrows-1]:
        for c in range(ncols)[1:ncols-1]:
            color = grid[r][c]
            if color not in graph and color > 0:
                graph[color] = set()

            # 현재 셀의 색상과 인접한 다른 색상들 찾기
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    neighbor_color = grid[nr][nc]
                    if color < 0 and neighbor_color < 0:
                        break
                    elif neighbor_color != color:
                        graph[color].add(neighbor_color)
    return graph



def draw_graph(graph):
    """
    주어진 그래프 정보를 바탕으로 그래프를 시각화.

    Parameters:
    graph (dict): 노드와 에지를 포함하는 그래프 정보. 각 키는 노드(색상)이며,
                  값은 해당 노드에 인접한 노드(색상)의 집합입니다.
    """
    G = nx.Graph()

    # 그래프 구성: 노드와 에지 추가
    for node, edges in graph.items():
        G.add_node(node)
        for edge in edges:
            G.add_edge(node, edge)
    print(G)
    # 노드 위치 결정
    pos = nx.spring_layout(G)

    # 그래프 드로잉
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10, font_weight='bold')

    plt.show()


def draw_weighted_graph(graph):
    """
    node의 weight에 따라 노드의 크기가 달라짐
    :param graph:
    :return:
    """
    pos = nx.spring_layout(graph)
    node_sizes = [1000 * graph.nodes[node]['weight']^2 for node in graph.nodes]
    nx.draw(graph, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', edge_color='gray')
    node_labels = nx.get_node_attributes(graph, 'weight')
    nx.draw_networkx_labels(graph, pos)
    plt.show()

def add_boundary(grid):
    rows, cols = len(grid), len(grid[0])

    # 새로운 grid의 크기를 기존보다 각각 2만큼 더 크게 설정.
    # 이렇게 하면 상하좌우에 한 칸씩 여유 공간이 생
    new_grid = [[-1] * (cols + 2) for _ in range(rows + 2)]

    # 기존 grid의 값을 새로운 grid에 복사.
    # 이때, 새로운 grid에서는 한 칸씩 내부로 이동하여 값을 설정.
    for r in range(rows):
        for c in range(cols):
            new_grid[r + 1][c + 1] = grid[r][c]

    return new_grid


def add_4way_boundary(grid):
    """
    동서남북 각 경계에 외부 공간을 추가하고 각기 다른 값을 할당
    북쪽 경계: -2 값 할당
    남쪽 경계: -3 값 할당
    동쪽 경계: -4 값 할당
    서쪽 경계: -5 값 할당
    기존 grid의 각 셀은 새로운 grid에서 한 칸씩 내부로 이동하여 값 유지
    :param grid: 입력 list of cell coord(row,col)
    :return:new_grid
    """
    rows, cols = len(grid), len(grid[0])

    # 새로운 grid의 크기를 기존보다 각각 2만큼 더 크게 설정하고,
    # 북쪽(-2)과 남쪽(-3) 경계를 설정합니다.
    new_grid = [[-2] * (cols + 2) if r == 0 or r == rows + 1 else [-1] * (cols + 2) for r in range(rows + 2)]

    # 동쪽(-4)과 서쪽(-5) 경계를 설정합니다.
    for r in range(1, rows + 1):
        new_grid[r][0] = -5  # 서쪽
        new_grid[r][cols + 1] = -4  # 동쪽

    # 남쪽 경계를 -3으로 설정합니다.
    new_grid[rows + 1] = [-3] * (cols + 2)

    # 기존 grid의 값을 새로운 grid에 복사합니다.
    for r in range(rows):
        for c in range(cols):
            new_grid[r + 1][c + 1] = grid[r][c]

    return new_grid



def test_build_graph():
    graph = build_graph(test_grid)
    for node, edges in graph.items():
        print(f"{node}: {edges}")
    draw_graph(graph)

def test_build_weighted_graph():
    graph = build_weighted_graph(test_grid)
    draw_weighted_graph(graph)

def test_add_boundary():
    new_grid = add_boundary(test_grid)
    print(new_grid)

def test_add_4way_boundary():
    new_grid = add_4way_boundary(test_grid)
    print(new_grid)

def dummy(grid):
    new_grid = [[-2, -2, -2, -2, -2, -2, -2],
                [-5,  2,  2,  2, -1, -1, -4],
                [-5,  4,  1,  1, -1, -1, -4],
                [-5,  4,  1,  3, -1, -1, -4],
                [-5,  4,  4,  3,  3,  3, -4],
                [-5,  4,  4,  3,  3,  3, -4],
                [-3, -3, -3, -3, -3, -3, -3]]
    graph = build_graph(new_grid)
    draw_graph(graph)

    graph = build_graph(new_grid)
    draw_graph(graph)



# 사용자 입력에 따라 모듈을 실행하는 메인 함수
def main():
    # 사용 가능한 모듈 이름 나열
    modules = {
        "1": test_add_4way_boundary,
        "2": test_add_boundary,
        "3": test_build_graph,
        "4": test_build_weighted_graph,
    }

    # 사용자에게 모듈 선택을 안내
    print("Available modules:")
    for key, name in modules.items():
        print(f"{key}. {name.__name__}")

    # 사용자 입력 받기
    choice = input("Enter the number of the module to run: ")

    # 선택된 모듈 실행
    if choice in modules:
        modules[choice]()
    else:
        print("Invalid module number.")



if __name__ == '__main__':
    main()
