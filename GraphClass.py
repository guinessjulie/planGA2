import networkx
import networkx as nx

from plan import create_floorplan
import matplotlib.pyplot as plt
import plan_utils
import constants
import trivial_utils


class GridGraph:
    def __init__(self, grid):
        self.nrows, self.ncols = len(grid), len(grid[0])
        self.grid = grid  # 있어야 되는지 없어도 되는지 잘 모르겠다. 있어야 될듯. 왜냐하면 grid를 받아서 nrow와 ncol이 생성되니까.
        self.graph = nx.Graph()

    def print_graph(self, graph):
        for node in graph.nodes:
            print(f'{node}: {list(graph.adj[node])}')

    def build_graph(self):
        return GraphBuilder.build_graph(self.grid)

    def __eq__(self, other):
        if not isinstance(other, GridGraph):
            return False
        return nx.is_isomorphic(self.graph, other.graph)

    def is_equal_node_exact(self, other):
        if not isinstance(other, GridGraph):
            return False
        return nx.is_isomorphic(self.graph, other.graph, node_match=lambda n1, n2:n1==n2)


class GraphBuilder:

    @staticmethod
    def init_graph(grid):
        return nx.Graph(), len(grid), len(grid[0])

    @staticmethod
    def print_graph(graph):
        for node in graph.nodes:
            print(f'{node}:{list(graph.adj[node])}')

    @staticmethod
    def build_graph(grid, arg1=None, arg2=None):

        graph, nrows, ncols = GraphBuilder.init_graph(grid)
        for r in range(nrows):
            for c in range(ncols):
                color = grid[r][c]
                # 각 셀의 색상을 노드로 추가. 이미 존재하는 경우 무시됨.
                graph.add_node(color)

                # 현재 셀의 색상과 인접한 다른 색상들 사이에 엣지 생성
                neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dr, dc in neighbors:  # 상하좌우 인접 셀 확인
                    nr, nc = r + dr, c + dc #인접 셀 위치
                    if 0 <= nr < nrows and 0 <= nc < ncols: # valid
                        neighbor_color = grid[nr][nc]
                        if neighbor_color != color:
                            # 서로 다른 색상 사이에 엣지 추가
                            graph.add_edge(color, neighbor_color)
        return graph

    @staticmethod
    def build_weighted_graph(grid,arg1=None, arg2=None):

        # 색상별 셀의 개수를 저장하기 위한 딕셔너리
        color_counts = {}
        graph, nrows, ncols = GraphBuilder.init_graph(grid)

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
                for dr, dc in constants.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        neighbor_color = grid[nr][nc]
                        if neighbor_color != color:
                            graph.add_edge(color, neighbor_color)

        return graph

        # 이걸 이렇게 할 게 아니라, grid는 고정되어 있다고 생각하고, 단지 inside cell만 연결되도록 하자. 0부터 nrows로 변경하고 -1 만 제외하면 될듯.
        # def build_graph_inside_cells(self):
        #
        #     for r in range(1, nrows-1):  # Adjusted range to exclude first and last row
        #         for c in range(1, ncols-1):  # Adjusted range to exclude first and last column
        #             color = grid[r][c]
        #             if color > 0:  # Check if color is positive before adding to graph
        #                 graph.add_node(color)  # Add node if not present
        #
        #             for dr, dc in DIRECTIONS:
        #                 nr, nc = r + dr, c + dc
        #                 if 0 <= nr < nrows and 0 <= nc < ncols:
        #                     neighbor_color = grid[nr][nc]
        #                     if color > 0 and neighbor_color > 0 and neighbor_color != color:
        #                         graph.add_edge(color, neighbor_color)  # Add an edge between different colors
        #
        # print(graph)

    @staticmethod
    def build_graph_with_inside_cells(grid, arg1=None, arg2=None):
        """
        build graph with element color of the cell is greater than 1
        :return:
        """
        graph, nrows, ncols = GraphBuilder.init_graph(grid)
        for r in range(nrows):  # Adjusted range to exclude first and last row
            for c in range(ncols):  # Adjusted range to exclude first and last column
                color = grid[r][c]
                if color > 0:  # Check if color is positive before adding to graph
                    graph.add_node(color)  # Add node if not present

                for dr, dc in constants.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        neighbor_color = grid[nr][nc]
                        if color > 0 and neighbor_color > 0 and neighbor_color != color:
                            graph.add_edge(color, neighbor_color)  # Add an edge between different colors
        GraphBuilder.print_graph(graph)
        return graph

        # def build_graph_inside_cells(self):

    #
    #     for r in range(self.nrows)[1:self.nrows-1]:
    #         for c in range(self.ncols)[1:self.ncols-1]:
    #             color = self.grid[r][c]
    #             if color not in self.graph and color > 0:
    #                 self.graph[color] = set()
    #
    #             # 현재 셀의 색상과 인접한 다른 색상들 찾기
    #             for dr, dc in DIRECTIONS:
    #                 nr, nc = r + dr, c + dc
    #                 if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
    #                     neighbor_color = self.grid[nr][nc]
    #                     if color < 0 or neighbor_color < 0:
    #                         break
    #                     elif neighbor_color != color:
    #                         self.graph[color].add(neighbor_color)
    #     return graph

    # def build_graph_not_connect_outside(grid):
    #     nrows, ncols = len(grid), len(grid[0])
    #     graph = {}
    #
    #     # 인접 셀을 확인하기 위한 방향 벡터
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #
    #     for r in range(nrows)[1:nrows-1]:
    #         for c in range(ncols)[1:ncols-1]:
    #             color = grid[r][c]
    #             if color not in graph and color > 0:
    #                 graph[color] = set()
    #
    #             # 현재 셀의 색상과 인접한 다른 색상들 찾기
    #             for dr, dc in directions:
    #                 nr, nc = r + dr, c + dc
    #                 if 0 <= nr < nrows and 0 <= nc < ncols:
    #                     neighbor_color = grid[nr][nc]
    #                     if color < 0 and neighbor_color < 0:
    #                         break
    #                     elif neighbor_color != color:
    #                         graph[color].add(neighbor_color)
    #     print(graph)

    # 4-way graph
    @staticmethod
    def build_graph_connect_4way(grid, arg1=None, arg2=None):

        graph = nx.Graph()
        nrows, ncols = len(grid), len(grid[0])

        for r in range(nrows)[1:nrows - 1]:
            for c in range(ncols)[1:ncols - 1]:
                color = grid[r][c]
                if color > 0 and color not in graph:
                    graph.add_node(color)  # Add the node to the graph if it's not already added

                # 현재 셀의 색상과 인접한 다른 색상들 찾기
                for dr, dc in constants.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        neighbor_color = grid[nr][nc]
                        if color < 0 and neighbor_color < 0:
                            break
                        # elif neighbor_color != color and neighbor_color > 0:
                        elif neighbor_color != color:
                            graph.add_edge(color, neighbor_color)  # Add an edge between different colors

        return graph


class GraphDrawer:

    @staticmethod
    def draw_graph(graph: networkx, save_path, arg2=None):
        """

        :type graph: networkx
        """
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700,
                font_size=10, font_weight='bold')

        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def draw_weighted_graph(graph, arg1=None, arg2=None):
        pos = nx.spring_layout(graph)
        node_sizes = [1000 * graph.nodes[node]['weight'] ^ 2 for node in graph.nodes]
        nx.draw(graph, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', edge_color='gray')
        node_labels = nx.get_node_attributes(graph, 'weight')
        nx.draw_networkx_labels(graph, pos)
        plt.show()

    @staticmethod
    def draw_graph_with_boundary(graph: networkx, arg1=None, arg2=None):
        """
        주어진 그래프 정보를 바탕으로 그래프를 시각화
        -2:맨 위로
        -3:맨 아래쪽
        -4:맨 오른쪽
        -5:맨 왼쪽

        Parameters:
        graph (dict): 노드와 에지를 포함하는 그래프 정보. 각 키는 노드(색상)이며,
                      값은 해당 노드에 인접한 노드(색상)의 집합입니다.
        """

        # 그래프 구성: 노드와 에지 추가
        # for node, edges in graph.items():
        #     G.add_node(node)
        #     for edge in edges:
        #         G.add_edge(node, edge)
        # print(G)

        # 노드 위치 결정
        # outside space: boundary는 맨 가장자리에 위치
        pos = nx.spring_layout(graph)
        pos[-2] = (0, 2)
        pos[-3] = (0, -2)
        pos[-4] = (2, 0)
        pos[-5] = (-2, 0)
        pos[2] = [-1, 1]
        pos[4] = [-1, -1]
        pos[3] = [1, -1]
        pos[-1] = [1, 1]
        # pos.update(nx.spring_layout(self.graph, pos=pos, fixed=[-2,-3,-4,-5]))

        # 그래프 드로잉
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700,
                font_size=10, font_weight='bold')

        # 표시
        plt.show()

    def arrange_corner(self):
        pass


# from commons import test_grid_boundary


# graph = GridGraph(grid)
# graph.build_graph()
# graph.draw_graph()
# graph.build_weighted_graph()
# graph.draw_weighted_graph()

# test module

