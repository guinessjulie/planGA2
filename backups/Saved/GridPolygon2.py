import numpy as np
from collections import defaultdict
import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from shapely.geometry import Point
from room_polygon import RoomPolygon
import math

class GridPolygon:
    def __init__(self, grid, cell_size=1000, padding_size=1000, min_area=None, min_length=None):
        self.grid = grid
        self.cell_size = cell_size
        self.padding_size = padding_size
        self.padded_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=-1)
        self.min_area = min_area
        self.min_length = min_length
        self.room_polygons = self.get_all_room_polygons()

    def get_cell_corners(self, room_number):
        rows, cols = self.padded_grid.shape
        corners = []
        for i in range(cols):
            for j in range(rows):
                if self.padded_grid[j, i] == room_number:
                    cell_corners = [
                        (i * self.cell_size, j * self.cell_size),
                        (i * self.cell_size, (j + 1) * self.cell_size),
                        ((i + 1) * self.cell_size, (j + 1) * self.cell_size),
                        ((i + 1) * self.cell_size, j * self.cell_size)
                    ]
                    corners.append(cell_corners)
        return corners

    def sort_and_remove_collinear(self, corners):
        def is_collinear(p1, p2, p3):
            return (p2[1] - p1[1]) * (p3[0] - p2[0]) == (p3[1] - p2[1]) * (p2[0] - p1[0])

        center = np.mean(corners, axis=0)
        corners.sort(key=lambda c: np.arctan2(c[1] - center[1], c[0] - center[0]))

        filtered_corners = []
        n = len(corners)
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            p3 = corners[(i + 2) % n]
            if not is_collinear(p1, p2, p3):
                filtered_corners.append(p2)
        return filtered_corners

    def get_polygon_corners(self, room_number):
        cell_corners = self.get_cell_corners(room_number)
        corner_count = defaultdict(int)

        for cell in cell_corners:
            for corner in cell:
                corner_count[corner] += 1

        outer_corners = [corner for corner, count in corner_count.items() if count < 4]
        return self.sort_and_remove_collinear(outer_corners)

    def get_all_room_polygons(self):
        room_numbers = np.unique(self.padded_grid)
        room_numbers = room_numbers[room_numbers > 0]
        room_polygons = {}

        for room_number in room_numbers:
            corners = self.get_polygon_corners(room_number)
            room_polygons[room_number] = RoomPolygon(corners)

        return room_polygons

    def get_combined_polygon(self):
        cell_corners = self.get_cell_corners_all()
        corner_count = defaultdict(int)

        for cell in cell_corners:
            for corner in cell:
                corner_count[corner] += 1

        outer_corners = [corner for corner, count in corner_count.items() if count < 4]
        return self.sort_and_remove_collinear(outer_corners)

    def get_cell_corners_all(self):
        rows, cols = self.padded_grid.shape
        corners = []
        for i in range(cols):
            for j in range(rows):
                if self.padded_grid[j, i] != -1:
                    cell_corners = [
                        (i * self.cell_size, j * self.cell_size),
                        (i * self.cell_size, (j + 1) * self.cell_size),
                        ((i + 1) * self.cell_size, (j + 1) * self.cell_size),
                        ((i + 1) * self.cell_size, j * self.cell_size)
                    ]
                    corners.append(cell_corners)
        return corners


def save_polygon_to_dxf(self, polygon, filename):
    flipped_polygon = [(x, self.cell_size * (self.grid.shape[0] + 1) - y) for x, y in polygon]
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    polyline = msp.add_lwpolyline(flipped_polygon, close=True)
    polyline.dxf.layer = "All_Rooms"

    doc.saveas(filename)



def save_room_polygons_to_dxf(self, room_polygons, filename):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    for room_number, room_polygon in room_polygons.items():
        corners = room_polygon.corners
        flipped_corners = [(x, self.cell_size * (self.grid.shape[0] + 1) - y) for x, y in corners]
        polyline = msp.add_lwpolyline(flipped_corners, close=True)
        polyline.dxf.layer = f"Room_{room_number}"

    doc.saveas(filename)

def save_polygon_to_png_with_dimensions(self, polygons, filename):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # 폴리건 그리기
    for room_number, room_polygon in polygons.items():
        polygon = room_polygon.corners
        color = (random.random(), random.random(), random.random())  # 무작위 색상 선택
        poly = patches.Polygon(polygon, closed=True, edgecolor='black', facecolor=color, alpha=0.5,
                               label=f'Room {room_number}')
        ax.add_patch(poly)

        # 폴리건의 중심 좌표 계산
        center_x = sum([point[0] for point in polygon]) / len(polygon)
        center_y = sum([point[1] for point in polygon]) / len(polygon)

        # 레퍼런스 포인트 빨간색으로 표시
        ax.plot(center_x, center_y, 'ro')

        # 룸 번호 기재
        ax.text(center_x, center_y, str(room_number), fontsize=12, ha='center', va='center', color='black')

    # 전체 그리드 설정
    rows, cols = self.padded_grid.shape
    ax.set_xlim(0, cols * self.cell_size)
    ax.set_ylim(0, rows * self.cell_size)
    ax.invert_yaxis()  # y축 뒤집기 (상하 반전)

    # 치수 표시
    offset = 50
    for room_number, room_polygon in polygons.items():
        polygon = room_polygon.corners
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000  # mm to m

            if x1 == x2:  # 세로 방향
                rotation = 'vertical'
                if x1 < cols * self.cell_size / 2:  # 서쪽
                    offset_x, offset_y = -offset, 0
                else:  # 동쪽
                    offset_x, offset_y = offset, 0
            else:  # 가로 방향
                rotation = 'horizontal'
                if y1 < rows * self.cell_size / 2:  # 북쪽
                    offset_x, offset_y = 0, -offset
                else:  # 남쪽
                    offset_x, offset_y = 0, offset

            ax.text(mid_x + offset_x, mid_y + offset_y, f'{distance:.2f}', fontsize=8, ha='center', va='center',
                    color='blue', rotation=rotation)

    plt.axis('off')  # 축 숨기기
    ax.legend(loc='upper right')  # 범례 추가
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_polygon_to_png(self, polygons, filename):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # 폴리건 그리기
    for room_number, room_polygon in polygons.items():
        polygon = room_polygon.corners
        color = (random.random(), random.random(), random.random())  # 무작위 색상 선택
        poly = patches.Polygon(polygon, closed=True, edgecolor='black', facecolor=color, alpha=0.5)
        ax.add_patch(poly)

        # 폴리건의 중심 좌표 계산
        center_x = sum([point[0] for point in polygon]) / len(polygon)
        center_y = sum([point[1] for point in polygon]) / len(polygon)

        # 룸 번호 기재
        ax.text(center_x, center_y, str(room_number), fontsize=12, ha='center', va='center', color='black')

    # 전체 그리드 설정
    rows, cols = self.padded_grid.shape
    ax.set_xlim(0, cols * self.cell_size)
    ax.set_ylim(0, rows * self.cell_size)
    ax.invert_yaxis()  # y축 뒤집기 (상하 반전)

    plt.axis('off')  # 축 숨기기
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_room_reference_point(self, polygon):
    x, y = zip(*polygon)
    return (sum(x) / len(polygon), sum(y) / len(polygon))

def adjust_polygon_edges(self, room_number, edge_index, distance):
    """
    Adjust the edges of the specified RoomPolygon and any other RoomPolygon affected by the move.

    :param room_number: The number of the room to adjust.
    :param edge_index: The index of the edge to move.
    :param distance: The distance to move the edge.
    """
    target_polygon = self.room_polygons[room_number]

    # Move the edge of the target polygon
    target_polygon.move_edge(edge_index, distance)

    # Update the room polygons in the grid
    self.room_polygons[room_number] = target_polygon

    # Adjust other polygons if they share the moved edge or corner
    for other_room_number, other_polygon in self.room_polygons.items():
        if other_room_number != room_number:
            shared_corners = target_polygon.get_shared_corners(other_polygon)
            if shared_corners:
                target_polygon.update_shared_edges(other_polygon, shared_corners)
                self.room_polygons[other_room_number] = other_polygon


def find_shared_edge(self, polygon1, polygon2):
    """
    Find the shared edge index between two polygons if it exists.

    :param polygon1: The first RoomPolygon.
    :param polygon2: The second RoomPolygon.
    :return: The index of the shared edge in polygon1 if it exists, else None.
    """
    for i in range(len(polygon1.corners)):
        for j in range(len(polygon2.corners)):
            if polygon1.corners[i] == polygon2.corners[j] and \
                    polygon1.corners[(i + 1) % len(polygon1.corners)] == polygon2.corners[
                (j + 1) % len(polygon2.corners)]:
                return i
            if polygon1.corners[i] == polygon2.corners[(j + 1) % len(polygon2.corners)] and \
                    polygon1.corners[(i + 1) % len(polygon1.corners)] == polygon2.corners[j]:
                return i
    return None

def print_room_metrics(room_polygons, combined_polygon, grid_polygon, stage):
    print(f"\nRoom Metrics {stage} Change:")
    for room_number, room_polygon in room_polygons.items():
        print(f"Room {room_number}: Area={room_polygon.area / 1_000_000:.2f} m², "
              f"Perimeter={room_polygon.perimeter / 1000:.2f} m, "
              f"Simplicity={room_polygon.simplicity:.2f}")

    combined_area = RoomPolygon(combined_polygon).area
    combined_perimeter = RoomPolygon(combined_polygon).perimeter
    combined_simplicity = RoomPolygon(combined_polygon).simplicity

    print(f"\nCombined Metrics {stage} Change:")
    print(f"Area={combined_area / 1_000_000:.2f} m²")
    print(f"Perimeter={combined_perimeter / 1000:.2f} m")
    print(f"Simplicity={combined_simplicity:.2f}")

    print(f"\nRoom Reference Points and Distances to Boundaries {stage} Change (m):")
    for room_number, room_polygon in room_polygons.items():
        if room_polygon.check_constraints(grid_polygon.min_area, grid_polygon.min_length):
            reference_point = grid_polygon.get_room_reference_point(room_polygon.corners)
            distances = room_polygon.get_distances_to_boundary(reference_point, combined_polygon)
            distances_m = {direction: distance for direction, distance in distances.items()}
            print(f"Room {room_number}: Reference Point {reference_point}, Distances {distances_m}")
        else:
            print(f'Room {room_number} does not meet the size constraint')

def save_files(grid_polygon, room_polygons, combined_polygon, suffix):
    grid_polygon.save_room_polygons_to_dxf(room_polygons, f"room_polygons_{suffix}.dxf")
    grid_polygon.save_polygon_to_dxf(combined_polygon, f"combined_polygon_{suffix}.dxf")
    grid_polygon.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")
    grid_polygon.save_polygon_to_png_with_dimensions(room_polygons, f"room_polygons_with_dimensions_{suffix}.png")

if __name__ == '__main__':
    grid = np.array([
        [ 4,  4,  4,  4,  1, -1, -1],
        [ 2,  2,  4,  1,  1,  5,  5],
        [ 2,  2,   3,  3,  1,  5, -1],
        [-1, -1,  3,  3, -1, -1, -1]
    ])

    grid_polygon = GridPolygon(grid, min_area=2000000, min_length=2000)
    room_polygons = grid_polygon.get_all_room_polygons()
    combined_polygon = grid_polygon.get_combined_polygon()

    # 변경 전 파일 저장 및 메트릭 출력
    save_files(grid_polygon, room_polygons, combined_polygon, "before")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "Before")

    # Room 4의 첫 번째 변을 500mm 이동 (수직 방향의 edge는 수평 방향으로 이동)
    grid_polygon.adjust_polygon_edges(room_number=4, edge_index=0, distance=500)

    # 이동 후 업데이트된 RoomPolygon 가져오기
    room_polygons = grid_polygon.room_polygons
    combined_polygon = grid_polygon.get_combined_polygon()  # combined_polygon도 업데이트

    # 변경 후 파일 저장 및 메트릭 출력
    save_files(grid_polygon, room_polygons, combined_polygon, "after")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "After")
def save_files(grid_polygon, room_polygons, combined_polygon, suffix):
    grid_polygon.save_room_polygons_to_dxf(room_polygons, f"room_polygons_{suffix}.dxf")
    grid_polygon.save_polygon_to_dxf(combined_polygon, f"combined_polygon_{suffix}.dxf")
    grid_polygon.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")
    grid_polygon.save_polygon_to_png_with_dimensions(room_polygons, f"room_polygons_with_dimensions_{suffix}.png")

if __name__ == '__main__':
    grid = np.array([
        [ 4,  4,  4,  4,  1, -1, -1],
        [ 2,  2,  4,  1,  1,  5,  5],
        [ 2,  2,   3,  3,  1,  5, -1],
        [-1, -1,  3,  3, -1, -1, -1]
    ])

    grid_polygon = GridPolygon(grid, min_area=2000000, min_length=2000)
    room_polygons = grid_polygon.get_all_room_polygons()
    combined_polygon = grid_polygon.get_combined_polygon()

    # 변경 전 파일 저장 및 메트릭 출력
    save_files(grid_polygon, room_polygons, combined_polygon, "before")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "Before")

    # Room 4의 첫 번째 변을 500mm 이동 (수직 방향의 edge는 수평 방향으로 이동)
    grid_polygon.adjust_polygon_edges(room_number=4, edge_index=0, distance=500)

    # 이동 후 업데이트된 RoomPolygon 가져오기
    room_polygons = grid_polygon.room_polygons
    combined_polygon = grid_polygon.get_combined_polygon()  # combined_polygon도 업데이트

    # 변경 후 파일 저장 및 메트릭 출력
    save_files(grid_polygon, room_polygons, combined_polygon, "after")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "After")