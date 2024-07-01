import numpy as np
from collections import defaultdict
import ezdxf
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from shapely.geometry import Point, Polygon, LineString


class GridPolygon:
    def __init__(self, grid, cell_size=1000, padding_size=1000, min_area=None, min_length=None):
        self.grid = np.flipud(grid)
        self.cell_size = cell_size
        self.padding_size = padding_size
        self.padded_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=-1)
        self.min_area = min_area
        self.min_length = min_length

    # 셀의 꼭지점 좌표를 구하는 함수
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


    # 외곽선 좌표를 구하는 함수
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
            room_polygons[room_number] = self.get_polygon_corners(room_number)

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

    def calculate_area(self, polygon):
        x, y = zip(*polygon)
        return 0.5 * abs(sum(x[i] * y[i - 1] - y[i] * x[i - 1] for i in range(len(polygon))))

    def get_all_room_areas(self):
        room_polygons = self.get_all_room_polygons()
        room_areas = {}
        for room_number, polygon in room_polygons.items():
            room_areas[room_number] = self.calculate_area(polygon)
        return room_areas

    def get_combined_area(self):
        combined_polygon = self.get_combined_polygon()
        return self.calculate_area(combined_polygon)

    def calculate_perimeter(self, polygon):
        perimeter = 0
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            perimeter += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return perimeter

    def calculate_min_length(self, polygon):
        min_length = float('inf')
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            min_length = min(min_length, length)
        return min_length


    def get_all_room_perimeters(self):
        room_polygons = self.get_all_room_polygons()
        room_perimeters = {}
        for room_number, polygon in room_polygons.items():
            room_perimeters[room_number] = self.calculate_perimeter(polygon)
        return room_perimeters


    def get_combined_perimeter(self):
        combined_polygon = self.get_combined_polygon()
        return self.calculate_perimeter(combined_polygon)

    def calculate_simplicity(self, polygon):
        return (16 * self.calculate_area(polygon) ) / ( self.calculate_perimeter(polygon) ** 2.0)

    def get_all_room_simplicities(self):
        room_polygons=self.get_all_room_polygons()
        room_simplicities = {}
        for room_number, polygon in room_polygons.items():
            room_simplicities[room_number] = self.calculate_simplicity(polygon)
        return room_simplicities

    def get_combined_simplicity(self):
        combined_polygon = self.get_combined_polygon()
        return self.calculate_simplicity(combined_polygon)



    # 클래스에 새로 추가한 함수들을 바인딩

    def save_polygon_to_dxf(self, polygon, filename):
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        polyline = msp.add_lwpolyline(polygon, close=True)
        polyline.dxf.layer = "All_Rooms"

        doc.saveas(filename)

    def save_room_polygons_to_dxf(self, room_polygons, filename):
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        for room_number, corners in room_polygons.items():
            polyline = msp.add_lwpolyline(corners, close=True)
            polyline.dxf.layer = f"Room_{room_number}"

        doc.saveas(filename)



    def save_polygon_to_png_with_dimensions(self, polygons, filename):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # 폴리건 그리기
        for room_number, polygon in polygons.items():
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
        # ax.invert_yaxis()  # y축 뒤집기 (상하 반전)

        # 치수 표시
        offset = 50
        for room_number, polygon in polygons.items():
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
        for room_number, polygon in polygons.items():
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


    def get_distances_to_boundary(self, reference_point, polygon):
        poly = Polygon(polygon)

        # 수직 및 수평 선 생성
        vertical_line = LineString([(reference_point[0], min(polygon, key=lambda x: x[1])[1]),
                                    (reference_point[0], max(polygon, key=lambda x: x[1])[1])])
        horizontal_line = LineString([(min(polygon, key=lambda x: x[0])[0], reference_point[1]),
                                      (max(polygon, key=lambda x: x[0])[0], reference_point[1])])

        def update_distances(intersections, north_dist, south_dist, east_dist, west_dist):
            for line in intersections.geoms if hasattr(intersections, 'geoms') else [intersections]:
                for x, y in line.coords:
                    if y > reference_point[1]:
                        north_dist = min(north_dist, y - reference_point[1])
                    elif y < reference_point[1]:
                        south_dist = min(south_dist, reference_point[1] - y)
                    if x > reference_point[0]:
                        east_dist = min(east_dist, x - reference_point[0])
                    elif x < reference_point[0]:
                        west_dist = min(west_dist, reference_point[0] - x)
            return north_dist, south_dist, east_dist, west_dist

        # 초기 거리 값을 무한대로 설정
        north_dist = south_dist = east_dist = west_dist = float('inf')

        if poly.intersects(vertical_line):
            vertical_intersections = poly.intersection(vertical_line)
            north_dist, south_dist, east_dist, west_dist = update_distances(
                vertical_intersections, north_dist, south_dist, east_dist, west_dist
            )

        if poly.intersects(horizontal_line):
            horizontal_intersections = poly.intersection(horizontal_line)
            north_dist, south_dist, east_dist, west_dist = update_distances(
                horizontal_intersections, north_dist, south_dist, east_dist, west_dist
            )

        # 예외 처리: 동일 선상에 놓여 있는 경우 inf 값 대체
        if north_dist == float('inf'):
            north_dist = max(polygon, key=lambda x: x[1])[1] - reference_point[1]
        if south_dist == float('inf'):
            south_dist = reference_point[1] - min(polygon, key=lambda x: x[1])[1]
        if east_dist == float('inf'):
            east_dist = max(polygon, key=lambda x: x[0])[0] - reference_point[0]
        if west_dist == float('inf'):
            west_dist = reference_point[0] - min(polygon, key=lambda x: x[0])[0]

        return {
            'North': north_dist / 1000,
            'South': south_dist / 1000,
            'East': east_dist / 1000,
            'West': west_dist / 1000
        }

    def check_constraints(self, polygon):
        if self.min_area and self.calculate_area(polygon) < self.min_area:
            return False
        if self.min_length and self.calculate_min_length(polygon) < self.min_length:
            return False
        return True



if __name__ == '__main__':
    # 예제 사용법
    grid = np.array([
        [ 4,  4,  4,  4,  1, -1, -1],
        [ 2,  2,  4,  1,  1,  5,  5],
        [ 2,  2,  3,  3,  1,  5, -1],
        [-1, -1,  3,  3, -1, -1, -1]
    ])

    grid_polygon = GridPolygon(grid, min_area=2000000, min_length=2000)
    room_polygons = grid_polygon.get_all_room_polygons()
    combined_polygon = grid_polygon.get_combined_polygon()

    # 각각의 방 폴리건을 DXF 파일로 저장
    grid_polygon.save_room_polygons_to_dxf(room_polygons, "room_polygons.dxf")

    # 전체 폴리건을 DXF 파일로 저장
    grid_polygon.save_polygon_to_dxf(combined_polygon, "combined_polygon.dxf")

    # 폴리건을 PNG 파일로 저장
    grid_polygon.save_polygon_to_png(room_polygons, "room_polygons.png")
    grid_polygon.save_polygon_to_png_with_dimensions(room_polygons, "room_polygons.png")

    # 길이 출력
    room_perimeters = grid_polygon.get_all_room_perimeters()
    combined_perimeter = grid_polygon.get_combined_perimeter()

    room_areas = grid_polygon.get_all_room_areas()
    combined_area = grid_polygon.get_combined_area()

    room_simplicities = grid_polygon.get_all_room_simplicities()

    # 출력
    print("Room Areas (m²):")
    for room_number, area_mm2 in room_areas.items():
        area_m2 = area_mm2 / 1_000_000  # mm² to m²
        print(f"Room {room_number}: {area_m2:.2f} m²")

    print("\nRoom Perimeters (m):")
    for room_number, perimeter_mm in room_perimeters.items():
        perimeter_m = perimeter_mm / 1_000  # mm to m
        print(f"Room {room_number}: {perimeter_m:.2f} m")

    combined_area_m2 = combined_area / 1_000_000  # mm² to m²
    combined_perimeter_m = combined_perimeter / 1_000  # mm to m

    print("\nRoom Simplicity (max:1)")
    for room_number, simplicity in room_simplicities.items():
        print(f'Room {room_number}: {simplicity:.2f}')

    print("\nCombined Area: {:.2f} m²".format(combined_area_m2))
    print("Combined Perimeter: {:.2f} m".format(combined_perimeter_m))


    # 각 방의 참조점과 경계까지의 거리 계산 및 출력
    print("\nRoom Reference Points and Distances to Boundaries (m):")
    for room_number, polygon in room_polygons.items():
        if grid_polygon.check_constraints(polygon):
            reference_point = grid_polygon.get_room_reference_point(polygon)
            distances = grid_polygon.get_distances_to_boundary(reference_point, combined_polygon)
            # distances = grid_polygon.get_distances_to_boundary_shapely2(reference_point, combined_polygon)
            distances_m = {direction: distance for direction, distance in distances.items()}  # mm to m
            print(f"Room {room_number}: Reference Point {reference_point}, Distances {distances_m}")
        else:
            print(f'Room {room_number} does not meet the size constraint')
