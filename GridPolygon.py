import numpy as np
from collections import defaultdict
import ezdxf
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from RoomPolygon import RoomPolygon
from PolygonExporter import PolygonExporter
import math
import sys
class GridPolygon:
    def __init__(self, grid, cell_size=1000, padding_size=1000, min_area=None, min_length=None):
        self.grid = grid
        self.cell_size = cell_size
        self.padding_size = padding_size
        self.padded_grid = np.pad(self.grid, pad_width=1, mode='constant', constant_values=-1)
        self.min_area = min_area
        self.min_length = min_length
        self.room_polygons = self.get_all_room_polygons()


    def get_cell_corners(self, room_number): # list all the points for all cells for room, redundantly
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

    def get_all_room_polygons(self):
        room_numbers = np.unique(self.padded_grid)
        room_numbers = room_numbers[room_numbers > 0]
        room_polygons = {}

        for room_number in room_numbers:
            # underway: RoomPolyg에 onroom_id 추가
            # underway: examin RoomPolygon 좌표와 area
            corners = self.get_polygon_corners(room_number)
            room_polygons[room_number] = RoomPolygon(corners, room_number)
            # room_polygons[room_number] = RoomPolygon(self.get_polygon_corners(room_number)) #info splited to two line above # todo to see how to get polygon corners

        return room_polygons

    def filter_corners(self, room_number):
        cell_corners = self.get_cell_corners(room_number)
        corner_count = defaultdict(int)

        for cell_corner in cell_corners:
            for corner in cell_corner:
                corner_count[corner] +=1

        outer_corners = [corner for corner, count in corner_count.items() if count == 1 or count == 3]
        return outer_corners

    def add_horizontal_edge(self,corners, next_rotation, last):
        next_rotation = 'vertical' if next_rotation == 'horizontal' else 'horizontal'
        horiz_start = last
        # print(f'horiz_start = {horiz_start}')
        horiz_points = [c for c in corners if c[1] == horiz_start[1]]
        lefts = [c for c in horiz_points if c[0] < horiz_start[0]]
        rights = [c for c in horiz_points if c[0] > horiz_start[0]]

        # 홀수 개수인 집합 고르기
        if len(lefts) % 2 != 0:
            odd_set = lefts
        else:
            odd_set = rights

        if len(odd_set) == 1:  # 하나이면 그냥 그거
            horiz_end = odd_set[0]
        else:
            horiz_end = min(odd_set, key=lambda point: math.sqrt(
                (point[0] - horiz_start[0]) ** 2 + (point[1] - horiz_start[1]) ** 2))

        print(f'horiz_start, horiz_end = ({horiz_start}, {horiz_end})')
        # outers.append((horiz_start, horiz_end))
        # print(f'outers={outers}')
        return (horiz_start, horiz_end)

    def add_vertical_edge(self,corners, next_rotation, last):
        # next_vertical
        next_rotation = 'vertical' if next_rotation == 'horizontal' else 'horizontal'
        vert_start = last

        vert_points = [c for c in corners if c[0] == vert_start[0]]  # x 축이 같은  y,  세로로 직선을 그어 만나면 거기서 end
        vert_points.sort(key=lambda c: c[1], reverse=True)
        # print(f'vert_points = {vert_points}')
        uppers = [c for c in vert_points if c[1] < vert_start[1]]
        unders = [c for c in vert_points if c[1] > vert_start[1]]

        # 홀수 개수인 집합 고르기
        if len(uppers) % 2 != 0:
            odd_set = uppers
        else:
            odd_set = unders

        if len(odd_set) == 1:  # 하나이면 그냥 그거
            vert_end = odd_set[0]
        else:
            vert_end = min(odd_set, key=lambda point: math.sqrt(
                (point[0] - vert_start[0]) ** 2 + (point[1] - vert_start[1]) ** 2))

        print(f'vert_end={vert_end}')
        # last = vert_end
        # outers.append((vert_start,vert_end))
        # print(f'outers = {outers}')
        return (vert_start, vert_end)

    def get_polygon_corners(self, room_number):
        corners = self.filter_corners(room_number)
        next_rotation = 'vertical'

        topy = min(corners, key=lambda coord: coord[1])[1]  # top
        print(topy)
        top_left = min([c for c in corners if c[1] == topy])
        last = top_left

        outers = []
        for i in range(int(len(corners) / 2)):
            points = self.add_horizontal_edge(corners, next_rotation, last)
            last = points[1]
            outers.append(points)
            print(f'points={points},\tlast={last}')
            print(f'outers={outers}')
            points = self.add_vertical_edge(corners, next_rotation, last)
            last = points[1]
            outers.append(points)
            print(f'points={points},\tlast={last}')
            print(f'outers={outers}')
        corners = [pair[0] for pair in outers]
        return corners


    def sort_corners_clockwise(self, corners):
        center = np.mean(corners, axis=0)
        corners.sort(key=lambda c: np.arctan2(c[1] - center[1], c[0] - center[0]))
        return corners

    def get_polygon_corners7(self, room_number):
        cell_corners = self.get_cell_corners(room_number)
        corner_count = defaultdict(int)

        for cell in cell_corners:
            for corner in cell:
                corner_count[corner] += 1
        print(f'[get_polygon_corners] Room{room_number} corner_count={corner_count}')
        outer_corners = [corner for corner, count in corner_count.items() if count ==1 or count ==3]
        print(f'[get_polygon_corners] Room{room_number} outer_corners = {outer_corners}')
        corners = self.sort_and_remove_collinear(outer_corners)

        return corners

    def sort_corners_clockwise(self, corners):
        center = np.mean(corners, axis=0)
        top_left = np.min(corners, axis=0)
        # corners.sort(key=lambda c: np.arctan2(c[1] - center[1], c[0] - center[0]))
        corners.sort(key=lambda c: np.arctan2(c[1] - top_left[1], c[0] - top_left[0]))
        return corners


    def sort_and_remove_collinear(self, corners):
        def is_collinear(p1, p2, p3):
            collinear = (p2[1] - p1[1]) * (p3[0] - p2[0]) == (p3[1] - p2[1]) * (p2[0] - p1[0])
            return collinear
        def filter_collinear(sorted_corners):
            filtered_corners = []
            n = len(sorted_corners)

            i = 0
            while i < n:
                p1 = sorted_corners[i]
                filtered_corners.append(p1)
                j = i + 1

                while j < n:
                    p2 = sorted_corners[j % n]
                    if p1[0] == p2[0] or p1[1] == p2[1]:
                        filtered_corners.append(p2)
                        i = j
                        j += 1
                    else:
                        break

                # Remove collinear points within the group
                if len(filtered_corners) > 2:
                    non_collinear_corners = [filtered_corners[0]]
                    for k in range(1, len(filtered_corners) - 1):
                        if not is_collinear(filtered_corners[k-1], filtered_corners[k], filtered_corners[k+1]):
                            non_collinear_corners.append(filtered_corners[k])
                    non_collinear_corners.append(filtered_corners[-1])
                    filtered_corners = non_collinear_corners

                i += 1

            return filtered_corners

        # Find the left-top corner to use as a reference point for sorting
        top_left = np.min(corners, axis=0)
        center = np.mean(corners, axis=0)
        min_x = min(corners, key=lambda c:c[0])[0]
        bottom_left = max((c for c in corners if c[0] == min_x), key=lambda c: c[1])
        # print(f"[sort_and_remove_collinear] top_left = {corners}")


        # corners.sort(key=lambda c: np.arctan2(c[1] - center[1], c[0] - center[0]))
        corners.sort(key=lambda c: np.arctan2(c[1] - top_left[1], c[0] - top_left[0]))
        # Print sorted corners for debugging
        print(f"[sort_and_remove_collinear] sorted_corners = {corners}")

        # filtered_corners = filter_collinear(corners) # 이미 filter되었음

        # Print filtered corners for debugging
        # print(f"[sort_and_remove_collinear] filtered_corners = {filtered_corners}")

        # return filtered_corners
        return corners


    def get_polygon_corners3(self, room_number):
        cell_corners = self.get_cell_corners(room_number)
        corner_count = defaultdict(int)

        for cell in cell_corners:
            for corner in cell:
                corner_count[corner] += 1

        outer_corners = [corner for corner, count in corner_count.items() if count < 4]

        corners = self.sort_and_remove_collinear(outer_corners)

        return corners





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

    def modify_room_boundary(self, room_number, edge_index, distance):
        target_polygon = self.room_polygons[room_number]

        # Deep copy of the original polygon before the move
        original_polygon = RoomPolygon(target_polygon.corners.copy())

        # Move the edge of the target polygon
        target_polygon.move_edge(edge_index, distance)

        # Update the room polygons in the grid
        self.room_polygons[room_number] = target_polygon

        # Adjust other polygons if they share the moved edge or corner
        for other_room_number, other_polygon in self.room_polygons.items():
            if other_room_number != room_number:
                shared_corners = self.get_shared_corners(original_polygon, other_polygon)
                if shared_corners:
                    target_polygon.update_shared_edges(other_polygon, shared_corners) # shared_corner에 대한 정보를 주면 정확하게 계산하지 않을까. 파라메터로 target_polygon.update_shared_edge(other_polygon, shared_corners)
                    self.room_polygons[other_room_number] = other_polygon

    def get_shared_corners(self, polygon1, polygon2):
        shared_corners = set(polygon1.corners).intersection(polygon2.corners)
        return list(shared_corners)

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
                   polygon1.corners[(i + 1) % len(polygon1.corners)] == polygon2.corners[(j + 1) % len(polygon2.corners)]:
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
            reference_point = room_polygon.get_reference_point()
            distances = room_polygon.get_distances_to_boundary(reference_point, combined_polygon)
            distances_m = {direction: distance for direction, distance in distances.items()}
            print(f"Room {room_number}: Reference Point {reference_point}, Distances {distances_m}")
        else:
            print(f'Room {room_number} does not meet the size constraint')

def save_files(polygon_exporter, room_polygons, combined_polygon, suffix):
    polygon_exporter.save_room_polygons_to_dxf(room_polygons, f"room_polygons_{suffix}.dxf")
    polygon_exporter.save_polygon_to_dxf(combined_polygon, f"combined_polygon_{suffix}.dxf")
    polygon_exporter.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")
    polygon_exporter.save_polygon_to_png_with_dimensions(room_polygons, f"room_polygons_with_dimensions_{suffix}.png")

