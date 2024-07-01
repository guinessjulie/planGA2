from GridPolygon import GridPolygon
from PolygonExporter import PolygonExporter

import numpy as np
from GridPolygon import GridPolygon
from PolygonExporter import PolygonExporter
from RoomPolygon import RoomPolygon
from main import get_floorplan

# Function to print room metrics
def test_adjust_polygon_edges():
    grid = np.array([
        [4, 4, 4, 4, 1, -1, -1],
        [2, 2, 4, 1, 1, 5, 5],
        [2, 2, 3, 3, 1, 5, -1],
        [-1, -1, 3, 3, -1, -1, -1]
    ])

    grid_polygon = GridPolygon(grid, min_area=2000000, min_length=2000)
    polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)
    room_polygons = grid_polygon.get_all_room_polygons()
    combined_polygon = grid_polygon.get_combined_polygon()

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

    save_files(polygon_exporter, room_polygons, combined_polygon, "before")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "Before")

    grid_polygon.modify_room_boundary(room_number=4, edge_index=1, distance=500)

    room_polygons = grid_polygon.room_polygons
    combined_polygon = grid_polygon.get_combined_polygon()

    save_files(polygon_exporter, room_polygons, combined_polygon, "after")
    print_room_metrics(room_polygons, combined_polygon, grid_polygon, "After")

def test_move_edge(room_number = 4, edge_index = 1, distance = 100):
    grid = np.array([
        [4, 4, 4, 4, 1, -1, -1],
        [2, 2, 4, 1, 1, 5, 5],
        [2, 2, 3, 3, 1, 5, -1],
        [-1, -1, 3, 3, -1, -1, -1]
    ])
#    grid = [
#        [2, 2, 2, 5, 3, -1, -1],
#        [2, 2, 4, 5, 3, 3, 3],
#        [2, 2, 4, 5, 5, 3, -1],
#        [-1, -1, 1, 1, -1, -1, -1]
#    ]

    grid_polygon = GridPolygon(grid, min_area=2000000, min_length=2000)
    polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)

    suffix = 'before'
    room_polygons = grid_polygon.get_all_room_polygons()
    polygon_exporter.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")


    suffix = 'after' + str(edge_index)
    grid_polygon.modify_room_boundary(room_number, edge_index, distance=distance)
    room_polygons = grid_polygon.room_polygons

    polygon_exporter.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")
    polygon_exporter.save_polygon_to_png_with_dimensions(room_polygons, f"room_polygons_with_dimensions_{suffix}.png")

if __name__ == '__main__':
#    test_adjust_polygon_edges()
    # 방번호는 부터 시작함
    test_move_edge(room_number = 1, edge_index=0)
    test_move_edge(room_number = 1, edge_index=1)
    test_move_edge(room_number = 1, edge_index=2)
    test_move_edge(room_number = 1, edge_index=3)
