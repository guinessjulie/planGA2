# RoomPolygon 액세스 단계

- main_ui > build_polygon
  - grid_polygon = `GridPolygon(floorplan)`
    - `def __init__(self)`
      - self.room_polygons = self.`get_all_room_polylgons()`
        - room_polygons[room_number] = `RoomPolygon`(self.get_polygon_corners(room_number))
          - class `RoomPolygon`:
            - def `__init__(self, corners)`
              - self.corners = `corners`
              - @corners.setter
              - def corners(self, new_corners):
                - `self.polygon = Polygon(new_corners)`
                - `self.area = calc_area()`
                - self.perimeter = calc_perimeter()
                - self.min_length = calc_min_length() 
                - self.simplicity = calc_simplicity()

# Fitness Call From Floorplan 
- `Floorplan class` > `Fitness` > `get_fitness`
- `def get_fitness(self)`
  - self.fit = `Fitness(floorplan, num_rooms, self.room_polygons)` <= build_polygon에서 생성
  - class Fitness:`
    - `__init__(self, floorplan, numrooms, room_polygons)` 
      - `self.room_polygons = room_polygons`
# To modify room's edge using GridPolygon Class
- use `grid_polygon.modify_room_boundary(room_number, edge_index, distance)`
- look at `test_move_edge.py` file
- 