import math
from shapely.geometry import LineString, Point, Polygon, MultiPoint, MultiLineString

class BoundingBox:
    def __init__(self, corners):
        self.min_x = min(point[0] for point in corners)
        self.max_x = max(point[0] for point in corners)
        self.min_y = min(point[1] for point in corners)
        self.max_y = max(point[1] for point in corners)

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y

    @property
    def area(self):
        return self.width * self.height

    @property
    def as_ratio(self):
        return self.width / self.height if self.height >= self.width else self.height / self.width


class RoomPolygon:
    def __init__(self, corners, room_id = None): # todo store room_id to debug
        self._corners = corners
        self.room_id = room_id
        self.bb = BoundingBox(corners)
        self.moved_corners = []  # 이동된 코너들을 추적합니다.
        self.polygon = Polygon(corners)  # 초기화시 다각형 설정
        self.area = self.calculate_area()  # todo only to compare with polylgon's area and perimeters
        self.perimeter = self.calculate_perimeter()  # todo only to compare with polylgon's area and perimeters
        self.min_length, max_length = self.calculate_min_max_length()
        self.simplicity = self.calc_simplicity()
        self.rectangularity =  self.calc_rectangularity()
        self.regularity = self.calc_regularity()
        self.pa_ratio = self.calc_pa_ratio()

    @property
    def corners(self):
        return self._corners

    @corners.setter
    def corners(self, new_corners):
        self._corners = new_corners
        self.bb = BoundingBox(new_corners)
        self.polygon = Polygon(new_corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length, max_length = self.calculate_min_max_length()
        self.simplicity = self.calc_simplicity()
        self.rectangularity =  self.calc_rectangularity()
        self.regularity = self.calc_regularity()
        self.pa_ratio= self.calc_pa_ratio()

    # underway convert to BoundingBox class

    def calc_simplicity(self): # todo _corners 값이 바뀌어도 실행이 되는지 확인
        vertex_count = len(self._corners)
        if vertex_count < 4:
            return 1.0
        return 4 / vertex_count

    def calc_rectangularity(self): # todo _corners 값이 바뀌어도 실행이 되는지 확인
        if self.bb.area < self.area:
            print(f'something wrong bb.area:{self.bb.area} < area:{self.area}')
        return self.area / self.bb.area if self.bb.area != 0 else float('inf')

    # todo _corners 값이 바뀌어도 실행이 되는지 확인
    def calc_regularity(self):
        return self.rectangularity * self.bb.as_ratio


    def calc_pa_ratio(self):
        return 16 * self.area / self.perimeter ** 2

    def calculate_metrics(self):
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()

    def get_min_max_coordinates(self):
        """
        각 x, y 좌표의 최소값과 최대값을 반환합니다.

        Returns:
            min_x (float): x 좌표의 최소값
            max_x (float): x 좌표의 최대값
            min_y (float): y 좌표의 최소값
            max_y (float): y 좌표의 최대값
        """
        x_coords = [corner[0] for corner in self.corners]
        y_coords = [corner[1] for corner in self.corners]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        return min_x, max_x, min_y, max_y

    def move_edge(self, edge_index, distance):
        new_corners = self.corners.copy()
        x1, y1 = new_corners[edge_index]
        x2, y2 = new_corners[(edge_index + 1) % len(new_corners)]

        if x1 == x2:  # Vertical edge
            direction = 'horizontal'
        elif y1 == y2:  # Horizontal edge
            direction = 'vertical'
        else:
            raise ValueError("Edge is not perfectly vertical or horizontal.")

        self.moved_corners.clear()  # 이동된 코너 목록을 초기화합니다.

        if direction == 'vertical':
            for i in range(len(new_corners)):
                if i == edge_index or i == (edge_index + 1) % len(new_corners):
                    original_corner = new_corners[i]
                    new_corners[i] = (new_corners[i][0], new_corners[i][1] + distance)
                    self.moved_corners.append((original_corner, new_corners[i]))
        elif direction == 'horizontal':
            for i in range(len(new_corners)):
                if i == edge_index or i == (edge_index + 1) % len(new_corners):
                    original_corner = new_corners[i]
                    new_corners[i] = (new_corners[i][0] + distance, new_corners[i][1])
                    self.moved_corners.append((original_corner, new_corners[i]))

        self.corners = new_corners
        self.polygon = Polygon(new_corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()

    def get_intersections(self, moved_edge, other_polygon):
        intersections = []
        moved_line = LineString(moved_edge)
        other_corners = other_polygon.corners  # Accessing corners from the other polygon

        for i in range(len(other_corners)):
            edge = LineString([other_corners[i], other_corners[(i + 1) % len(other_corners)]])
            if moved_line.intersects(edge):
                intersection = moved_line.intersection(edge)
                if isinstance(intersection, Point):
                    intersections.append((intersection.x, intersection.y))
                elif isinstance(intersection, LineString):
                    intersections.extend([(point[0], point[1]) for point in intersection.coords])
                elif isinstance(intersection, MultiPoint):
                    intersections.extend([(point.x, point.y) for point in intersection])
                elif isinstance(intersection, MultiLineString):
                    for geom in intersection:
                        intersections.extend([(point[0], point[1]) for point in geom.coords])
                elif intersection.geom_type == 'GeometryCollection':
                    for geom in intersection:
                        if isinstance(geom, (Point, LineString)):
                            intersections.extend([(point[0], point[1]) for point in geom.coords])
        return intersections

    def insert_point_in_order(self, polygon, point):
        min_distance = float('inf')
        insert_index = -1

        for i in range(len(polygon.corners)):
            p1 = polygon.corners[i]
            p2 = polygon.corners[(i + 1) % len(polygon.corners)]
            line = LineString([p1, p2])
            distance = line.distance(Point(point))

            if distance < min_distance:
                min_distance = distance
                insert_index = i + 1

        polygon.corners.insert(insert_index, point)
        polygon.polygon = Polygon(polygon.corners)

    def sync_all_adjacent_corners(self):
        for corner_index in range(len(self.corners)):
            updated_axis = 'x' if self.corners[corner_index][0] != self.corners[corner_index - 1][0] else 'y'
            self.sync_adjacent_corner(corner_index, updated_axis)

    def update_shared_edges(self, other_polygon, shared_corners):
        original_corners = [tup[0] for tup in self.moved_corners]
        shared_updated = set(shared_corners).intersection(
            set(original_corners))  # shared corner 중 updated된 코너의 이동 전의 좌표 = 작업 대상
        dict_moved_corners = {tup[0]: tup[1] for tup in self.moved_corners}
        for updating_corner_index, original_corner in enumerate(
                shared_updated):  # updating_corner_index = 작업할 대상을 차례로 방문할 때의 index
            if original_corner in other_polygon.corners:
                idx_other = other_polygon.corners.index(original_corner)  # idx_other 번째 코너의 index
                new_corner = dict_moved_corners[original_corner]
                other_polygon.move_corner(idx_other, new_corner)

        self.sync_all_adjacent_corners()
        other_polygon.sync_all_adjacent_corners()
        # 교차점 추가

        # 교차점 추가
        moved_edge = [corner for _, corner in self.moved_corners]
        intersections = self.get_intersections(moved_edge, other_polygon)  # <--- 여기서 에러
        for intersection in intersections:
            if intersection not in self.corners:
                self.insert_point_in_order(self, intersection)
            if intersection not in other_polygon.corners:
                self.insert_point_in_order(other_polygon, intersection)

    def move_corner(self, corner_index, new_corner):
        """
        설명: corner_index 번째 코너를 new_corner의 값으로 이동

        Args:
            corner_index (int): 대상 코너 인덱스.
            new_corner (int, int):  새 코너 좌표
        """
        old_corner = self.corners[corner_index]
        self.corners[corner_index] = new_corner
        updated_axis = 'x' if old_corner[0] != new_corner[0] else 'y'
        self.sync_adjacent_corner(corner_index, updated_axis)
        self.polygon = Polygon(self.corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()

    def is_corner_perpendicular(self, corner_index):
        num_corners = len(self.corners)

        # Get the previous, current, and next corners
        prev_corner = self.corners[(corner_index - 1) % num_corners]
        current_corner = self.corners[corner_index]
        next_corner = self.corners[(corner_index + 1) % num_corners]

        # Calculate vectors
        prev_vector = (current_corner[0] - prev_corner[0], current_corner[1] - prev_corner[1])
        next_vector = (next_corner[0] - current_corner[0], next_corner[1] - current_corner[1])

        # Check for perpendicularity by calculating the dot product
        dot_product = prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1]

        # If the dot product is zero, the vectors are perpendicular
        return dot_product == 0

    def sync_adjacent_corner(self, corner_index, updated_axis):
        num_corners = len(self.corners)

        # Get the previous, current, and next corners
        prev_corner = self.corners[(corner_index - 1) % num_corners]
        current_corner = self.corners[corner_index]
        next_corner = self.corners[(corner_index + 1) % num_corners]

        # Check alignment with previous and next corners
        prev_aligned = (prev_corner[0] == current_corner[0] or prev_corner[1] == current_corner[1])
        next_aligned = (next_corner[0] == current_corner[0] or next_corner[1] == current_corner[1])

        if prev_aligned and next_aligned:
            return True  # Both previous and next corners are aligned

        # Convert tuples to lists to allow modification
        prev_corner = list(prev_corner)
        next_corner = list(next_corner)

        # If not aligned, update the specified axis to match the current corner
        if not prev_aligned:
            if updated_axis == 'x':
                prev_corner[0] = current_corner[0]
            else:
                prev_corner[1] = current_corner[1]

        if not next_aligned:
            if updated_axis == 'x':
                next_corner[0] = current_corner[0]
            else:
                next_corner[1] = current_corner[1]

        # Convert lists back to tuples and update the corners
        self.corners[(corner_index - 1) % num_corners] = tuple(prev_corner)
        self.corners[(corner_index + 1) % num_corners] = tuple(next_corner)

        return False  # At least one corner was not aligned

    def get_shared_corners(self, other_polygon):
        shared_corners = set(self.corners).intersection(other_polygon.corners)
        return list(shared_corners)

    def calculate_area(self):
        x, y = zip(*self.corners)
        return 0.5 * abs(sum(x[i] * y[i - 1] - y[i] * x[i - 1] for i in range(len(self.corners))))

    def calculate_perimeter(self):
        perimeter = 0
        for i in range(len(self.corners)):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % len(self.corners)]
            perimeter += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return perimeter

    def calculate_min_length(self):
        min_length = float('inf')
        for i in range(len(self.corners)):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % len(self.corners)]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            min_length = min(min_length, length)
        return min_length

    def calculate_min_max_length(self):
        min_length = float('inf')
        max_length = float(0)
        for i in range(len(self.corners)):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % len(self.corners)]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            min_length = min(min_length, length)
            max_length = max(max_length, length)
        return min_length, max_length

    def calculate_simplicity(self):
        return (16 * self.area) / (self.perimeter ** 2.0)

    def get_reference_point(self):
        x, y = zip(*self.corners)
        return (sum(x) / len(self.corners), sum(y) / len(self.corners))

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

    def check_constraints(self, min_area, min_length):
        if min_area and self.area < min_area:
            return False
        if min_length and self.min_length < min_length:
            return False
        return True
