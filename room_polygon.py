import math
from shapely.geometry import Polygon, LineString

class RoomPolygon:
    def __init__(self, corners):
        self.corners = corners
        self.polygon = Polygon(corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()

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

    def calculate_simplicity(self):
        return (16 * self.area) / (self.perimeter ** 2.0)

    def check_constraints(self, min_area=None, min_length=None):
        if min_area and self.area < min_area:
            print(f'constraint min_area={min_area}, area={self.area}')
            return False
        if min_length and self.min_length < min_length:
            print(f'constraint min_length={min_length}, min_length={self.min_length}')
            return False
        return True

    def get_distances_to_boundary(self, reference_point, boundary_polygon):
        poly = Polygon(boundary_polygon)

        vertical_line = LineString([(reference_point[0], min(boundary_polygon, key=lambda x: x[1])[1]),
                                    (reference_point[0], max(boundary_polygon, key=lambda x: x[1])[1])])
        horizontal_line = LineString([(min(boundary_polygon, key=lambda x: x[0])[0], reference_point[1]),
                                      (max(boundary_polygon, key=lambda x: x[0])[0], reference_point[1])])

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

        if north_dist == float('inf'):
            north_dist = max(boundary_polygon, key=lambda x: x[1])[1] - reference_point[1]
        if south_dist == float('inf'):
            south_dist = reference_point[1] - min(boundary_polygon, key=lambda x: x[1])[1]
        if east_dist == float('inf'):
            east_dist = max(boundary_polygon, key=lambda x: x[0])[0] - reference_point[0]
        if west_dist == float('inf'):
            west_dist = reference_point[0] - min(boundary_polygon, key=lambda x: x[0])[0]

        return {
            'North': north_dist / 1000,
            'South': south_dist / 1000,
            'East': east_dist / 1000,
            'West': west_dist / 1000
        }

    def move_edge(self, edge_index, distance):
        """
        Move an edge of the polygon in a specified direction by a certain distance.

        :param edge_index: The index of the edge to move.
        :param distance: The distance to move the edge.
        """
        new_corners = self.corners.copy()
        x1, y1 = new_corners[edge_index]
        x2, y2 = new_corners[(edge_index + 1) % len(new_corners)]

        if x1 == x2:  # Vertical edge
            direction = 'horizontal'
        elif y1 == y2:  # Horizontal edge
            direction = 'vertical'
        else:
            raise ValueError("Edge is not perfectly vertical or horizontal.")

        if direction == 'vertical':
            for i in range(len(new_corners)):
                if i == edge_index or i == (edge_index + 1) % len(new_corners):
                    new_corners[i] = (new_corners[i][0], new_corners[i][1] + distance)
        elif direction == 'horizontal':
            for i in range(len(new_corners)):
                if i == edge_index or i == (edge_index + 1) % len(new_corners):
                    new_corners[i] = (new_corners[i][0] + distance, new_corners[i][1])

        self.corners = new_corners
        self.polygon = Polygon(new_corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()

    def get_shared_corners(self, other_polygon):
        shared_corners = set(self.corners).intersection(other_polygon.corners)
        return list(shared_corners)

    def update_shared_edges(self, other_polygon, shared_corners):
        for corner in shared_corners:
            idx_self = self.corners.index(corner)
            idx_other = other_polygon.corners.index(corner)

            if idx_self != -1 and idx_other != -1:
                self.move_corner(idx_self, corner)
                other_polygon.move_corner(idx_other, corner)

    def move_corner(self, corner_index, new_corner):
        self.corners[corner_index] = new_corner
        self.polygon = Polygon(self.corners)
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.min_length = self.calculate_min_length()
        self.simplicity = self.calculate_simplicity()
