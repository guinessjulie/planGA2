import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
from matplotlib import cm
import RoomPolygon
from typing import List, Tuple, Union
class PolygonExporter:
    def __init__(self, cell_size, grid_shape, padding_size=1000):
        self.cell_size = cell_size
        self.grid_shape = grid_shape
        self.padding_size = padding_size
        self.colors = self.generate_colors()

    def apply_padding(self, corners):
        return [(x + self.padding_size, y + self.padding_size) for x, y in corners]

    def flip_y_axis_with_padding(self, corners):
        padded_corners = self.apply_padding(corners)
        max_y = self.cell_size * self.grid_shape[0] + 2 * self.padding_size
        return [(x, max_y - y) for x, y in padded_corners]

    def generate_colors(self, num_colors=100):
        """Generate a list of distinct colors using a colormap."""
        colormap = cm.get_cmap('tab20', num_colors)
        colors = [colormap(i) for i in range(num_colors)]
        return colors

    def get_color(self, index):
        """Get a fixed color from the list."""
        return self.colors[index % len(self.colors)]

    def save_polygon_to_dxf(self, polygon, filename):
        padded_polygon = self.apply_padding(polygon)
        flipped_polygon = [(x, self.cell_size * (self.grid_shape[0] + 1) - y) for x, y in padded_polygon]
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        polyline = msp.add_lwpolyline(flipped_polygon, close=True)
        polyline.dxf.layer = "All_Rooms"

        doc.saveas(filename)

    def save_room_polygons_to_dxf(self, room_polygons, filename):
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        for room_number, room_polygon in room_polygons.items():
            corners = self.apply_padding(room_polygon.corners)
            flipped_corners = [(x, self.cell_size * (self.grid_shape[0] + 1) - y) for x, y in corners]
            polyline = msp.add_lwpolyline(flipped_corners, close=True)
            polyline.dxf.layer = f"Room_{room_number}"

        doc.saveas(filename)

    def save_polygon_to_png(self, polygons, filename):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # 폴리건 그리기
        for room_number, room_polygon in polygons.items():
            polygon = self.apply_padding(room_polygon.corners)
            color = (random.random(), random.random(), random.random())  # 무작위 색상 선택
            poly = patches.Polygon(polygon, closed=True, edgecolor='black', facecolor=color, alpha=0.5)
            ax.add_patch(poly)

            # 폴리건의 중심 좌표 계산
            center_x = sum([point[0] for point in polygon]) / len(polygon)
            center_y = sum([point[1] for point in polygon]) / len(polygon)

            # 룸 번호 기재
            ax.text(center_x, center_y, str(room_number), fontsize=12, ha='center', va='center', color='black')

        # 전체 그리드 설정
        rows, cols = self.grid_shape
        ax.set_xlim(0, cols * self.cell_size + 2 * self.padding_size)
        ax.set_ylim(0, rows * self.cell_size + 2 * self.padding_size)
        ax.invert_yaxis()  # y축 뒤집기 (상하 반전)

        plt.axis('off')  # 축 숨기기
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


    def determine_boundary_location(self,room, rotation, x1, y1) -> str:
        min_x, max_x, min_y, max_y = room.get_min_max_coordinates()
        ((min_x,min_y), (max_x, max_y))= self.apply_padding([(min_x, min_y), (max_x, max_y)])
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)
        if rotation == 'vertical':
            if x1 <= center_x: # x1 == x2
                return 'west'
            else: return 'east'
        else:
            if y1 <= center_y:
                return 'north'
            else: return 'south'

    def detemine_edge_rotation(self,x1, y1, x2, y2):
        # print(f'x1,y1,x2,yw={x1}.{y1},{x2},{y2}')
        if x1 == x2:  # 세로 방향
            return 'vertical'
        elif y1== y2:
            return 'horizontal'


    def save_polygon_to_png_with_dimensions(self, polygons, filename, show_area=True,dpi=300):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # 폴리건 그리기
        for room_number, room_polygon in polygons.items():
            polygon = self.apply_padding(room_polygon.corners)
            color = (random.random(), random.random(), random.random())  # 무작위 색상 선택
            poly = patches.Polygon(polygon, closed=True, edgecolor='black', facecolor=color, alpha=0.5,
                                   label=f'Room {room_number}')
            ax.add_patch(poly)

            # 폴리건의 중심 좌표 계산
            center_x = sum([point[0] for point in polygon]) / len(polygon)
            center_y = sum([point[1] for point in polygon]) / len(polygon)

            # 룸 번호 기재
            ax.text(center_x, center_y, str(room_number), fontsize=12, ha='center', va='center', color='black')

            # 면적 표시 (show_area가 True일 경우)
            if show_area:
                area = room_polygon.calculate_area() / 1000000  # mm^2에서 m^2로 변환
                ax.text(center_x, center_y + 200, f'{area:.2f} m²', fontsize=10, ha='center', va='center', color='red')

        # 전체 그리드 설정
        rows, cols = self.grid_shape
        ax.set_xlim(0, cols * self.cell_size + 2 * self.padding_size)
        ax.set_ylim(0, rows * self.cell_size + 2 * self.padding_size)
        ax.invert_yaxis()  # y축 뒤집기 (상하 반전)

        # 치수 표시
        offset = 100
        for room_number, room_polygon in polygons.items():
            polygon = room_polygon.corners
            for i in range(len(polygon)):
                x1, y1 = self.apply_padding([polygon[i]])[0]
                x2, y2 = self.apply_padding([polygon[(i + 1) % len(polygon)]])[0]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000  # mm to m
                rotation = self.detemine_edge_rotation(x1, y1, x2, y2)

                if rotation == 'vertical':
                    direction = self.determine_boundary_location(room_polygon, rotation, x1, y1)
                    if direction == 'east':
                        offset_x, offset_y = offset, 0
                    else:  # 서쪽
                        offset_x, offset_y = -offset, 0

                else:  # rotation == 'horizontal'
                    direction = self.determine_boundary_location(room_polygon, rotation, x1, y1)
                    if direction == 'north':
                        offset_x, offset_y = 0, -offset
                    else:  # 'south'
                        offset_x, offset_y = 0, offset

                # print (f'{mid_x + offset_x}, {mid_y + offset_y}, {distance:.1f}', {rotation})
                ax.text (mid_x + offset_x, mid_y + offset_y, f'{distance:.1f}', fontsize=8, ha='center', va='center',
                        color='blue', rotation=rotation)

        plt.axis('off')  # 축 숨기기
        ax.legend(loc='upper right')  # 범례 추가
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
