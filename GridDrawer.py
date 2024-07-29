from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
import numpy as np
import textwrap
from constants import UNIT_SCALE
class GridDrawer:

    @staticmethod
    def draw_grid_reversed(grid, savepath):
        print(grid)
        # 이미지 객체 생성
        im = Image.new('RGB', (150, 150), (255, 255, 255))
        draw = ImageDraw.Draw(im)

        # s 내의 각 좌표에 사각형 그리기
        for x, y in grid:
            # x, y 좌표에 따라 사각형 그리기 위치 계산
            rect_start_x = x * 20 + 10
            rect_start_y = y * 20 + 10
            rect_end_x = (x + 1) * 20 + 10
            rect_end_y = (y + 1) * 20 + 10
            draw.rectangle([rect_start_x, rect_start_y, rect_end_x, rect_end_y],
                           fill=(192, 192, 192), outline=None, width=0)
            # x, y 좌표에 따라 텍스트 위치 계산 및 텍스트 추가
            text_position_x = rect_start_x + 5
            text_position_y = rect_start_y + 5
            draw.text((text_position_x, text_position_y), f"{x}{y}", fill=(0, 0, 0))

        # 이미지 보기
        im.show()

        # 이미지 파일로 저장
        im.save(savepath)

    @staticmethod
    def draw_grid(coords, outfile):
        """
        주어진 좌표 리스트를 사용하여 사각형과 좌표 텍스트를 그린 이미지를 생성하고 저장하는 함수.
        parameters:
        coords (list of tuples):그리기를 원하는 좌표의 리스트. 각 튜플은 (행, 열) 형식.
        outfile (str): 생성된 이미지를 저장할 파일 이름.
        """
        # 이미지 객체 생성
        im = Image.new('RGB', (150, 150), (255, 255, 255))
        draw = ImageDraw.Draw(im)

        # s 내의 각 좌표에 사각형 그리기 (첫 번째 원소를 행으로, 두 번째 원소를 열로)
        for row, col in coords:
            # row, col 좌표에 따라 사각형 그리기 위치 계산
            rect_start_x = col * 20 + 10
            rect_start_y = row * 20 + 10
            rect_end_x = (col + 1) * 20 + 10
            rect_end_y = (row + 1) * 20 + 10
            draw.rectangle([rect_start_x, rect_start_y, rect_end_x, rect_end_y],
                           fill=(192, 192, 192), outline=None, width=0)
            # row, col 좌표에 따라 텍스트 위치 계산 및 텍스트 추가
            text_position_x = rect_start_x + 5
            text_position_y = rect_start_y + 5
            draw.text((text_position_x, text_position_y), f"{row}{col}", fill=(0, 0, 0))

        # 이미지 보기
        im.show()

        # 이미지 파일로 저장
        im.save(outfile)

    @staticmethod
    def draw_plan_padded(grid):

        # Define the original grid structure
        # original_grid = np.array([
        #     [2, 2, 5, 5, 5, -1, -1],
        #     [2, 2, 4, 4, 1, 3, 3],
        #     [2, 2, 4, 4, 1, 1, -1],
        #     [-1, -1, 4, 4, -1, -1, -1]
        # ])
        original_grid = grid
        # Add a border of -1 around the original grid
        padded_grid = np.pad(original_grid, pad_width=1, mode='constant', constant_values=-1)

        # Define the scale
        scale = 1000  # 1 unit = 1000mm
        wall_thickness = 5  # Wall thickness in mm

        # Create a plot
        fig, ax = plt.subplots()

        # Get the unique room identifiers and their corresponding colors
        rooms = np.unique(padded_grid)
        colors = {
            0: 'white', 1: 'red', 2: 'blue', 3: 'yellow',
            4: 'cyan', 5: 'green', -1: 'gray'
        }

        # Plot each room with the corresponding color
        for room in rooms:
            if room != -1:  # Ignore external space
                room_coords = np.argwhere(padded_grid == room)
                for coord in room_coords:
                    y, x = coord
                    rect = plt.Rectangle((x * scale, y * scale), scale, scale, facecolor=colors[room],
                                         edgecolor='black', linewidth=0)
                    ax.add_patch(rect)

        # Add thick borders for each room
        for y in range(padded_grid.shape[0]):
            for x in range(padded_grid.shape[1]):
                if padded_grid[y, x] != -1:
                    if y == 0 or padded_grid[y - 1, x] != padded_grid[y, x]:  # Top border
                        ax.plot([x * scale, (x + 1) * scale], [y * scale, y * scale], color='black',
                                linewidth=wall_thickness)
                    if y == padded_grid.shape[0] - 1 or padded_grid[y + 1, x] != padded_grid[y, x]:  # Bottom border
                        ax.plot([x * scale, (x + 1) * scale], [(y + 1) * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == 0 or padded_grid[y, x - 1] != padded_grid[y, x]:  # Left border
                        ax.plot([x * scale, x * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == padded_grid.shape[1] - 1 or padded_grid[y, x + 1] != padded_grid[y, x]:  # Right border
                        ax.plot([(x + 1) * scale, (x + 1) * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)

        # Set labels and grid
        ax.set_xticks(np.arange(0, (padded_grid.shape[1] + 1) * scale, scale))
        ax.set_yticks(np.arange(0, (padded_grid.shape[0] + 1) * scale, scale))
        ax.set_xticklabels(np.arange(0, padded_grid.shape[1] + 1))
        ax.set_yticklabels(np.arange(0, padded_grid.shape[0] + 1))
        plt.xlabel('X (1000 mm units)')
        plt.ylabel('Y (1000 mm units)')
        ax.grid(False)  # Disable the grid

        # Add room labels
        for room in rooms:
            if room > 0:
                room_coords = np.argwhere(padded_grid == room)
                area_sqm = len(room_coords) * (scale / 1000) ** 2  # Calculate area in square meters
                y_mean = np.mean(room_coords[:, 0]) * scale + scale / 2
                x_mean = np.mean(room_coords[:, 1]) * scale + scale / 2
                ax.text(x_mean, y_mean, f'Room {room}\n{area_sqm:.1f} m²', color='black', weight='bold',
                        ha='center', va='center', fontsize=12)

        # Adjust plot limits to account for wall thickness
        ax.set_xlim(-wall_thickness * 2, padded_grid.shape[1] * scale + wall_thickness * 2)
        ax.set_ylim(-wall_thickness * 2, padded_grid.shape[0] * scale + wall_thickness * 2)

        # Display the plot
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_colored_grid2(grid, filename):
        """
        display a colored grid regarding numbers of the cells.

        Parameters:
        - m (int): The number of rows in each grid.
        - n (int): The number of columns in each grid.
        - k (int): The number of color groups in each grid.
        """

        plt.figure()
        colors = mpl.colormaps['Accent']
        plt.matshow(grid, cmap=colors)
        plt.axis('off')
        plt.savefig(filename)
        plt.show()

    def draw_plan(floorplan):
        # Define the new grid structure
        if floorplan is None:
            grid = np.array([
                [2, 2, 5, 5, 5, -1, -1],
                [2, 2, 4, 4, 1, 3, 3],
                [2, 2, 4, 4, 1, 1, -1],
                [-1, -1, 4, 4, -1, -1, -1]
            ])
        else:
            grid = floorplan
        # Define the scale
        scale = UNIT_SCALE  # 1 unit = 1000mm
        wall_thickness = 5  # Wall thickness in mm

        # Create a plot
        fig, ax = plt.subplots()

        # Get the unique room identifiers and their corresponding colors
        rooms = np.unique(grid)
        colors = {
            0: 'white', 1: 'red', 2: 'blue', 3: 'yellow',
            4: 'cyan', 5: 'green', -1: 'gray'
        }

        # Plot each room with the corresponding color
        for room in rooms:
            if room != -1:  # Ignore external space
                room_coords = np.argwhere(grid == room)
                for coord in room_coords:
                    y, x = coord
                    rect = plt.Rectangle((x * scale, y * scale), scale, scale, facecolor=colors[room],
                                         edgecolor='black', linewidth=0)
                    ax.add_patch(rect)

        # Add thick borders for each room
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] != -1:
                    if y == 0 or grid[y - 1, x] != grid[y, x]:  # Top border , 다음 row와 같은 색이 아닐 경우
                        ax.plot([x * scale, (x + 1) * scale], [y * scale, y * scale], color='black',
                                linewidth=wall_thickness)
                    if y == grid.shape[0] - 1 or grid[y + 1, x] != grid[y, x]:  # Bottom border
                        ax.plot([x * scale, (x + 1) * scale], [(y + 1) * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == 0 or grid[y, x - 1] != grid[y, x]:  # Left border
                        ax.plot([x * scale, x * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == grid.shape[1] - 1 or grid[y, x + 1] != grid[y, x]:  # Right border
                        ax.plot([(x + 1) * scale, (x + 1) * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    # edge가 얇게 나오는 현상을 픽스
                    # if y == 0:
                    #     ax.plot([x * scale, (x + 1) * scale], [y * scale, y * scale], color='black',
                    #             linewidth=wall_thickness * 2)
                    # if x == 0 :
                    #     ax.plot([x * scale, x * scale], [y * scale, (y + 1) * scale], color='black',
                    #                 linewidth=wall_thickness*2)
                    # if y == grid.shape[0] - 1 :# Bottom border
                    #     ax.plot([x * scale, (x + 1) * scale], [(y + 1) * scale, (y + 1) * scale], color='black',
                    #             linewidth=wall_thickness*2)
                    # if x == grid.shape[1] - 1:  # Right border
                    #     ax.plot([(x + 1) * scale, (x + 1) * scale], [y * scale, (y + 1) * scale], color='black',
                    #             linewidth=wall_thickness*2)

        # Set labels and grid
        ax.set_xticks(np.arange(0, (grid.shape[1] + 1) * scale, scale))
        ax.set_yticks(np.arange(0, (grid.shape[0] + 1) * scale, scale))
        ax.set_xticklabels(np.arange(0, grid.shape[1] + 1))
        ax.set_yticklabels(np.arange(0, grid.shape[0] + 1))
        plt.xlabel('X (1000 mm units)')
        plt.ylabel('Y (1000 mm units)')
        ax.grid(False)  # Disable the grid

        # Add room labels
        for room in rooms:
            if room > 0:
                room_coords = np.argwhere(grid == room)
                area_sqm = len(room_coords) * (scale / 1000) ** 2  # Calculate area in square meters
                y_mean = np.mean(room_coords[:, 0]) * scale + scale / 2
                x_mean = np.mean(room_coords[:, 1]) * scale + scale / 2
                ax.text(x_mean, y_mean, f'Room {room}\n{area_sqm:.1f} m²', color='black', weight='bold',
                        ha='center', va='center', fontsize=12)

        # Adjust plot limits to account for wall thickness
        ax.set_xlim(-wall_thickness, grid.shape[1] * scale + wall_thickness)
        ax.set_ylim(-wall_thickness, grid.shape[0] * scale + wall_thickness)

        # Display the plot
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def draw_plan_equal_thickness(grid):
        # Define the new grid structure
        grid = np.array([
            [2, 2, 5, 5, 5, -1, -1],
            [2, 2, 4, 4, 1, 3, 3],
            [2, 2, 4, 4, 1, 1, -1],
            [-1, -1, 4, 4, -1, -1, -1]
        ])

        # Define the scale
        scale = 1000  # 1 unit = 1000mm
        wall_thickness = 5  # Wall thickness in mm

        # Create a plot
        fig, ax = plt.subplots()

        # Get the unique room identifiers and their corresponding colors
        rooms = np.unique(grid)
        colors = {
            0: 'white', 1: 'red', 2: 'blue', 3: 'yellow',
            4: 'cyan', 5: 'green', -1: 'gray'
        }

        # Plot each room with the corresponding color
        for room in rooms:
            if room != -1:  # Ignore external space
                room_coords = np.argwhere(grid == room)
                for coord in room_coords:
                    y, x = coord
                    rect = plt.Rectangle((x * scale, y * scale), scale, scale, facecolor=colors[room], edgecolor='black',
                                         linewidth=0)
                    ax.add_patch(rect)

        # Add thick borders for each room
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] != -1:
                    if y == 0 or grid[y - 1, x] != grid[y, x]:  # Top border
                        ax.plot([x * scale, (x + 1) * scale], [y * scale, y * scale], color='black',
                                linewidth=wall_thickness)
                    if y == grid.shape[0] - 1 or grid[y + 1, x] != grid[y, x]:  # Bottom border
                        ax.plot([x * scale, (x + 1) * scale], [(y + 1) * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == 0 or grid[y, x - 1] != grid[y, x]:  # Left border
                        ax.plot([x * scale, x * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)
                    if x == grid.shape[1] - 1 or grid[y, x + 1] != grid[y, x]:  # Right border
                        ax.plot([(x + 1) * scale, (x + 1) * scale], [y * scale, (y + 1) * scale], color='black',
                                linewidth=wall_thickness)

        # Set labels and grid
        ax.set_xticks(np.arange(0, (grid.shape[1] + 1) * scale, scale))
        ax.set_yticks(np.arange(0, (grid.shape[0] + 1) * scale, scale))
        ax.set_xticklabels(np.arange(0, grid.shape[1] + 1))
        ax.set_yticklabels(np.arange(0, grid.shape[0] + 1))
        plt.xlabel('X (1000 mm units)')
        plt.ylabel('Y (1000 mm units)')
        ax.grid(False)  # Disable the grid

        # Add room labels
        for room in rooms:
            if room > 0:
                room_coords = np.argwhere(grid == room)
                area_sqm = len(room_coords) * (scale / 1000) ** 2  # Calculate area in square meters
                y_mean = np.mean(room_coords[:, 0]) * scale + scale / 2
                x_mean = np.mean(room_coords[:, 1]) * scale + scale / 2
                ax.text(x_mean, y_mean, f'Room {room}\n{area_sqm:.1f} m²', color='black', weight='bold',
                        ha='center', va='center', fontsize=12)

        # Adjust plot limits to account for wall thickness
        ax.set_xlim(-wall_thickness * 2, grid.shape[1] * scale + wall_thickness * 2)
        ax.set_ylim(-wall_thickness * 2, grid.shape[0] * scale + wall_thickness * 2)

        # Display the plot
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_colored_grid(grid, filename):
        """
        display a colored grid regarding numbers of the cells.

        Parameters:
        - m (int): The number of rows in each grid.
        - n (int): The number of columns in each grid.
        - k (int): The number of color groups in each grid.
        """
        # Get unique positive values in the grid
        unique_values = sorted(set(np.array(grid).flatten()))

        cmap = plt.cm.jet
        colors = cmap(np.linspace(0, 1, cmap.N))
        # Replace the first color with white
        colors[0] = [1, 1, 1, 1]

        # Create a custom colormap
        custom_cmap = ListedColormap(colors, name='custom_cmap')



        # Save Plot the colored grid
        plt.figure(figsize=(8,6))
        plt.matshow(grid, cmap=custom_cmap, extent=[0, len(grid[0]), 0, len(grid)])
        # Display colorbar
        cbar = plt.colorbar(ticks=unique_values, extend='min')
        cbar.set_ticklabels(['-1: NONE', '0: OUT','1:LV', '2:KT', '3:BED1', '4:BED2', '5:BATH' ])
        cbar.set_label('Space')
        plt.axis('off')

        # Show the plot
        plt.savefig(filename)
        plt.show()

    @ staticmethod


    @staticmethod
    def color_cells_by_value(grid, filename, text=None):
        def text_align(text):
            text_list = text.split('\n')
            short_text = [s for s in text_list if len(s) < 80]
            long_text = [s for s in text_list if len(s) >= 80][0] if len(
                [s for s in text_list if len(s) >= 80]) > 0 else ''
            wrapped_text = textwrap.wrap(long_text, width=80)
            aligned_text_list = short_text + wrapped_text
            aligned_text = '\n'.join(aligned_text_list)
            return aligned_text

        # 각 정수 값에 대응하는 RGB 색상 정의
        colors = np.array([(1, 1, 1),  # -1: 검정
                           (1, 1, 1),  # 0 : 흰색
                           (1, 0, 0),  # 빨간색
                           (0, 1, 0),  # 초록색
                           (0, 0, 1),  # 파란색
                           (1, 1, 0),  # 노란색
                           (0, 1, 1),  # 청록색
                           ])

        cmap = ListedColormap(colors)

        # 데이터 값의 범위에 따른 경계값 설정 (여기서는 -1에서 5까지)
        boundaries = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(boundaries, cmap.N)

        # matshow로 데이터 시각화
        fig, ax = plt.subplots()
        cax = ax.matshow(grid, cmap=cmap, norm=norm)
        # fig.colorbar(cax, ticks=range(7))
        # todo d위에 텍스트를 적기 위해 아래쪽으로 컬러바를 옮겼다. orientation과 shirink옵션 추가됨. 나중에 필요하면 제거
        colorbar = fig.colorbar(cax, shrink=0.8,ticks=[-1, 0, 1, 2, 3, 4, 5])
        colorbar.set_ticklabels(['OUT','OUT','1', '2', '3', '4', '5'])
        # tick label의 크기와 방향 조정

        # 하단 여백을 조정하여 텍스트를 추가할 공간 확보
        plt.subplots_adjust(bottom = 0.2)
        if text:
            aligned_text = text_align(text)
            plt.text(0.5, 0.02, aligned_text, ha='center', fontsize=10, transform=fig.transFigure)

        # # 하단 여백을 조정하여 텍스트를 추가할 공간 확보
        # plt.subplots_adjust(bottom=0.2)
        #
        # if text:
        #     plt.figtext(0.5, -0.1, text, ha='center', fontsize=12)

        plt.savefig(filename)
        print(f'{filename} with {text} saved')
        plt.show()

    @staticmethod
    def plot_grids(grids, m, n, k, rows=None, cols=5, fitness_scores=None, savefilename=None):
        """
        Creates a plot with multiple subplots arranged in specified rows and columns, each displaying a grid.
        cols가 디폴트인 경우 cols=5, 2x5 10 개의 서브 플롯을 그린다.
        Parameters:
        - m (int): The number of rows in each grid.
        - n (int): The number of columns in each grid.
        - k (int): The number of color groups in each grid.
        - rows (int): The number of subplot rows.
        - cols (int): The number of subplot columns.
        """
        if rows is None:
            cols = int(cols)
            rows = int(ceil(len(grids) / cols))
        if cols is None:
            rows = int(rows)
            cols = int(ceil(len(grids) / rows))
        rows, cols = int(rows), int(cols)
        fig, axes = plt.subplots(rows, cols, figsize=(n * cols, m * rows))
        xy_coords = [(x, y) for x in range(rows) for y in range(cols)]
        for ij, grid in enumerate(grids):
            colors = mpl.colormaps['Accent']
            ax = axes[xy_coords[ij]]
            ax.matshow(grid, cmap=colors)
            print(grid)
            if fitness_scores is not None:
                ax.set_title(f'{fitness_scores[ij]:.2f}', fontsize=10)
            ax.axis('off')

        if savefilename is not None:
            plt.savefig(savefilename)
        plt.show()
