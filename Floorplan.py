import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from fitness import Fitness

import trivial_utils
from main import GridDrawer, exchange_protruding_cells, categorize_boundary_cells, GraphBuilder, GraphDrawer, run_selected_module,  exit_module
from simplify import exchange_protruding_cells, count_cascading_cells
from plan import create_floorplan, locate_initial_cell
from GridPolygon import GridPolygon
from options import Options
from PolygonExporter import PolygonExporter
from config_reader import load_config
import configparser
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo create population class
# todo use seed to recreate floorplan
# todo let equal thickness function work customize

class FloorplanApp:
    # todo 0818 1. 동일 seed에서 만들어진 floorplan을 10개 이상 가지고 있어보자.
    # info: done: in self.simplified_candidates have diff floorplans from the same initiialized_floorplan. after choosing best simplified floorplans. assigns to self.floorplan to set final result
    # todo 1-1 : create_floorplan과 simplify를 합치자.
    # todo: Area 및 wall thickness 표시
    def __init__(self, root, init_grid, num_rooms, callback):
        self.root = root
        self.root.title("Floorplan UI")
        self.room_polygons = None

        self.init_grid = init_grid
        self.num_rooms = num_rooms
        self.options = Options()# todo 만들어놓기만 하고 사용 안했음
        self.callback = callback  # Main App으로 floorplan을 반환하기 위한 콜백 함수
        self.floorplan = None
        self.floorplans = []
        self.simplified_candidates = []
        self.simplified_floorplan = None
        self.seed = None
        self.initial_cells = None
        self.fit = None
        self.path = None
        self.option = None
        self.create_widgets()


    def create_widgets(self):
        # 왼쪽 프레임 생성 및 배치 (메뉴 버튼을 위한 프레임)
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 오른쪽 메인 프레임 생성 및 배치 (캔버스와 피트니스 프레임을 포함)
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 오른쪽 메인 프레임의 상단에 두 개의 캔버스를 위한 프레임 배치
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 오른쪽 프레임을 좌우 두 개의 하위 프레임으로 나누기
        left_right_frame = tk.Frame(right_frame)
        left_right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_right_frame = tk.Frame(right_frame)
        right_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 버튼의 너비 설정
        button_width = 20

        # 버튼 리스트
        buttons = [
            ("Initial Room Location", self.initialize_room_location),
            ('Create Floorplan', self.initialize_floorplan),
            ("Simplify Floorplan", self.exchange_cells),
            ("Choose Most Simplified", self.choose_simplified),
            ("Build Polygon", self.build_polygon),
            ("Draw Floorplan", self.draw_floorplan_menu),
            ("Fitness", self.get_fitness),
            ("Draw Plan Equal Thickness", self.draw_floorplan_menu),
            ("Draw Plan with Value", self.draw_plan_with_values),
            ("Return Floorplan", self.return_floorplan),
            ("Exit", self.root.quit),
        ]

        # 버튼들을 왼쪽 프레임에 추가
        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command, width=button_width).pack(pady=5, padx=5, fill=tk.X)

        # 왼쪽 캔버스 생성 및 배치 (초기 셀 배치용)
        self.initial_canvas = tk.Canvas(left_right_frame, width=400, height=600)
        self.initial_canvas.pack(fill=tk.BOTH, expand=True)

        # 오른쪽 캔버스 생성 및 배치 (결과 Floorplan 디스플레이용)
        self.final_canvas = tk.Canvas(right_right_frame, width=400, height=600)
        self.final_canvas.pack(fill=tk.BOTH, expand=True)

        # 피트니스 결과를 위한 프레임 (오른쪽 프레임 아래쪽에 배치)
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # 피트니스 결과를 위한 레이블
        self.fitness_label = tk.Label(bottom_frame, text="Fitness Results", font=("Arial", 14))
        self.fitness_label.pack(pady=10)


    def show_plot_on_canvas(self, fig, target_canvas):
        for widget in target_canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=target_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def return_floorplan(self):
        if self.floorplan:
            self.callback(self.floorplan)
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Create the floorplan first")

    def initialize_room_location(self):
        if self.init_grid is not None:
            self.seed, self.initial_cells = locate_initial_cell(self.init_grid, self.num_rooms)
            self.draw_floorplan(self.seed, self.initial_canvas)
        else:
            messagebox.showwarning('Error', 'init_grid is not given')

    def initialize_floorplan(self):
        self.simplified_candidates.clear()
        if self.seed is not None:
            self.floorplan, self.initial_cells  = create_floorplan(self.seed, self.initial_cells, self.num_rooms, self.options) # todo to change plan.create_floorplan
            if self.floorplan is not None:
                self.draw_floorplan(self.floorplan, self.final_canvas)
            else:
                messagebox.showwarning('Error', 'Initialize_room_location First. No possible seed for Adjacency Constraint')
        else:
            messagebox.showwarning('Error', 'init_grid is None')

    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')

    def draw_floorplan_menu(self):
        self.create_path()
        if self.simplified_floorplan is not None:
            fig = GridDrawer.color_cells_by_value(self.simplified_floorplan, self.full_path, display=True, save=False, num_rooms=self.num_rooms)
            self.show_plot_on_canvas(fig,self.initial_canvas)
        elif self.floorplan is not None:
            fig = GridDrawer.color_cells_by_value(self.floorplan, self.full_path, display=True, save=False, num_rooms=self.num_rooms)
            self.show_plot_on_canvas(fig,self.initial_canvas)
        else: messagebox.showwarning('Error', 'Create Floorplan First')


    # todo place canvas parameter to all calling function
    def draw_floorplan(self, floorplan, canvas ):

        self.create_path()
        if floorplan is not None:
            fig = GridDrawer.color_cells_by_value(floorplan, self.full_path, display=False, save=True, num_rooms = self.num_rooms)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_plan_with_values(self):

        if self.path is None:
            self.create_path()
        # todo: 이렇게 하면 꼬인다. simlified_floorplan을 표시하느냐 floorplan을 표시하느냐 이걸 결정해서 로직을 바꾸자.
        if self.simplified_floorplan is not None:
            floorplan = self.simplified_floorplan
        elif self.floorplan is not None:
            floorplan = self.floorplan
        else:
            messagebox.showwarning("Warning", "Load floorplan first")
            return

        self.draw_plan(floorplan, self.final_canvas)

    def draw_plan(self, floorplan, canvas):

        full_path = trivial_utils.create_filename(self.path, 'Floorplan', '', '', 'png')
        fig = GridDrawer.draw_plan_with_metrics(floorplan, full_path, display=False, save=False, num_rooms=self.num_rooms, metrics=self.fit.room_areas)
        self.show_plot_on_canvas(fig, self.final_canvas)

    def draw_padded(self):
        if self.floorplan:
            GridDrawer.draw_plan_padded(self.floorplan)
            messagebox.showinfo("Info", "Plan drawn with padding")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")
    # underway rectangularity 확인
    def exchange_cells(self):
        if self.floorplan is not None:
            self.simplified_floorplan = exchange_protruding_cells(self.floorplan, 10)
            self.simplified_candidates.append(self.simplified_floorplan)
            self.draw_floorplan(self.simplified_floorplan, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def choose_simplified(self):
        min_cas_len = np.sum(self.floorplan >= 1) # max_cascading_cell_length
        min_cas_list = []
        min_idx = 0
        for idx, fl in enumerate(self.simplified_candidates):
            cas_length, cas_list = count_cascading_cells(fl) # todo cascading_cell의 갯수가 가장 적은 floorplan을 구하자. 이 부분은 exchange_proturding_cells를 여러번 한 후 가장 optimum한 것을 고를 때 필요함
            if cas_length < min_cas_len:
                min_cas_len = cas_length
                min_idx = idx
                max_cas_list = cas_list
        self.floorplan = self.simplified_candidates[min_idx]

    # todo 현재는 일일히 단추를 눌러서 하나씩 하지만 모든 걸 한꺼번에 할 수 있어야 한다.
    def get_fitness(self):
        # 피트니스 값 계산 로직 (예시)
        if self.floorplan is None:
            messagebox('Warning', 'Floorplan not created')
            return
        if self.room_polygons is None:
            messagebox.showwarning('Warning', 'Build Polygon First')
            return

        self.fit = Fitness(self.floorplan, self.num_rooms, self.room_polygons) # todo to change Fitness
        fitness_values = {

            "Rectangularity": self.fit.rectangularity, # todo property 이기 때문에 method를 반납함. 따라서 Fitness에서 rectangularity를 직접 가지고 있어야 함
            "Room Shape Complexity": self.fit.complexity,
            "Room Regularity": self.fit.regularity,
            "Size Satisfaction": self.fit.size_satisfaction,
            "Adjacency Satisfaction": self.fit.adj_satisfaction,
            "Circulation Efficiency": 0.75,
            "Total Fitness": 0.83
        }

        # 결과를 문자열로 포맷팅
        fitness_result = "\n".join([f"{key}: {value:.2f}" for key, value in fitness_values.items()])

        # 레이블에 피트니스 결과 표시
        self.fitness_label.config(text=f"Fitness Results:\n{fitness_result}")

    def build_polygon(self):
        if self.floorplan is not None:
            grid_polygon = GridPolygon(self.floorplan) #todo get scale from constraints.ini
            self.room_polygons = grid_polygon.room_polygons
            polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)

            suffix = 'before'
            # todo fitness class를 콜할 때 이 room_polygon을 넘겨주자.
            # Fitness call하기 전에 room_pollygon이 있어야 한다.
            fig  = polygon_exporter.save_polygon_to_png(self.room_polygons, f"room_polygons_{suffix}.png")
            self.show_plot_on_canvas(fig, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def categorize_cells(self):
        if self.floorplan is not None:
            horiz, vert = categorize_boundary_cells(self.floorplan, 1)
            messagebox.showinfo("Info", f"Boundary cells categorized: h = {horiz}, v = {vert}")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def build_graph(self):
        if self.floorplan is not None:
            build_modules = {
                "1": GraphBuilder.build_graph,
                "2": GraphBuilder.build_weighted_graph,
                "3": GraphBuilder.build_graph_with_inside_cells,
                "4": GraphBuilder.build_graph_connect_4way,
                "9": exit_module
            }
            self.graph = run_selected_module(build_modules, self.floorplan)
            messagebox.showinfo("Info", "Graph built")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def draw_graph(self):
        if self.graph is not None and self.full_path is not None:
            draw_modules = {
                "1": GraphDrawer.draw_graph,
                "2": GraphDrawer.draw_weighted_graph,
                "3": GraphDrawer.draw_graph_with_boundary,
                "9": exit_module
            }
            run_selected_module(draw_modules, self.graph, self.full_path)
            messagebox.showinfo("Info", "Graph drawn")
        else:
            messagebox.showwarning("Warning", "Build graph and create path first")


    def create_widgets_save(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        button_width = 20

        buttons = [
            ("Create Initial Floorplan", self.initialize_floorplan),
            ("Draw Floorplan", self.draw_floorplan),
            ("Simplify Floorplan", self.exchange_cells),
            ("Build Polygon", self.build_polygon),
            ("Draw Plan Equal Thickness", self.draw_equal_thickness),
            ("Return Floorplan", self.return_floorplan),
            ("Exit", self.root.quit),
        ]

        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command, width=button_width).pack(pady=5, padx=5, fill=tk.X)

        self.canvas = tk.Canvas(right_frame, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)