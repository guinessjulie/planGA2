import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import trivial_utils
from main import GridDrawer, exchange_protruding_cells, categorize_boundary_cells, GraphBuilder, GraphDrawer, run_selected_module, build_polygon, exit_module
from plan import create_floorplan, locate_initial_cell

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo 1. show simplify
# todo 2. draw plan equal thickness show option
class FloorplanApp:
    def __init__(self, root, init_grid, num_rooms, callback):
        self.root = root
        self.root.title("Floorplan UI")

        self.init_grid = init_grid
        self.num_rooms = num_rooms
        self.callback = callback  # Main App으로 floorplan을 반환하기 위한 콜백 함수
        self.floorplan = None
        self.simlified_floorplan = None
        self.seed = None
        self.initial_cells = None
        self.create_widgets()
        self.path = None

    def create_widgets(self):
        # 왼쪽 프레임 생성 및 배치
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 오른쪽 프레임 생성 및 배치
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 오른쪽 프레임을 상하 두 개의 하위 프레임으로 나누기
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
            ("Draw Floorplan", self.draw_floorplan),
            ("Simplify Floorplan", self.exchange_cells),
            ("Draw Plan Equal Thickness", self.draw_equal_thickness),
            ("Return Floorplan", self.return_floorplan),
            ("Exit", self.root.quit),
        ]

        # 버튼들을 왼쪽 프레임에 추가
        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command, width=button_width).pack(pady=5, padx=5, fill=tk.X)

        # 상단 캔버스 생성 및 배치 (초기 셀 배치용)
        self.initial_canvas = tk.Canvas(left_right_frame, width=800, height=300)
        self.initial_canvas.pack(fill=tk.BOTH, expand=True)

        # 하단 캔버스 생성 및 배치 (결과 Floorplan 디스플레이용)
        self.final_canvas = tk.Canvas(right_right_frame, width=800, height=300)
        self.final_canvas.pack(fill=tk.BOTH, expand=True)





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
        if self.seed is not None:
            self.floorplan, self.initial_cells  = create_floorplan(self.seed, self.initial_cells, self.num_rooms)
            if self.floorplan is not None:
                self.draw_floorplan(self.floorplan, self.final_canvas)
            else:
                messagebox.showwarning('Error', 'Initialize_room_location First. No possible seed for Adjacency Constraint')
        else:
            messagebox.showwarning('Error', 'init_grid is None')

    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')

    # todo place canvas parameter to all calling function
    def draw_floorplan(self, floorplan, canvas ):
        self.create_path()
        if floorplan is not None:
            fig = GridDrawer.color_cells_by_value(floorplan, self.full_path, display=False, save=True, num_rooms = self.num_rooms)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_equal_thickness(self, floorplan):
        if floorplan is not None:
            full_path = trivial_utils.create_filename(self.path, 'Floorplan', '', '', 'png')
            fig = GridDrawer.draw_plan_equal_thickness(floorplan, full_path, display=False, save=True, num_rooms=self.num_rooms)
            self.show_plot_on_canvas(fig, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def draw_padded(self):
        if self.floorplan:
            GridDrawer.draw_plan_padded(self.floorplan)
            messagebox.showinfo("Info", "Plan drawn with padding")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def exchange_cells(self):
        if self.floorplan is not None:
            exchange_protruding_cells(self.floorplan, 10)
            self.draw_floorplan(self.floorplan, self.final_canvas)
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

    def build_polygon(self):
        if self.floorplan:
            polygon_module = {
                "1": build_polygon,
                "9": exit_module
            }
            run_selected_module(polygon_module, self.floorplan)
            messagebox.showinfo("Info", "Polygon built")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

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
            ("Draw Plan Equal Thickness", self.draw_equal_thickness),
            ("Return Floorplan", self.return_floorplan),
            ("Exit", self.root.quit),
        ]

        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command, width=button_width).pack(pady=5, padx=5, fill=tk.X)

        self.canvas = tk.Canvas(right_frame, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)