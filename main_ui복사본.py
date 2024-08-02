import tkinter as tk
from tkinter import filedialog, messagebox
import trivial_utils  # Replace with the actual module
import constants
from main import (
    get_floorplan, GridDrawer, exchange_protruding_cells,
    categorize_boundary_cells, GraphBuilder, GraphDrawer,
    run_selected_module, build_polygon, exit_module
)
import sys

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import file_trans
import planB
from plan_utils import grid_to_coordinates, expand_grid
import trivial_utils
from GraphClass import GridGraph, GraphDrawer, GraphBuilder
from GridDrawer import GridDrawer
from GridPolygon import GridPolygon
from PolygonExporter import PolygonExporter
from measure import categorize_boundary_cells
from plan import create_floorplan
from cell_variation import exchange_protruding_cells
from config_reader import read_constraint, read_config_boolean, read_config_int
from trivial_utils import create_filename_with_datetime
import constants
import numpy as np

from plan_utils import grid_to_coordinates, expand_grid

import tkinter as tk
from tkinter import messagebox



class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title = 'Main'
        self.create_sample_init_grid()
        self.num_room = read_config_int('constraints.ini', 'Numbers','num_rooms''')
        self.floorplan = None

        # todo
        self.n_pops = 10
        self.population = []

        self.create_widgets()
        # 일단 floorplan을 가져오고 그걸 모으자.


    def create_widgets(self):
        tk.Button(self.root, text="Draw Footprint Boundary", command=self.draw_plan_base_grid).pack(pady=20)
        tk.Button(self.root, text='Open Floorplan App', command=self.open_floorplan_app).pack(pady = 20)
        tk.Button(self.root, text="Show Floorplan", command=self.show_floorplan).pack(pady=20)

        tk.Button(self.root, text='Exit', command=self.root.quit).pack(pady = 20)

    def create_sample_init_grid(self):
        # 임시 예제 데이터
        self.grid = constants.floor_grid
        self.grid = expand_grid(self.grid)

    def draw_plan_base_grid(self):
        coords = grid_to_coordinates(self.grid)
        # coords = np.argwhere(self.grid == 1)

        path = trivial_utils.create_folder_by_datetime()  # todo test
        full_path = trivial_utils.create_filename(path, 'Grid', '', '', 'png')
        GridDrawer.draw_grid(coords, full_path)

    def open_floorplan_app(self):
        floorplan_window = tk.Toplevel(self.root)
        self.floorplan_app = FloorplanApp(floorplan_window, self.grid, self.num_room, self.receive_floorplan)

    def draw_grid_wrapper(self):
        coords = grid_to_coordinates(self.grid)
        path = trivial_utils.create_folder_by_datetime()  # todo test
        full_path = trivial_utils.create_filename(path, 'Grid', '', '', 'png')
        GridDrawer.draw_grid(coords, full_path)

    def receive_floorplan(self, floorplan):
        self.floorplan = floorplan
        messagebox.showinfo("Info", "Floorplan received in Main App")

    def show_floorplan(self):
        if self.floorplan:
            print("Floorplan:")
            for row in self.floorplan:
                print(row)
        else:
            messagebox.showinfo("Info", "Floorplan not yet created")


class FloorplanApp:
    def __init__(self, root, init_grid, callback):
        self.root = root
        self.root.title("Floorplan UI")

        self.init_grid = init_grid
        self.callback = callback  # Main App으로 floorplan을 반환하기 위한 콜백 함수
        self.floorplan = None

        self.create_widgets()
        self.path = None
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Create Initial Floorplan", command=self.initialize_floorplan).pack(pady=5)
        tk.Button(self.root, text="Draw Floorplan", command=self.draw_floorplan).pack(pady=5)
        tk.Button(self.root, text="Simplify Floorplan", command=self.exchange_cells).pack(pady=5)
        tk.Button(self.root, text="Draw Plan Equal Thickness", command=self.draw_equal_thickness).pack(pady=5)
        tk.Button(self.root, text='Return Floorplan', command=self.return_floorplan).pack(pady=5)
        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=20)


    #        tk.Button(self.root, text="Draw Plan Padded", command=self.draw_padded).pack(pady=5)
#        tk.Button(self.root, text="Categorize Boundary Cells", command=self.categorize_cells).pack(pady=5)
#        tk.Button(self.root, text="Build Graph", command=self.build_graph).pack(pady=5)
#        tk.Button(self.root, text="Draw Graph", command=self.draw_graph).pack(pady=5)
#        tk.Button(self.root, text="Build Polygon", command=self.build_polygon).pack(pady=5)
#        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=5)

    def return_floorplan(self):
        if self.floorplan:
            self.callback(self.floorplan)
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Create the floorplan first")
    def initialize_floorplan(self):
        if self.init_grid is not None:
            self.floorplan = create_floorplan(self.init_grid, self.num_rooms)
            messagebox.showinfo("Info", "Initial floorplan created")


    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')
        messagebox.showinfo("Info", f"Path created: {self.full_path}")

    def draw_floorplan(self):
        self.create_path()
        if self.floorplan is not None:
            GridDrawer.color_cells_by_value(self.floorplan, self.full_path)
            messagebox.showinfo("Info", "Cells colored by Room value")
        else:
            messagebox.showwarning("Warning", "Load floorplan and create path first")

    def draw_equal_thickness(self):
        if self.floorplan:
            GridDrawer.draw_plan_equal_thickness(self.floorplan)
            messagebox.showinfo("Info", "Plan drawn with equal thickness")
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
            messagebox.showinfo("Info", "Protruding cells exchanged")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
