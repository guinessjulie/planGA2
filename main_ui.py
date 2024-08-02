import tkinter as tk
from tkinter import filedialog, messagebox
import trivial_utils  # Replace with the actual module
import constants
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

from main import (
    get_floorplan, GridDrawer, exchange_protruding_cells,
    categorize_boundary_cells, GraphBuilder, GraphDrawer,
    run_selected_module, build_polygon, exit_module
)
import sys

from plan_utils import grid_to_coordinates, expand_grid
from config_reader import read_constraint, read_config_boolean, read_config_int
from trivial_utils import create_filename_with_datetime
import numpy as np
from Floorplan import FloorplanApp  # 여기서 FloorplanApp을 임포트


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Main')
        self.create_sample_init_grid()
        self.num_room = read_config_int('constraints.ini', 'Numbers', 'num_rooms')
        self.floorplan = None

        # todo
        self.n_pops = 10
        self.population = []

        self.create_widgets()
        # 일단 floorplan을 가져오고 그걸 모으자.

    def create_widgets(self):
        tk.Button(self.root, text="Draw Footprint Boundary", command=self.draw_plan_base_grid).pack(pady=20)
        tk.Button(self.root, text='Open Floorplan App', command=self.open_floorplan_app).pack(pady=20)
        tk.Button(self.root, text="Show Floorplan", command=self.show_floorplan).pack(pady=20)
        tk.Button(self.root, text='Exit', command=self.root.quit).pack(pady=20)

    def create_sample_init_grid(self):
        # 임시 예제 데이터
        self.grid = constants.floor_grid
        self.grid = expand_grid(self.grid)

    def draw_plan_base_grid(self):
        coords = grid_to_coordinates(self.grid)
        path = trivial_utils.create_folder_by_datetime()  # todo test
        full_path = trivial_utils.create_filename(path, 'Grid', '', '', 'png')
        GridDrawer.draw_grid(coords, full_path)

    def open_floorplan_app(self):
        floorplan_window = tk.Toplevel(self.root)
        self.floorplan_app = FloorplanApp(floorplan_window, self.grid, self.num_room, self.receive_floorplan)

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




if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()