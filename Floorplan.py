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
from main import GridDrawer, create_floorplan, exchange_protruding_cells, categorize_boundary_cells, GraphBuilder, GraphDrawer, run_selected_module, build_polygon, exit_module

class FloorplanApp:
    def __init__(self, root, init_grid, num_rooms, callback):
        self.root = root
        self.root.title("Floorplan UI")

        self.init_grid = init_grid
        self.num_rooms = num_rooms
        self.callback = callback  # Main App으로 floorplan을 반환하기 위한 콜백 함수
        self.floorplan = None

        self.create_widgets()
        self.path = None

    def create_widgets(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(left_frame, text="Create Initial Floorplan", command=self.initialize_floorplan).pack(pady=5)
        tk.Button(left_frame, text="Draw Floorplan", command=self.draw_floorplan).pack(pady=5)
        tk.Button(left_frame, text="Simplify Floorplan", command=self.exchange_cells).pack(pady=5)
        tk.Button(left_frame, text="Draw Plan Equal Thickness", command=self.draw_equal_thickness).pack(pady=5)
        tk.Button(left_frame, text='Return Floorplan', command=self.return_floorplan).pack(pady=5)
        tk.Button(left_frame, text="Exit", command=self.root.quit).pack(pady=20)

        self.canvas = tk.Canvas(right_frame, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)


    def show_plot_on_canvas(self, fig):
        for widget in self.canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def return_floorplan(self):
        if self.floorplan:
            self.callback(self.floorplan)
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Create the floorplan first")

    def initialize_floorplan(self):
        if self.init_grid is not None:
            self.floorplan = create_floorplan(self.init_grid, self.num_rooms)
            self.draw_floorplan()

    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')

    def draw_floorplan(self):
        self.create_path()
        if self.floorplan is not None:
            fig = GridDrawer.color_cells_by_value(self.floorplan, self.full_path, display=False, save=True, num_rooms = self.num_rooms)
            self.show_plot_on_canvas(fig)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_equal_thickness(self):
        if self.floorplan is not None:
            GridDrawer.draw_plan_equal_thickness(self.floorplan)
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
            self.draw_floorplan()
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