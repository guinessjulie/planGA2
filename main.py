# This is a sample Python script.
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
from simplify import exchange_protruding_cells
from reqs import Req
from trivial_utils import create_filename_with_datetime
import constants
import numpy as np


def run_util(): #TODO to connect all the main utils
    file_trans.gui_main()
# Press the green button in the gutter to run the script.



def run_selected_module(modules, param1, param2=None):
    print(f'param1={param1}, parma2={param2}')
    for key, module in modules.items():
        print(f'{key}:{module.__name__}')
    choice = input('Choose: ')
    if choice in modules:
        return modules[choice](param1, param2)



def run_build_floorplan():
    floor_grid = constants.floor_grid
    m,n,k = len(floor_grid), len(floor_grid[0]),constants.NUM_SPACE
    reqs = Req()
    floorshape = create_floorplan(m,n,k,floor_grid,reqs=reqs)
    return floorshape

def exit_module(param1= None, param2=None):
    sys.exit(1)



def choose_colord_grid():
    grids = {
        "1": constants.test_grid,
        "2": constants.test_grid2,
        "3": run_build_floorplan
    }
    for key, base in grids.items():
        base = f'{base.__name__}' if callable(base) else f'{base}'
        print(f'[{key}]:{base}')
    print('choose grid to draw')
    selected = grids[input('Select base floor: ')]

    #  함수면 콜해서 리턴을 받고, 리스트면 사용
    if callable(selected) :
        grid = selected()
    else:
        grid = selected
    return grid


def draw_plan_base_grid(grid):
    print('Draw Plan')
    # todo choose grid or load fromm  last years project
    # coords = plan_utils.grid_to_coordinates(constants.floor_grid )
    coords = grid_to_coordinates(grid )

    path = trivial_utils.create_folder_by_datetime() # todo test
    full_path = trivial_utils.create_filename(path,'Grid', '', '', 'png')

    grid_draw_modules = {
        "1": GridDrawer.draw_grid, # draw as it is
        "2": GridDrawer.draw_grid_reversed,
        "9": exit_module
    }
    # run_selected_module(grid_draw_modules,coords,grid_draw_file)
    run_selected_module(grid_draw_modules,coords,full_path) #todo test

def get_floorplan():
    grid = constants.floor_grid
    grid = expand_grid(grid)
    draw_plan_base_grid(grid)
    nrows,ncols,k=len(grid), len(grid[0]), 5
    floorplan = create_floorplan(grid, k)
    return floorplan

def build_polygon(floorplan):

    grid_polygon = GridPolygon(floorplan, min_area=2000000, min_length=2000)
    polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)

    suffix = 'before'
    room_polygons = grid_polygon.get_all_room_polygons()
    polygon_exporter.save_polygon_to_png(room_polygons, f"room_polygons_{suffix}.png")



def test_main():
    floorplan = get_floorplan()

    path = trivial_utils.create_folder_by_datetime() # todo test
    full_path = trivial_utils.create_filename(path,'Plan', '', '', 'png')
    print(full_path)


    GridDrawer.color_cells_by_value(floorplan, full_path)
    GridDrawer.draw_plan_equal_thickness(floorplan)
    GridDrawer.draw_plan_padded(floorplan)
    exchange_protruding_cells(floorplan, 10)
    horiz, vert = categorize_boundary_cells(floorplan, 1)
    print(f'test_categorized_boundary_cells: length = h = {horiz}, v = {vert}')

    build_modules = {
        "1": GraphBuilder.build_graph,
        "2": GraphBuilder.build_weighted_graph,
        "3": GraphBuilder.build_graph_with_inside_cells,
        "4": GraphBuilder.build_graph_connect_4way,
        "9": exit_module
    }
    graph = run_selected_module(build_modules, floorplan)

    draw_modules = {
        "1": GraphDrawer.draw_graph,
        "2": GraphDrawer.draw_weighted_graph,
        "3": GraphDrawer.draw_graph_with_boundary,
        "9" : exit_module
    }
    run_selected_module(draw_modules, graph, full_path)

    polygon_module = {
        "1" : build_polygon,
        "9" : exit_module
    }

    run_selected_module(polygon_module, floorplan)



def simple_test():
    # Mock floorplan for demonstration, replace with actual implementation details
    floorplan = np.array([
    [1, 0, 0, 2, -1, -1],
    [3, 5, 0, 0, -1, -1],
    [4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
    ])
    obtainable_cells = [(0, 0), (0, 3), (1, 0), (1 ,1), (2, 0)]  # Starting cells that are not zero

    updated_floorplan = exchange_protruding_cells(floorplan, obtainable_cells)
    print("Updated floorplan:")
    for row in updated_floorplan:
        print(row)


if __name__ == '__main__':
    test_main()
    #simple_test()
