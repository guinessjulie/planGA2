# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import file_trans
import planB
import plan_utils
import trivial_utils
from GraphClass import GridGraph, GraphDrawer, GraphBuilder
from GridDrawer import GridDrawer
from measure import categorize_boundary_cells
from plan import create_floorplan, place_room
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
    floorshape = create_floorplan(m,n,k,floor_grid)
    return floorshape

def exit_module(param1= None, param2=None):
    return


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
    coords = plan_utils.grid_to_coordinates(grid )

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
    draw_plan_base_grid(grid)
    nrows,ncols,k=len(grid), len(grid[0]), 5
    floorplan = create_floorplan(grid, k)
    return floorplan

def test_main():
    floorplan = get_floorplan()

    path = trivial_utils.create_folder_by_datetime() # todo test
    full_path = trivial_utils.create_filename(path,'Plan', '', '', 'png')
    print(full_path)


    #GridDrawer.plot_colored_grid(floorplan,gridname)
    #GridDrawer.color_cells_by_value(floorplan, gridname)
    GridDrawer.color_cells_by_value(floorplan, full_path)
    GridDrawer.draw_plan_equal_thickness(floorplan)
    GridDrawer.draw_plan_padded(floorplan)
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

    updated_floorplan = place_room(floorplan, obtainable_cells)
    print("Updated floorplan:")
    for row in updated_floorplan:
        print(row)


if __name__ == '__main__':
    test_main()
    #simple_test()
