def is_valid_cell(floorplan, cell):
    is_valid = 0 <= cell[0] < floorplan.shape[0] and 0 <= cell[1] < floorplan.shape[1] and floorplan[cell] > 0
    # print(f'{cell} is_valid={is_valid}')
    return is_valid

def is_in(floorplan, cell):
    return  0 <= cell[0] < floorplan.shape[0] and 0 <= cell[1] < floorplan.shape[1]


