from util import Util
from landgrid import LandGrid

from options import Options
import math

import constant
class Fitness:
    def __init__(self, element, width, height, numCell, id=0):  # floor:[(x,y),...]
        config = Options()
        self._width, self._height = width, height,
        self.config_options = lambda ky: config.config_options(ky)
        self._grid= element
        self.adjacency_graph = element.cell_adjacent_graph_for_rooms()
        self._attrs, self._fits, self.edges = self.build_attrs(element, numCell, id)
        self._id = id
        # Util.display_str_dict(self._fits, 'Fitness') #get rid of fitness values on the console


#먼저 init을 좀 정리하자.attr fit를 딴데서
    def build_attrs(self, gridpos, numCell, id = 0):
        attrs = {}
        fits = {}
        config = Options()
        grid_by_col = gridpos.sorted_by_col()
        grid_by_row = gridpos.sorted_by_row()

        attrs['number of cells'] = numCell
        max_width = grid_by_col[numCell - 1].x - grid_by_col[0].x + 1
        max_height = grid_by_row[numCell - 1].y - grid_by_row[0].y + 1
        attrs['id'] = id
        attrs['max width'] = max_width
        attrs['max height'] = max_height
        attrs['real max width'] = max_width *  config.config_options('cell_length')
        real_vertical_length = max_height *  config.config_options('cell_length')
        attrs['real max height'] = real_vertical_length

        fits['id'] = id
        buildingends, lotends, fits['Fulfill Building Line'] = self.fulfill_building_line(grid_by_row, grid_by_col)
        fits['Setbacks'] = 'Failure' if fits['Fulfill Building Line'] == 'Failure' else self.check_setbakcs(buildingends, lotends, grid_by_row, grid_by_col)
        attrs['boundary_walls'], attrs['area'], attrs['perimeter'],f_par, attrs['real faratio'] = self.pa_ratio(numCell)
        # for debug only
        if f_par > 1:
            print(f_par)
        fits['f(BCR)'] = attrs['real faratio'] / self.config_options('required_faratio')
        fits['f(PAR)'] = f_par
        edges = self.get_edges()
        aspect_ratio = self.get_aspect_ratio(edges)
        south_side = self.get_south_ratio(edges)
        fits['FSH(Sunlight Hours)'], fits['f(FSH)'] = self.get_daylight_15min(edges) # todo test and finish to get detailed 15min sunlight
        fits['f(SSR)'] = south_side
        fits['AR(Aspect Ratio)'] = aspect_ratio
        optimal_aspect_ratio = self.config_options('optimal_aspect_ratio')[0] # because it returns list
        optimal_aspect_ratio_value = constant.RATIOS[optimal_aspect_ratio]
        fits['f(AR)'] = min(aspect_ratio, optimal_aspect_ratio_value) / max(optimal_aspect_ratio_value, aspect_ratio)

        # fits['f(AR)'] = aspect_ratio / optimal_aspect_ratio_value
        fits['f(VSymm)'], fits['f(HSymm)'] = self.get_symmetry()
        fits['f(CC)'] = len(gridpos.connected_component())
        # fits['daylight hour'] = self.get_daylight_hour(edges, real_vertical_length) # todo edges not set yet
        return attrs, fits, edges
        # Util.printAdjGraph(self.adjacency_graph) #todo: for Debug

    # for Column name renaming 
    def build_attrs2(self, gridpos, numCell):
        attrs = {}
        fits = {}
        config = Options()
        grid_by_col = gridpos.sorted_by_col()
        grid_by_row = gridpos.sorted_by_row()

        attrs['number of cells'] = numCell
        max_width = grid_by_col[numCell - 1].x - grid_by_col[0].x + 1
        max_height = grid_by_row[numCell - 1].y - grid_by_row[0].y + 1
        attrs['max width'] = max_width
        attrs['max height'] = max_height
        attrs['real max width'] = max_width *  config.config_options('cell_length')
        real_vertical_length = max_height *  config.config_options('cell_length')
        attrs['real max height'] = real_vertical_length

        buildingends, lotends, fits['Fulfill Building Line'] = self.fulfill_building_line(grid_by_row, grid_by_col)
        fits['Setbacks'] = 'Failure' if fits['Fulfill Building Line'] == 'Failure' else self.check_setbakcs(buildingends, lotends, grid_by_row, grid_by_col)
        attrs['boundary_walls'], attrs['area'], attrs['perimeter'],f_par, attrs['real faratio'] = self.pa_ratio(numCell)
        fits['f(BCR)'] = attrs['real faratio'] / self.config_options('required_faratio')
        fits['f(PAR)'] = f_par
        edges = self.get_edges()
        aspect_ratio = self.get_aspect_ratio(edges)
        south_side = self.get_south_ratio(edges)
        fits['FSH(Sunlight Hours)'], fits['f(FSH)'] = self.get_daylight_15min(edges) # todo test and finish to get detailed 15min sunlight
        fits['f(SSR)'] = south_side
        fits['AR(Aspect Ratio)'] = aspect_ratio
        optimal_aspect_ratio = self.config_options('optimal_aspect_ratio')[0] # because it returns list
        optimal_aspect_ratio_value = constant.RATIOS[optimal_aspect_ratio]
        fits['f(AR)'] = min(aspect_ratio, optimal_aspect_ratio_value) / max(optimal_aspect_ratio_value, aspect_ratio)
        fits['f(VSymm)'], fits['f(HSymm'] = self.get_symmetry()
        # fits['daylight hour'] = self.get_daylight_hour(edges, real_vertical_length) # todo edges not set yet
        return attrs, fits, edges
        # Util.printAdjGraph(self.adjacency_graph) #todo: for Debug#


# 따로 떼자 너무 복잡

    def fulfill_building_line(self, rows, cols):
        road_side= self.config_options('road_side')[0] #todo f
        road_width = self.config_options('road_width')
        cell_length = self.config_options('cell_length')
        setback_requirement = self.config_options('setback_requirement')
        setback_for_road_width = 0
        if(road_width < 4):
            setback_for_road_width = (4 - road_width) / 2

        plan_setbacks = {}
        required_setbacks = {}

        required_setbacks[road_side] = setback_for_road_width + setback_requirement
        buildingends = {}
        lotends = {}
        buildingends['south'] = lambda rows: rows[len(rows)-1].y
        buildingends['north'] = lambda rows: rows[0].y
        buildingends['east'] = lambda cols: cols[len(cols)-1].x
        buildingends['west'] = lambda cols:cols[0].x
        lotends['east'] = self._width - 1
        lotends['south'] = self._height - 1
        lotends['west'] = 0
        lotends['north'] = 0
        plan_setbacks[road_side] = abs(buildingends[road_side](rows) - lotends[road_side]) * cell_length #todo doing this now go back here

        # first test building_line
        if required_setbacks[road_side] < plan_setbacks[road_side]:
            return buildingends, lotends, 'Success'
        else:
            return buildingends, lotends, 'Failure'



    def check_setbakcs(self, buildingends, lotends, rows, cols):
        cell_length = self.config_options('cell_length')
        setback_requirement = self.config_options('setback_requirement')
        plan_setbacks = {}
        required_setbacks = {}

        if abs(buildingends['south'](rows) - lotends['south'])*2 > setback_requirement  and \
           abs(buildingends['north'](rows) - lotends['north'])*2 > setback_requirement  and \
           abs(buildingends['east'](rows) - lotends['east'])*2 > setback_requirement and \
           abs(buildingends['west'](rows) - lotends['west'])*2 > setback_requirement:
            return 'Success'
        else :
            return 'Failure'
        #
        # required_setbacks[road_side] = building_line_setback + setback_requirement
        # plan_setbacks[road_side] = (building_end_north - land_end_north ) * cell_length

        # 있는 경우에 Failed가 되었다면, 이미 전체가 다 Fail해서 더이상 조사하지 않고 리턴해버렸다.
        # None인데 아직 이 문장을 확인한다는 것은 있다는 거는 succeeded
        # if required_setbacks.get('south') is None: # road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['south'] ='Success' \
        #         if (land_end_south - building_end_south) * 2 > setback_requirement \
        #         else 'Failure'
        # if required_setbacks.get('east') is None: #road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['east'] ='Success' \
        #         if (land_end_east - building_end_east) * 2 > setback_requirement \
        #         else 'Failure'
        # if required_setbacks.get('north') is None: # road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['north'] ='Success' \
        #         if (building_end_north - land_end_north ) * 2 > setback_requirement \
        #         else 'Failure'
        # if required_setbacks.get('west') is None: #road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['west'] ='Success' \
        #         if (building_end_west - land_end_west) * 2 > setback_requirement \
        #         else 'Failure'
        #     return required_building_line_result, 'Failure'



        # second, test setback for all

        # return required_building_line_result, all

    def __str__(self): #get_fitness 에서 print(fitness)를 지워서 이거 필요없음
        fits = self._fits
        config = lambda key : self.config_options(key)
        strFitness = f'Fitness: {self._id}\n\
         1. Area to length of outer wall = {fits["pa_ratio"]:.4f} \n\
         2. Optimal Ratio({config("optimal_aspect_ratio")[0]}) = {fits["optimal_ratio"]:.4f} (Aspect ratio: {fits.get("aspect_ratio")})\n\
         3. Symmetry: (Vertical: {fits["symmetry"][0]:.2f}, Horizontal:{fits["symmetry"][1]:.2f})\n\
         4. South View Ratio: {fits["south_side"]:.4f}\n\
         5. Solar Hour: {fits["sunlight hour"]}hours'
        return strFitness

    def verify_setback_requirement(self):
        pass

    def pa_ratio(self, numCell):
        wall_length = self.config_options('cell_length')
        cell_area = wall_length ** 2 # area per cell
        boundary_walls =  self.boundary_length(numCell) #여기서
        perimeter = boundary_walls * wall_length
        # numCell = self._attrs['number of cells']
        floorarea = numCell * cell_area
        landarea = cell_area * self._width * self._height
        return boundary_walls, floorarea, perimeter, 16*floorarea / perimeter**2, floorarea/landarea

    #남향 방향으로 얼마나 빈 공간이 있느냐 하는 것
    def south_gap(self, edges):
        cell_length = self.config_options('cell_length')
        front_space_south =sum([self._grid.height-1-cell.y for cell in edges['south']]) * cell_length
        return front_space_south

    def get_daylight_hour(self, edges, real_vertical_length):
        h = self.config_options('height_diff_south')
        d = self.config_options('adjacent_distance_south')
        d = d + self.south_gap(edges, real_vertical_length)
        altitudes = []
        solar_data = {}
        with open('solar_elevation_hourly.txt', 'r') as fp:
            for line in fp:
                altitude = [float(x) for x in line.split('\t') if x.strip()]
                solar_data[int(altitude[0])] = altitude[1]
                altitudes.append(altitude[1])
        solar_hour = sum(1 for altitude in altitudes
                            if altitude > 0
                            and float(h)/math.tan(math.radians(altitude)) < d)
        return solar_hour
    def get_daylight_15min(self, edges):
        h = self.config_options('height_diff_south')
        occluded_distance_to_south= self.config_options('adjacent_distance_south')
        south_gap_edges = self.south_gap(edges)
        avg_d = south_gap_edges + occluded_distance_to_south
        altitudes = []
        solar_data = {}
        with open(constant.SOLAR_ELEVATION_ANGLE, 'r') as fp:
            for line in fp:
                altitude = [float(x) for x in line.split('\t') if x.strip()]
                solar_data[int(altitude[0])] = altitude[1]
                altitudes.append(altitude[1])
        solar_15min = sum(0.25 for altitude in altitudes
                            if altitude > 0
                            and float(h)/math.tan(math.radians(altitude)) < avg_d)
        fitness_sunlight  = solar_15min / sum(1/4 for altitude in altitudes if altitude > 0)

        return solar_15min, fitness_sunlight
    # todo: not yet used. get polygon coordinates
    # todo: to get polygon coordinates for num_vertices and etc
    def get_polygon_vertice(self):
        tupPos = [tuple((p.x, p.y)) for p in self._grid.poses]
        return Util.trace(tupPos)

    def bound_box(self):
        newpos = Util.move_topleft((self._grid.poses))
        #print(newpos) #todo:recover
        print(Util.bounding_box(newpos)) # todo recover

    def get_symmetry(self):
        # return 0,0
        newpos = Util.move_topleft(self._grid.poses)
        bb = Util.bounding_box(newpos)
        width = bb[1]+1; height = bb[3]+1
        newgrid = LandGrid(newpos, width, height)

        horz_diff = 0; vertical_diff = 0;
        i = 0; k = height - 1

        # Horizontal Symmetry
        while(i < width // 2): #floor division operator
            for j in range(width):
                try:
                    if(newgrid.grid[i][j] != newgrid.grid[k][j]):
                        horz_diff +=1
                except IndexError:
                    return 0,0
            i += 1; k -= 1

        # Vertical Symmetry
        i = 0; k = width - 1
        while (i < height // 2):
            # Checking each cell of a row.
            for j in range(height):
                try:
                    if (newgrid.grid[j][i] != newgrid.grid[j][k]):
                        vertical_diff += 1
                except IndexError:
                    return 0, 0
            k -= 1; i += 1

        vertical_symm = 1 - vertical_diff / len(self._grid.poses)
        horz_symm = 1 - horz_diff / len(self._grid.poses)

        return  vertical_symm, horz_symm,

    def boundary_length(self, numCell):  # 외피 사이즈
        sum_adjs = sum(len(v) for v in self.adjacency_graph.values())  # sum of count adajcencies
        return numCell*4 - sum_adjs

    @staticmethod
    def get_aspect_ratio(edges):
        return len(edges['south']) / len(edges['east'])

    def get_south_ratio(self, edges):
        cell_length = self.config_options('cell_length')
        south_side_length = len(edges['south']) * cell_length
        land_horiz_length = self._width * cell_length
        setback = self.config_options('setback_requirement')
        one_side_setback = cell_length if setback < cell_length else setback
        both_side_setback =  one_side_setback * 2
        max_south_length = land_horiz_length - both_side_setback
        return south_side_length / max_south_length

    # def side_cells(self):
    #
    #     list_of_edges = [edges[i] for i in edges]
    #     set_edges = set([cell for cells in list_of_edges  for cell in cells]) #flattened=>set
    #     feasible_south_ratio = self._width / len(set_edges)
    #     south_side_ratio =  len(edges['south'] ) / len(set_edges)
    #     new_ratio = south_side_ratio / feasible_south_ratio
    #     south_side_ratio2 = len(edges['south']) /self._width
    #     # return edges, aspect_ratio, south_side_ratio
    #     return edges, aspect_ratio, south_side_ratio2

    def get_edges(self):
        rows = self._grid.grouped_by_row()
        cols = self._grid.grouped_by_col()
        edges = {}
        edges['east'] = [max(row, key=lambda pos: pos.x) for row in rows]
        edges['west'] = [min(row, key=lambda pos: pos.x) for row in rows]
        edges['south'] = [max(col, key=lambda pos: pos.y) for col in cols]
        edges['north'] = [min(col, key=lambda pos: pos.y) for col in cols]
        return edges


    def boundary_length_old(self, numCell):  # 외피 사이즈
        # This is the Fitness Function
        cc = self._grid.connected_component()
        # Util.printCC(cc) # todo for DEBUG

        insideWall = 0;
        for cell in self.adjacency_graph:
            insideWall += int(len(self.adjacency_graph[cell]) /2)
        # return self._attrs['number of cells'] * 4 - insideWall
        v = sum(len(v) for v in self.adjacency_graph.values())
        print(f'insidewall = {insideWall}, sum of adjacency graph values: {v}')
        previously_returned = numCell * 4 - insideWall
        newcalc = numCell*4 - v
        print(f'numCell*4-insideWall:={previously_returned}, new calc : {newcalc}')
        return newcalc
