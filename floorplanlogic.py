import numpy as np
from fitness import Fitness
from plan import create_floorplan, locate_initial_cell
from GridPolygon import GridPolygon
from PolygonExporter import PolygonExporter
import trivial_utils
from reqs import Req
from options import Options
import constants
from plan_utils import expand_grid
from simplify import exchange_protruding_cells, count_cascading_cells
from trivial_utils import generate_unique_id, create_fitness_info

# info 이 클라스는  list of dictionary {seed : list of tuple (floorplans:fitness)} 를 갖는다.
class FloorplanLogic:
    def __init__(self, num_rooms):
        self.init_grid = self.get_initial_footprint_grid() # todo footprint
        self.num_rooms = num_rooms #info use options.num_rooms or count from cell
        self.options = Options()
        self.reqs = Req()
        self.seed = None
        self.floorplan = None
        self.floorplans = []
        self.floorplans_dict = {} #info: structure {seed:floorplans} # floorplans = [(floorplan, fit)]
        self.candidates = []
        self.simplified_candidates = []
        self.simplified_floorplan = None
        self.seeds_coords = None
        self.fit = None
        self.path = trivial_utils.create_folder_by_datetime()
        self.option = None

    # 여기에 Floorplan 관련 로직을 넣음
    # 예: create_room_start_cell, initialize_floorplan, draw_floorplan_menu 등

    def repeate_creating_floorplan(self,repeatation=10, seed=None, options=None, reqs=None):
        floorplans = []
        for i in range(repeatation):
            simplified_floorplan, fit = self.build_floorplan_from_seed(seed, options=options, reqs =reqs)
            floorplans.append((simplified_floorplan, fit))
        return floorplans

    def build_floorplan_from_seed(self, seed, options, reqs):
        num_rooms = self.options.num_rooms
        reqs = self.reqs
        initial_floorplan = create_floorplan(seed, k= num_rooms, options = options, reqs=reqs)
            # self.draw_floorplan(initial_floorplan, self.initial_canvas)
        simplified_floorplan, fit= self.get_optimal_from_initial_floorplan(initial_floorplan)
        return simplified_floorplan, fit

    def get_initial_footprint_grid(self): # create_sample_init_grid(self):  from main_ui
        # 임시 예제 데이터
        grid = constants.floor_grid
        return expand_grid(grid)

    # info: from FloorplanApp Menu [batch processing] >
    #  FloorplanApp.seed = FloorplanApp.initialize_location >
    #  Floorplan.seed = floorplan.locate_initial_cell >
    #  seed, seeds_coords = plan.locate_initial_cell
    def locate_initial_cell(self, num_rooms):
        if self.init_grid is not None:
            self.seed, self.seeds_coords, assgigned_seed_by = locate_initial_cell(self.init_grid, num_rooms) #todo self.seed가 여기서 필요한지 안필요한지 모름
            return self.seed, assgigned_seed_by
        else:
            return None
    # info FloorplanApp.create_floorplans_from_seed() > floorplan.iterate_for_optimal_floorplan
    # uses local variables
    # todo iterate for a given seed
    def iterate_floorplan_creation(self, seed, num_iter):  # 이미 silence 모드인지 확인해서 넘겨줬음
        floorplans = []
        for i in range(num_iter):
            initial_floorplan = create_floorplan(seed, k=self.num_rooms, options=self.options, reqs=self.reqs)
            simplified_floorplan, fit = self.get_best_simplified_floorplan(initial_floorplan)
            floorplans.append((simplified_floorplan, fit))

        # average_fitness_label on label
        fits = [fl[1] for fl in floorplans]  # 모든 fitness들을 가져와서
        average_text_result = self.generate_average_fitness_text(fits)
        average_text_result = f'Average Fitness Result \n {average_text_result}'

        # best_fitness_label on best_fitness_label
        avg_fits = [fit.fitness for fit in fits]
        best_fits_index = np.argmax(avg_fits)
        best_fit = floorplans[best_fits_index][1]
        best_fit_floorplan = floorplans[best_fits_index][0]
        best_fit_result = f'Best Fitness Result\nreate_fitness_info(best_fit)'

        room_areas = [room.area for room in best_fit.room_polygons.values()]
        seed_set = {tuple(el) for el in np.argwhere(self.floorplan.seed > 0)} # todo remove redundancy
        unique_id = generate_unique_id(seed_set)
        self.floorplans_dict[unique_id] = floorplans

        return floorplans, average_text_result, best_fit_result, best_fit_floorplan, room_areas  # todo 일단 10 개의 floorplans가 return, best_fit을 가진 floorplan을 return 하게 될 수도 있다. 먼저 작동 확인하자.

    def iterate_optimal_floorplans(self, seed, num_iter): # 이미 silence 모드인지 확인해서 넘겨줬음

        floorplans = []
        for i in range(num_iter):
            initial_floorplan = create_floorplan(seed, k=self.num_rooms, options=self.options, reqs=self.reqs)
            simplified_floorplan, fit = self.get_best_simplified_floorplan(initial_floorplan)
            floorplans.append((simplified_floorplan, fit))

        # average_fitness_label on label
        fits = [fl[1] for fl in floorplans]  # 모든 fitness들을 가져와서
        average_text_result = self.generate_average_fitness_text(fits)
        average_text_result = f'Average Fitness Result \n {average_text_result}'

        # best_fitness_label on best_fitness_label
        avg_fits = [fit.fitness for fit in fits]
        best_fits_index = np.argmax(avg_fits)
        best_fit = floorplans[best_fits_index][1]
        best_fit_floorplan = floorplans[best_fits_index][0]
        best_fit_result = f'Best Fitness Result\n' + create_fitness_info(best_fit)

        room_areas = [room.area for room in best_fit.room_polygons.values()]
        seed_set = {tuple (el) for el in np.argwhere(seed > 0)}
        unique_id = generate_unique_id(seed_set)
        self.floorplans_dict[unique_id] = floorplans
        return floorplans, average_text_result, best_fit_result, best_fit_floorplan, room_areas  # todo 일단 10 개의 floorplans가 return, best_fit을 가진 floorplan을 return 하게 될 수도 있다. 먼저 작동 확인하자.

    def generate_average_fitness_text(self,fits):
        # 2. 고유 식별자를 사용해 결과 저장
        fitness_values = {
            "Adjacency": np.average([fit.adj_satisfaction for fit in fits]),
            "Orientation": np.average([fit.orientation_satisfaction for fit in fits]),
            "Size": np.average([fit.size_satisfaction for fit in fits]),
            "Regularity": np.average([fit.regularity for fit in fits]),
            "Aspect Ratio": np.average([fit.pa_ratio for fit in fits]),
            "Total Fitness": np.average([fit.fitness for fit in fits]),
        }
        text_result = "\n".join([f"{key}: {value:.2f}" for key, value in fitness_values.items()])
        return text_result

    def get_optimal_from_initial_floorplan(self, initial_floorplan):
        optimal_candidates = self.create_candidate_floorplans(initial_floorplan)

        # print(f'{len(optimal_candidates)} floorplan candidates generated')
        # info what self.build_polygon() does

        if len(optimal_candidates) == 1:
            optimal_floorplan = optimal_candidates[0]
            grid_polygon = GridPolygon(optimal_floorplan)
            fit = Fitness(optimal_floorplan, self.num_rooms, grid_polygon.room_polygons, self.reqs)

        else:
            fitnesses={}
            best_fitness = -float('inf')
            optimal_floorplan = None
            best_idx = -1
            for i, fl in enumerate(optimal_candidates):
                grid_polygon = GridPolygon(fl)
                # info what self.get_fitness() does
                fitnesses[i] = Fitness(fl, self.num_rooms, grid_polygon.room_polygons, self.reqs)
                tot_fit = fitnesses[i].fitness
                print('f"tot_fitness={tot_fit}')
                if tot_fit > best_fitness:
                    best_fitness = tot_fit
                    optimal_floorplan = fl
                    best_idx = i

            fit = fitnesses[best_idx]

        # info draw_plan_with_values > draw_plan
        full_path = trivial_utils.create_filename(self.path, 'Optimal_plan', '', '', 'png')
        room_areas = [room.area for room in fit.room_polygons.values()]
        self.floorplans.append((optimal_floorplan, fit)) # info to move from app to logic 0922
        # self.draw_on_canvas_metric(optimal_floorplan,  room_areas, self.final_canvas)
        #     fig = GridDrawer.draw_plan_with_metrics(optimal_floorplan, full_path, display=False, save=False, num_rooms=self.num_rooms, metrics=room_areas)
        #     self.show_plot_on_canvas(fig, self.final_canvas)

        # fitness_result = create_fitness_info(fit)
        # 결과를 문자열로 포맷팅

        # 레이블에 피트니스 결과 표시
        # self.fitness_label.config(text=f"Fitness Results:\n{fitness_result}")
        return optimal_floorplan, fit #info add fitness_result to return argument

    # info 여러 개의 simplified version 중 가장 높은 것을 선택
    def get_best_simplified_floorplan(self, initial_floorplan):
        optimal_candidates = self.create_candidate_floorplans(initial_floorplan)

        if len(optimal_candidates) == 1:
            optimal_floorplan = optimal_candidates[0]
            grid_polygon = GridPolygon(optimal_floorplan)
            fit = Fitness(optimal_floorplan, self.num_rooms, grid_polygon.room_polygons, self.reqs)

        else:
            fitnesses={}
            best_fitness = -float('inf')
            optimal_floorplan = None
            best_idx = -1
            for i, fl in enumerate(optimal_candidates):
                grid_polygon = GridPolygon(fl)
                # info what self.get_fitness() does
                fitnesses[i] = Fitness(fl, self.num_rooms, grid_polygon.room_polygons, self.reqs)
                tot_fit = fitnesses[i].fitness

                if tot_fit > best_fitness:
                    best_fitness = tot_fit
                    optimal_floorplan = fl
                    best_idx = i

            fit = fitnesses[best_idx]

        room_areas = [room.area for room in fit.room_polygons.values()]
        

    
        #     fig = GridDrawer.draw_plan_with_metrics(optimal_floorplan, full_path, display=False, save=False, num_rooms=self.num_rooms, metrics=room_areas)
        #     self.show_plot_on_canvas(fig, self.final_canvas)

        # 결과를 문자열로 포맷팅

        # 레이블에 피트니스 결과 표시
        
        return optimal_floorplan, fit


    # info candidates []에 simplified를 여러개 한 거 중에서 가장 단순화한 것을 리턴.
    #  exchange_protruding_cells의 결과들을 넣는데,
    #  cascading_cells가 가장 작은 값을 리턴한다.
    def create_candidate_floorplans(self, initial_floorplan):
        min_cas = np.sum(initial_floorplan >= 1)  # cascading_cell의 최대 갯수
        num_cas = min_cas
        candidates = []
        iteration_count = 0
        max_iterations = 5

        num_cas_dict = {}
        while num_cas != 0 and iteration_count < max_iterations:
            candidate = exchange_protruding_cells(initial_floorplan, 10)  # todo check iteration count in exchange_...
            candidate_cas, _ = count_cascading_cells(candidate)
            num_cas = candidate_cas
            # print(f'num_cascading_cell = {candidate_cas}')

            if candidate_cas < min_cas:
                candidates.append(candidate)
                min_cas = candidate_cas
                num_cas_dict[len(candidates) -1] = candidate_cas
            iteration_count += 1

        candidates = self.remove_duplicate_floorplan(candidates)

        # 가장 작은 num_cas 값을 가진 모든 candidate를 선택
        min_candidates = [candidates[idx] for idx, cas in num_cas_dict.items() if cas == min_cas]
        # print(f'candidate size = {len(min_candidates)}')
        return min_candidates if min_candidates else [initial_floorplan]


    def remove_duplicate_floorplan(self,candidates):
        unique_candidates = []
        for fl in candidates:
            if not any(np.array_equal(fl, unique) for unique in unique_candidates):
                unique_candidates.append(fl)
        return unique_candidates
