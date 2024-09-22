import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from fitness import Fitness
from trivial_utils import generate_unique_id,create_fitness_info

import trivial_utils
from main import GridDrawer, exchange_protruding_cells, categorize_boundary_cells, GraphBuilder, GraphDrawer, run_selected_module,  exit_module
from simplify import exchange_protruding_cells, count_cascading_cells
from plan import create_floorplan, locate_initial_cell
from GridPolygon import GridPolygon
from options import Options
from PolygonExporter import PolygonExporter
from ga import GeneticAlgorithm
from floorplanlogic import FloorplanLogic

from reqs import Req
from config_reader import load_config
import configparser
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo create population class
# todo use seed to recreate floorplan
# todo let equal thickness function work customize
# todo run_iteration에서 self.floorplans로 춴래 처리했었느데 로칼변수 floorplans를 선언하고 거기에 리스트를 만들어서 리턴하고 있음. 이 로직을 잘 살펴서 계산해야 함
class FloorplanApp:
    # info: done: in self.simplified_candidates have diff floorplans from the same initiialized_floorplan. after choosing best simplified floorplans. assigns to self.floorplan to set final result
    def __init__(self, root, init_grid, num_rooms, callback):
        self.root = root
        self.root.title("Floorplan UI")
        self.options = Options()
        self.FL = FloorplanLogic(self.options.num_rooms)
        self.create_widgets()
        self.seed_iteration = 0
        self.temperary_seed = None


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
            # ("Initial Room Location", self.create_room_start_cell),
            # ("Batch Processing", self.get_optimal_from_initial_floorplan), todo 이걸로 바꾸려고 했는데 왜 바꾸려고 했는지를 다시 알아내야 함
            ("Create Floorplan", self.run_batch_from_seed),
            # ("Simplify Floorplan", self.exchange_cells),
            # ('Create Floorplan', self.initialize_floorplan),
            ("Method Analysis", self.method_comparison_analysis),
            ("Seed Analysis", self.seed_comparison_analysis),

            ("Evolve", self.evolve),
            # ("Choose Most Simplified", self.choose_simplified),
            # ("Build Polygon", self.build_polygon),
            # ("Draw Floorplan", self.draw_floorplan_menu),
            # ("Fitness", self.get_fitness),
            # ("Draw Plan Equal Thickness", self.draw_floorplan_menu),
            # ("Draw Plan with Value", self.draw_plan_with_values),
            # ("Return Floorplan", self.return_floorplan),
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

        # bottom_frame을 두 개의 하위 프레임으로 나눔
        current_fitness_frame = tk.Frame(bottom_frame)
        current_fitness_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        best_fitness_frame = tk.Frame(bottom_frame)
        best_fitness_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 현재 Fitness 결과를 위한 레이블
        self.fitness_label = tk.Label(current_fitness_frame, text="Current Fitness: N/A", font=("Arial", 14))
        self.fitness_label.pack(pady=10)

        # Best Fitness 결과를 위한 레이블
        self.best_fitness_label = tk.Label(best_fitness_frame, text="Best Fitness: N/A", font=("Arial", 14))
        self.best_fitness_label.pack(pady=10)

        # 버튼을 수평으로 배열하기 위한 프레임 생성
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(side=tk.BOTTOM, pady=20)  # 이 프레임을 bottom_frame의 바닥에 붙임

        # 기존 OK 버튼 코드 (이 부분을 button_frame 내에 삽입)
        self.ok_button = ttk.Button(button_frame, text="OK", command=self.next_iteration)
        self.ok_button.pack(side=tk.LEFT, padx=10)  # 수평 배열을 위해 side=tk.LEFT 사용
        self.ok_button.config(state=tk.DISABLED)  # 첫 시작에는 비활성화

        # 새로운 버튼 추가
        self.save_seed_button = ttk.Button(button_frame, text="Save Seed", command=self.save_seed)
        self.save_seed_button.pack(side=tk.LEFT, padx=10)  # 수평 배열을 위해 side=tk.LEFT 사용
        self.save_seed_button.config(state=tk.DISABLED)  # 첫 시작에는 비활성화

        self.save_floorplan_button = ttk.Button(button_frame, text="Save Floorplan", command=self.save_floorplan)
        self.save_floorplan_button.pack(side=tk.LEFT, padx=10)  # 수평 배열을 위해 side=tk.LEFT 사용
        self.save_floorplan_button.config(state=tk.DISABLED)  # 첫 시작에는 비활성화

    def save_seed(self):
        pass

    def save_floorplan(self):
        pass

    # info Menu event from: [Create Initial Population]
    def run_batch_from_seed(self):
        self.temperary_seed = self.create_room_start_cell()
        if self.options.silence_mode: # todo App에서 options를 가지고 있어야 하냐 한다. 여기서 가지고 Floorplan에 넘겨주자
            self.create_floorplans_from_seed(self.temperary_seed)
        else:
            self.run_iteration()

    def create_column_title(self, assigned_seed_by):
        constr = []
        if self.options.min_size_alloc:
            constr.append('size')
        for cons in assigned_seed_by:
            constr.append(cons)
        return constr

    def seed_comparison_analysis(self):
        num_rooms = self.options.num_rooms
        reqs = Req()
        n_iter = self.options.iteration_from_seed
        savefilename = trivial_utils.create_filename_with_datetime(ext='csv', prefix='Seed_Analysis')
        seed_no = 0
        for i in range(100):
            seed, assigned_seed_by = self.create_room_start_cell()  # 이 부분에서 assigned_seed_by가 바뀜
            seed_no += 1
            fl_fit = []  # 이 안에서 fl_fit 초기화
            seed_fit = dict()
            for j in range(n_iter):
                initial_floorplan = create_floorplan(seed, k=num_rooms, options=self.options, reqs=reqs)

                # Optimal candidates 처리
                optimal_candidates = self.FL.create_candidate_floorplans(initial_floorplan)

                # 각 floorplan에 대해 fitness 계산
                for fl in optimal_candidates:
                    grid_polygon = GridPolygon(fl)
                    fit = Fitness(fl, num_rooms, grid_polygon.room_polygons, reqs)
                    if seed_no not in seed_fit:
                        seed_fit[seed_no] = []
                    else:
                        seed_fit[seed_no].append(fit)
            # 매번 assigned_seed_by가 변할 때마다 데이터 저장
            constraints = self.create_column_title(assigned_seed_by)
            trivial_utils.save_results_by_seed_to_csv(seed_fit, constraints=constraints, filename=savefilename)  # 여기서 CSV 파일에 저장


    def method_comparison_analysis(self):
        n_iter = self.options.iteration_from_seed
        num_rooms = self.options.num_rooms
        reqs = Req()
        savefilename = trivial_utils.create_filename_with_datetime(ext='csv', prefix='Analysis')
        for i in range(100):
            seed, assigned_seed_by = self.create_room_start_cell()  # 이 부분에서 assigned_seed_by가 바뀜

            fl_fit = []  # 이 안에서 fl_fit 초기화
            for j in range(n_iter):
                initial_floorplan = create_floorplan(seed, k=num_rooms, options=self.options, reqs=reqs)

                # Optimal candidates 처리
                optimal_candidates = self.FL.create_candidate_floorplans(initial_floorplan)

                # 각 floorplan에 대해 fitness 계산
                for fl in optimal_candidates:
                    grid_polygon = GridPolygon(fl)
                    fit = Fitness(fl, num_rooms, grid_polygon.room_polygons, reqs)
                    fl_fit.append(fit)

            # 매번 assigned_seed_by가 변할 때마다 데이터 저장
            constraints = self.create_column_title(assigned_seed_by)
            trivial_utils.save_results_to_csv(fl_fit,  constraints=constraints, filename = savefilename)  # 여기서 CSV 파일에 저장

    def create_column_title_str(self, assigned_seed_by):
        title = 'Size ' if self.options.min_size_alloc else ''
        for cons in assigned_seed_by:
            title += cons
        title = 'None' if title == '' else title
        return title

    def create_column_title(self, assigned_seed_by):
        constr = []
        # Size constraint 추가 여부
        constr.append('Size' if self.options.min_size_alloc else 'None')

        # 다른 constraint 추가 (예: assigned_seed_by 리스트에 있는 제약 조건)
        for cons in assigned_seed_by:
            constr.append(cons)

        # 리스트 길이를 3으로 맞추고, 부족하면 'None'으로 채우기
#         while len(constr) < 3:
#             constr.append('None')

        return constr

    def method_comparison_analysis_일단_세이브(self): # copied and joined from run_batch_from_seed > create_floorplans_from_seed
            num_iter_for_seed = 5
            num_rooms = self.options.num_rooms
            reqs = Req()
            fl_fit = []
            assigned_seed_by = []
            for i in range(10):
                seed, assigned_seed_by = self.create_room_start_cell()
                n_iter = self.options.iteration_from_seed

                for i in range(num_iter_for_seed):
                    initial_floorplan = create_floorplan(seed, k=num_rooms, options=self.options, reqs=reqs)
                    # simplified_floorplan, fit = self.FL.get_best_simplified_floorplan(initial_floorplan) # underway 0905  이 내용이 아래에 다시 썼다.
                    optimal_candidates = self.FL.create_candidate_floorplans(initial_floorplan) # 여러개

                    fitnesses = {}
                    for i, fl in enumerate(optimal_candidates):
                        grid_polygon = GridPolygon(fl)
                        # info what self.get_fitness() does
                        fit = Fitness(fl, num_rooms, grid_polygon.room_polygons, reqs)
                        fl_fit.append((fl,fit))
            col_title = self.create_column_title_str(assigned_seed_by)
            fits = [flt[1] for flt in fl_fit]
            # create colum title
            trivial_utils.save_results_to_csv(fits, constraint = col_title, filename=".csv")


    def compare_methods(num_iterations=100, config_file='config.ini', constraints_file='constraints.ini'):
        """
        Compare two methods: `allocate_rooms` and `allocate_rooms_with_size` over a number of iterations.
        """
        results = []

        for _ in range(num_iterations):
            # Allocate floorplan using `allocate_rooms`
            initial_floorplan1 = allocate_rooms(
                floorplan_template)  # Modify according to how allocate_rooms is called
            best_floorplan1, fitness1 = get_optimal_from_initial_floorplan(initial_floorplan1)
            results.append(('allocate_rooms', fitness1))

            # Allocate floorplan using `allocate_rooms_with_size`
            initial_floorplan2 = allocate_rooms_with_size(
                floorplan_template)  # Modify according to how allocate_rooms_with_size is called
            best_floorplan2, fitness2 = get_optimal_from_initial_floorplan(initial_floorplan2)
            results.append(('allocate_rooms_with_size', fitness2))

        # Save results to CSV
        save_results_to_csv(results)

        print(f"Comparison complete. Results saved to 'floorplan_fitness_results.csv'.")

    def show_plot_on_canvas(self, fig, target_canvas):
        for widget in target_canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=target_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def return_floorplan(self):
        if self.FL:
            self.callback(self.FL)
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Create the floorplan first")
    # info run_batch_from_seed
    def create_room_start_cell(self):

        seed, assigned_seed_by = self.FL.locate_initial_cell(self.options.num_rooms) # todo remove self.seed
        if seed is not None:
            self.draw_on_canvas(seed, self.initial_canvas)
        else:
            messagebox.showwarning('Error', 'init_grid is not given')
        return seed, assigned_seed_by

    # todo 1. create_floorplan =>
    #  2. simplified until no cascading exists. or max 5 times with same_cascading number => Done
    #  3. compare with fitness values. withdraw if one is same as others save into self.floorplans
    #  4. repeate create_floorplan => simplified_floorplan -> self.floorplans
    #  5. batch fitnesses for all floorplans of one seed
    #  6. keep track of finnesses of the seed


    # info FloorplanApp.run_batch_from_seed >
    #  create_floorplans_from_seed(seed)
    def create_floorplans_from_seed(self, seed):
        n_iter = self.options.iteration_from_seed
        floorplans, average_text_result, best_fit_result, best_fit_floorplan, room_areas  = self.FL.iterate_optimal_floorplans(seed, n_iter) # 여기서 10번 함

        self.fitness_label.config(text=average_text_result)
        self.best_fitness_label.config(text = best_fit_result)
        self.ok_button.config(state=tk.DISABLED)
        self.draw_on_canvas_metrics(best_fit_floorplan, room_areas,  self.final_canvas) # info best floorplan 하나만 출력



    def initialize_floorplan(self):
        self.simplified_candidates.clear()
        if self.seed is not None:
            floorplan = create_floorplan(self.seed, k = self.num_rooms, options= self.options, reqs = self.reqs) # todo to change plan.create_floorplan
            if self.FL is not None:
                self.draw_on_canvas(self.FL, self.final_canvas)
            else:
                messagebox.showwarning('Error', 'Initialize_room_location First. No possible seed for Adjacency Constraint')
        else:
            messagebox.showwarning('Error', 'init_grid is None')

    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')

    def draw_on_canvas_metrics(self, floorplan, metrics, canvas):
        self.create_path()
        display = self.options.display
        save = self.options.save
        num_rooms = self.options.num_rooms
        if floorplan is not None:
            fig = GridDrawer.draw_plan_with_metrics(floorplan, self.full_path, display=display,
                                                    save=save,
                                                    num_rooms=num_rooms, metrics=metrics)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_on_canvas (self, floorplan, canvas): # draw on the canvas

        self.create_path()
        if floorplan is not None:
            fig = GridDrawer.color_cells_by_value(floorplan, self.full_path, display=False, save=True, num_rooms = self.options.num_rooms)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    # underway rectangularity 확인
    def exchange_cells(self):
        if self.FL is not None:
            self.simplified_floorplan = exchange_protruding_cells(self.FL, 10)
            self.simplified_candidates.append(self.simplified_floorplan)
            self.draw_on_canvas(self.simplified_floorplan, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")




    def build_polygon(self):
        if self.FL is not None:
            grid_polygon = GridPolygon(self.FL) #todo get scale from constraints.ini
            self.room_polygons = grid_polygon.room_polygons
            polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)

            suffix = 'before'
            # todo fitness class를 콜할 때 이 room_polygon을 넘겨주자.
            # Fitness call하기 전에 room_pollygon이 있어야 한다.
            fig  = polygon_exporter.save_polygon_to_png(self.room_polygons, f"room_polygons_{suffix}.png")
            self.show_plot_on_canvas(fig, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    # todo 이것은 하나씩 확인하고자 할 때 쓰인다. 대량의 population을 일으키려면 while이나 for를 이용하여 반복해야 한다.
    #  최종적으로는 population을 일으킬 때 UI 없이 해야 할 것이다.
    #  일단 option에다가 silence 모드를 넣고 이 모듈을 복사해서 silence 하게 해보자.



    def evolve(self):
        # GeneticAlgorithm에 population 전달
        ga = GeneticAlgorithm(list(self.FL.floorplans_dict.values()))

        ga.run(num_generations=2, population_size=10, mutation_rate=0.01)



    def run_iteration(self):
        def generate_average_fitness_text(fits):
            # 2. 고유 식별자를 사용해 결과 저장
            fitness_values = {
                "Adjacency": np.average([fit.adj_satisfaction for fit in fits]),
                "Orientation": np.average([fit.orientation_satisfaction for fit in fits]),
                "Size": np.average([fit.size_satisfaction for fit in fits]),
                "Regularity": np.average([fit.regularity for fit in fits]),
                "Aspect Ratio": np.average([fit.pa_ratio for fit in fits]),
                "Total Fitness": np.average([fit.fitness for fit in fits]),
                "Best Floorplan Fitness": np.max([fit.fitness for fit in fits])
            }
            text_result = "\n".join([f"{key}: {value:.2f}" for key, value in fitness_values.items()])
            text_result = f"Best Floorplan Above\nAverage fitness For the seed:\n{text_result}"
            return text_result

        num_rooms = self.options.num_rooms
        reqs = Req()
        n_iter = self.options.iteration_from_seed

        if self.seed_iteration < n_iter:
            self.ok_button.config(state = tk.DISABLED)

            initial_floorplan = create_floorplan(self.temperary_seed, k= num_rooms, options = self.options, reqs=reqs)
            # self.draw_floorplan(initial_floorplan, self.initial_canvas)
            simplified_floorplan, fit= self.FL.get_optimal_from_initial_floorplan(initial_floorplan)
            fit_result = create_fitness_info(fit)
            self.fitness_label.config(text=f"Fitness Results:\n{fit_result}")
            self.draw_on_canvas(simplified_floorplan, self.final_canvas)
            room_areas = [room.area for room in fit.room_polygons.values()]
            self.draw_on_canvas_metrics(simplified_floorplan, room_areas, self.final_canvas)
            #결과 저장
            # self.floorplans.append( (simplified_floorplan, fit)) # error here MOVED TO LOGIC 0912

            # OK 버튼 활성화
            self.ok_button.config(state=tk.NORMAL)
            self.save_seed_button.config(state=tk.NORMAL)
            self.save_floorplan_button.config(state=tk.NORMAL)

        else:
            self.fitness_label.config(text="Batch processing complete.")
            self.ok_button.config(state=tk.DISABLED)

            # 1. 고유 식별자 생성
            unique_id = generate_unique_id(self.room_seed_dict)

            # label
            fits = [fl[1] for fl in self.floorplans] # 모든 fitness들을 가져와서
            text_result = generate_average_fitness_text(fits)
            self.fitness_label.config(text=text_result)


            # floorplan
            avg_fits  =  [fit.fitness for fit in fits]
            best_fits_index = np.argmax(avg_fits)
            best_fit = self.floorplans[best_fits_index][1]
            max_fit_floorplan = self.floorplans[best_fits_index][0]
            max_fit_result = self.create_fitness_info(best_fit)
            print(f'max_fit_result = {max_fit_result}')
            self.fitness_label.config(text=max_fit_result)

            room_areas = [room.area for room in best_fit.room_polygons.values()]
            self.draw_on_canvas_metrics(max_fit_floorplan, room_areas,  self.final_canvas) # info best floorplan 하나만 출력
            # self.draw_on_canvas(self.seed, self.initial_canvas)

    def next_iteration(self):
        self.seed_iteration += 1
        self.run_iteration()

    def get_optimal_from_initial_floorplan(self, initial_floorplan):
        optimal_candidates = FL.create_candidate_floorplans(initial_floorplan)

        print(f'{len(optimal_candidates)} floorplan candidates generated')
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

        self.draw_on_canvas_metric(optimal_floorplan,  room_areas, self.final_canvas)
        #     fig = GridDrawer.draw_plan_with_metrics(optimal_floorplan, full_path, display=False, save=False, num_rooms=self.num_rooms, metrics=room_areas)
        #     self.show_plot_on_canvas(fig, self.final_canvas)

        fitness_result = self.create_fitness_info(fit)
        # 결과를 문자열로 포맷팅

        # 레이블에 피트니스 결과 표시
        self.fitness_label.config(text=f"Fitness Results:\n{fitness_result}")
        return optimal_floorplan, fit

    def draw_on_canvas_metric(self, floorplan, metrics, canvas):
        self.create_path()
        if floorplan is not None:

            fig = GridDrawer.draw_plan_with_metrics(floorplan, self.full_path, display=self.options.display, save=self.options.save,
                                                  num_rooms=self.num_rooms, metrics=metrics)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")


    # info: important seed로 floorplans를 검색하는 방법
    def get_floorplan_with_seed(self, room_seed_dict):

        unique_id = generate_unique_id(room_seed_dict)
        if unique_id in self.floorplans_dict:
            floorplans = self.floorplans_dict[unique_id]
            return floorplans
        else:
            return None





    def build_graph(self):
        if self.FL is not None:
            build_modules = {
                "1": GraphBuilder.build_graph,
                "2": GraphBuilder.build_weighted_graph,
                "3": GraphBuilder.build_graph_with_inside_cells,
                "4": GraphBuilder.build_graph_connect_4way,
                "9": exit_module
            }
            self.graph = run_selected_module(build_modules, self.FL)
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
            ("Draw Floorplan", self.draw_on_canvas),
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