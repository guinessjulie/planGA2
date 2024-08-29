import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from fitness import Fitness
from trivial_utils import generate_unique_id

import trivial_utils
from main import GridDrawer, exchange_protruding_cells, categorize_boundary_cells, GraphBuilder, GraphDrawer, run_selected_module,  exit_module
from simplify import exchange_protruding_cells, count_cascading_cells
from plan import create_floorplan, locate_initial_cell
from GridPolygon import GridPolygon
from options import Options
from PolygonExporter import PolygonExporter
from config_reader import load_config
import configparser
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo create population class
# todo use seed to recreate floorplan
# todo let equal thickness function work customize

class FloorplanApp:
    # todo 0818 1. 동일 seed에서 만들어진 floorplan을 10개 이상 가지고 있어보자.
    # info: done: in self.simplified_candidates have diff floorplans from the same initiialized_floorplan. after choosing best simplified floorplans. assigns to self.floorplan to set final result
    # todo 1-1 : create_floorplan과 simplify를 합치자.
    # todo: Area 및 wall thickness 표시
    def __init__(self, root, init_grid, num_rooms, callback):
        self.root = root
        self.root.title("Floorplan UI")
        self.room_polygons = None

        self.init_grid = init_grid
        self.num_rooms = num_rooms
        self.options = Options()
        self.callback = callback  # Main App으로 floorplan을 반환하기 위한 콜백 함수
        self.floorplan = None
        self.floorplans = []
        self.candidates = []
        self.simplified_candidates = [] # todo check with self.candidates.  # info this comes from choose_best_simplified and save directly to floorplan
        self.simplified_floorplan = None
        self.seed = None
        self.initial_cells = None
        self.fit = None
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.option = None
        self.create_widgets()


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
            ("Initial Room Location", self.initialize_room_location),
            # ("Batch Processing", self.get_optimal_from_initial_floorplan), todo 이걸로 바꾸려고 했는데 왜 바꾸려고 했는지를 다시 알아내야 함
            ("Batch Processing", self.run_batch_from_same_seed),
            ('Create Floorplan', self.initialize_floorplan),
            ("Simplify Floorplan", self.exchange_cells),
            ("Choose Most Simplified", self.choose_simplified),
            ("Build Polygon", self.build_polygon),
            ("Draw Floorplan", self.draw_floorplan_menu),
            ("Fitness", self.get_fitness),
            ("Draw Plan Equal Thickness", self.draw_floorplan_menu),
            ("Draw Plan with Value", self.draw_plan_with_values),
            ("Return Floorplan", self.return_floorplan),
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

        # 피트니스 결과를 위한 레이블
        self.fitness_label = tk.Label(bottom_frame, text="Fitness Results", font=("Arial", 14))
        self.fitness_label.pack(pady=10)

        # OK 버튼 추가 및 배치
        self.ok_button = ttk.Button(bottom_frame, text="OK", command=self.next_iteration)
        self.ok_button.pack(side=tk.BOTTOM, pady=20)
        self.ok_button.config(state=tk.DISABLED)  # 첫 시작에는 비활성화

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
            self.draw_on_canvas(self.seed, self.initial_canvas)
        else:
            messagebox.showwarning('Error', 'init_grid is not given')

    def initialize_floorplan(self):
        self.simplified_candidates.clear()
        if self.seed is not None:
            self.floorplan = create_floorplan(self.seed, k = self.num_rooms, options= self.options) # todo to change plan.create_floorplan
            if self.floorplan is not None:
                self.draw_on_canvas(self.floorplan, self.final_canvas)
            else:
                messagebox.showwarning('Error', 'Initialize_room_location First. No possible seed for Adjacency Constraint')
        else:
            messagebox.showwarning('Error', 'init_grid is None')

    def create_path(self):
        self.path = trivial_utils.create_folder_by_datetime()  # todo test
        self.full_path = trivial_utils.create_filename(self.path, 'Plan', '', '', 'png')

    def draw_floorplan_menu(self):
        self.create_path()
        if self.simplified_floorplan is not None:
            fig = GridDrawer.color_cells_by_value(self.simplified_floorplan, self.full_path, display=True, save=False, num_rooms=self.num_rooms)
            self.show_plot_on_canvas(fig,self.initial_canvas)
        elif self.floorplan is not None:
            fig = GridDrawer.color_cells_by_value(self.floorplan, self.full_path, display=True, save=False, num_rooms=self.num_rooms)
            self.show_plot_on_canvas(fig,self.initial_canvas)
        else: messagebox.showwarning('Error', 'Create Floorplan First')


    # todo place canvas parameter to all calling function

    def draw_on_canvas_metrics(self, floorplan, metrics, canvas):
        self.create_path()
        display = self.options.display
        save = self.options.save
        if floorplan is not None:
            fig = GridDrawer.draw_plan_with_metrics(floorplan, self.full_path, display=self.options.display,
                                                    save=self.options.save,
                                                    num_rooms=self.num_rooms, metrics=metrics)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_on_canvas (self, floorplan, canvas): # draw on the canvas

        self.create_path()
        if floorplan is not None:
            fig = GridDrawer.color_cells_by_value(floorplan, self.full_path, display=False, save=True, num_rooms = self.num_rooms)
            self.show_plot_on_canvas(fig, canvas)
        else:
            messagebox.showwarning("Warning", "Create Floorplan First")

    def draw_plan_with_values(self):

        if self.path is None:
            self.create_path()
        # todo: 이렇게 하면 꼬인다. simlified_floorplan을 표시하느냐 floorplan을 표시하느냐 이걸 결정해서 로직을 바꾸자.
        if self.simplified_floorplan is not None:
            floorplan = self.simplified_floorplan
        elif self.floorplan is not None:
            floorplan = self.floorplan
        else:
            messagebox.showwarning("Warning", "Load floorplan first")
            return

        self.draw_plan(floorplan, self.final_canvas)

    def draw_plan(self, floorplan, canvas):

        full_path = trivial_utils.create_filename(self.path, 'Floorplan', '', '', 'png')
        fig = GridDrawer.draw_plan_with_metrics(floorplan, full_path, display=False, save=False, num_rooms=self.num_rooms, metrics=self.fit.room_areas)
        self.show_plot_on_canvas(fig, canvas)

    def draw_padded(self):
        if self.floorplan:
            GridDrawer.draw_plan_padded(self.floorplan)
            messagebox.showinfo("Info", "Plan drawn with padding")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")
    # underway rectangularity 확인
    def exchange_cells(self):
        if self.floorplan is not None:
            self.simplified_floorplan = exchange_protruding_cells(self.floorplan, 10)
            self.simplified_candidates.append(self.simplified_floorplan)
            self.draw_on_canvas(self.simplified_floorplan, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def choose_simplified(self):
        min_cas_len = np.sum(self.floorplan >= 1) # max_cascading_cell_length
        min_cas_list = []
        min_idx = 0
        for idx, fl in enumerate(self.simplified_candidates):
            cas_length, cas_list = count_cascading_cells(fl) # todo cascading_cell의 갯수가 가장 적은 floorplan을 구하자. 이 부분은 exchange_proturding_cells를 여러번 한 후 가장 optimum한 것을 고를 때 필요함
            if cas_length < min_cas_len:
                min_cas_len = cas_length
                min_idx = idx
                max_cas_list = cas_list
        self.floorplan = self.simplified_candidates[min_idx]


    def create_fitness_info(self, fit):
        fitness_values = {
            "Adjacency Satisfaction": fit.adj_satisfaction,
            "Orientation Satisfaction":fit.orientation_satisfaction,
            "Size Satisfaction": fit.size_satisfaction,
            "Rectangularity": fit.rectangularity, # todo property 이기 때문에 method를 반납함. 따라서 Fitness에서 rectangularity를 직접 가지고 있어야 함
            "Room Shape Simplicity": fit.simplicity,
            "Room Regularity": fit.regularity,
            "Squareness Measure": fit.pa_ratio,
            "Total Fitness": fit.fitness
        }
        return fitness_values

    # todo 현재는 일일히 단추를 눌러서 하나씩 하지만 모든 걸 한꺼번에 할 수 있어야 한다.
    def get_fitness(self):
        # 피트니스 값 계산 로직 (예시)
        if self.floorplan is None:
            messagebox('Warning', 'Floorplan not created')
            return
        if self.room_polygons is None:
            messagebox.showwarning('Warning', 'Build Polygon First')
            return

        self.fit = Fitness(self.floorplan, self.num_rooms, self.room_polygons) # todo to change Fitness
        fitness_values = self.create_fitness_info(self.fit)

        # 결과를 문자열로 포맷팅
        fitness_result = "\n".join([f"{key}: {value:.2f}" for key, value in fitness_values.items()])

        # 레이블에 피트니스 결과 표시
        self.fitness_label.config(text=f"Fitness Results:\n{fitness_result}")

    def build_polygon(self):
        if self.floorplan is not None:
            grid_polygon = GridPolygon(self.floorplan) #todo get scale from constraints.ini
            self.room_polygons = grid_polygon.room_polygons
            polygon_exporter = PolygonExporter(grid_polygon.cell_size, grid_polygon.padded_grid.shape, padding_size=1000)

            suffix = 'before'
            # todo fitness class를 콜할 때 이 room_polygon을 넘겨주자.
            # Fitness call하기 전에 room_pollygon이 있어야 한다.
            fig  = polygon_exporter.save_polygon_to_png(self.room_polygons, f"room_polygons_{suffix}.png")
            self.show_plot_on_canvas(fig, self.final_canvas)
        else:
            messagebox.showwarning("Warning", "Load floorplan first")

    def categorize_cells(self):
        if self.floorplan is not None:
            horiz, vert = categorize_boundary_cells(self.floorplan, 1)
            messagebox.showinfo("Info", f"Boundary cells categorized: h = {horiz}, v = {vert}")
        else:
            messagebox.showwarning("Warning", "Load floorplan first")
    # todo 1. create_floorplan =>
    #  2. simplified until no cascading exists. or max 5 times with same_cascading number => Done
    #  3. compare with fitness values. withdraw if one is same as others save into self.floorplans
    #  4. repeate create_floorplan => simplified_floorplan -> self.floorplans
    #  5. batch fitnesses for all floorplans of one seed
    #  6. keep track of finnesses of the seed

    # underway: 1.  일단 running 하는지 확인
    def run_batch_from_same_seed(self):
        self.seed, self.room_seed_dict = locate_initial_cell(self.init_grid, self.num_rooms)
        self.draw_on_canvas(self.seed, self.initial_canvas)
        # self.draw_floorplan(self.seed, self.initial_canvas) # info best floorplan 한 개만 출력해보자
        self.iteration = 0
        self.floorplans_dict = {} #info: structure {seed:floorplans} # floorplans = [(floorplan, fit)]
        self.run_iteration()

    # todo 0828 seed를 변화시켜서 fitness를 통계내어 봅시다.
    def run_iteration(self):
        def create_total_fitness_text(fits):
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

        if self.iteration < 10:
            self.ok_button.config(state = tk.DISABLED)

            initial_floorplan = create_floorplan(self.seed, k= self.num_rooms, options = self.options)
            # self.draw_floorplan(initial_floorplan, self.initial_canvas)
            simplified_floorplan, fit = self.get_optimal_from_initial_floorplan(initial_floorplan)
            self.draw_on_canvas(simplified_floorplan, self.final_canvas)
            room_areas = [room.area for room in fit.room_polygons.values()]
            self.draw_on_canvas_metrics(simplified_floorplan, room_areas, self.final_canvas)
            #결과 저장
            self.floorplans.append( (simplified_floorplan, fit))

            # OK 버튼 활성화
            self.ok_button.config(state=tk.NORMAL)

        else:
            self.fitness_label.config(text="Batch processing complete.")
            self.ok_button.config(state=tk.DISABLED)

            # 1. 고유 식별자 생성
            unique_id = generate_unique_id(self.room_seed_dict)

            # label
            fits = [fl[1] for fl in self.floorplans]
            text_result = create_total_fitness_text(fits)
            self.fitness_label.config(text=text_result)

            # floorplan
            avg_fits  =  [fit.fitness for fit in fits]
            best_fits_index = np.argmax(avg_fits)
            best_fit = self.floorplans[best_fits_index][1]
            max_fit_floorplan = self.floorplans[best_fits_index][0]
            room_areas = [room.area for room in best_fit.room_polygons.values()]
            self.draw_on_canvas_metrics(max_fit_floorplan, room_areas,  self.final_canvas) # info best floorplan 하나만 출력
            # self.draw_on_canvas(self.seed, self.initial_canvas)

    def next_iteration(self):
        self.iteration += 1
        self.run_iteration()

    def get_optimal_from_initial_floorplan(self, initial_floorplan):
        optimal_candidates = self.create_candidate_floorplans(initial_floorplan)

        print(f'{len(optimal_candidates)} floorplan candidates generated')
        # info what self.build_polygon() does

        if len(optimal_candidates) == 1:
            optimal_floorplan = optimal_candidates[0]
            grid_polygon = GridPolygon(optimal_floorplan)
            fit = Fitness(optimal_floorplan, self.num_rooms, grid_polygon.room_polygons)

        else:
            fitnesses={}
            best_fitness = -float('inf')
            optimal_floorplan = None
            best_idx = -1
            for i, fl in enumerate(optimal_candidates):
                grid_polygon = GridPolygon(fl)
                # info what self.get_fitness() does
                fitnesses[i] = Fitness(fl, self.num_rooms, grid_polygon.room_polygons)
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

        fitness_values = self.create_fitness_info(fit)
        # 결과를 문자열로 포맷팅
        fitness_result = "\n".join([f"{key}: {value:.2f}" for key, value in fitness_values.items()])
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


#        self.draw_floorplan_menu()
#        self.get_fitness()
#        self.draw_floorplan_menu()
#        self.draw_plan_with_values()
#        self.return_floorplan()
#        self.root.quit()

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
            print(f'num_cascading_cell = {candidate_cas}')

            if candidate_cas < min_cas:  #
                candidates.append(candidate)
                min_cas = candidate_cas
                num_cas_dict[len(candidates) -1] = candidate_cas
            iteration_count += 1

        candidates = self.remove_duplicate_floorplan(candidates)

        # 가장 작은 num_cas 값을 가진 모든 candidate를 선택
        min_candidates = [candidates[idx] for idx, cas in num_cas_dict.items() if cas == min_cas]

        return min_candidates if min_candidates else [initial_floorplan]

    def remove_duplicate_floorplan(self,candidates):
        unique_candidates = []
        for fl in candidates:
            if not any(np.array_equal(fl, unique) for unique in unique_candidates):
                unique_candidates.append(fl)
        return unique_candidates



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