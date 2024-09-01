from pareto import is_dominated
import random
import numpy as np
from fitness import Fitness
from simplify import count_cascading_cells
from plan import find_parallel_adjacent_cells,assign_cells_to_adjacent_room
from options import Options
from convert import min_max_scaling
class GeneticAlgorithm:
    def __init__(self, population):
        # info: population taken from floorplans_dict {seed:[(fl, ft), (fl, ft], (fl, ft) ... , seed2: [(fl, ft), (fl, ft), ...]} => take only values() part => into list
        #  so population structure:  [(fl, ft),(fl, ft),...(fl, ft)], ...[(fl, ft),(fl, ft),...(fl, ft)]
        self.population = population
        self.options = Options() #todo num_rooms 만 가져오자
    # todo population과 self.population  차이


    def select_parents(self, num_parents):
        # info  self.population은 [fp1, fp2, ...] 형태의 리스트이며, 각 fp는 (np.array, fitness) 형태의 튜플
        #  [fp[1] for fp in self.population]는 전체 튜플을 가져옴. 이는 self.population이 중첩된 리스트 구조를 가지기 때문
        all_floorplans_fitness = [flt for seed_sub in self.population for flt in seed_sub]
        fitness_scores = [flt[1].fitness for flt in all_floorplans_fitness]
        # todo info comment out after testing crossover, 파퓰레이션이 적을 때에는 똑같은 해끼리 부모가 되는 경우가 많아서 일단 이 연산을 배제하고 나중에 다시 붙여서 테스트한다.
        # fitness_scores = min_max_scaling(fitness_scores) # todo fitness 객체를 모두 가지고 있지 말고, 대표값만 가지고 있는 것이 좋겠다.
        parents = random.choices(all_floorplans_fitness, weights=fitness_scores, k=num_parents)
        average_fitness = np.average([flt[1].fitness for flt in all_floorplans_fitness]) # todo to display in the label
        parent_average_fitness = np.average([flt[1].fitness for flt in parents]) # todo to display in the label
        print(f'average fitness value \n all_floorplans = {average_fitness} \n parents = {parent_average_fitness}')
        return parents


    def crossover(self, floorplan_fit1, floorplan_fit2):
        """
        두 플로어플랜을 교차하여 자식 플로어플랜을 생성한다.

        Parameters:
        - floorplan1: 2D numpy array, 첫 번째 부모 플로어플랜
        - floorplan2: 2D numpy array, 두 번째 부모 플로어플랜

        Returns:
        - child_floorplan: 2D numpy array, 생성된 자식 플로어플랜
        """
        floorplan1, floorplan2 = floorplan_fit1[0], floorplan_fit2[0]
        rows, cols = floorplan1.shape
        child_floorplan = np.zeros((rows, cols), dtype=int)

        # 교차점을 임의로 선택 (예: 사분면)
        split_row = random.randint(1, rows - 2)
        split_col = random.randint(1, cols - 2)

        # 사분면을 기준으로 교차
        child_floorplan[:split_row, :split_col] = floorplan1[:split_row, :split_col]
        child_floorplan[split_row:, split_col:] = floorplan2[split_row:, split_col:]

        # 남은 부분 처리
        child_floorplan[:split_row, split_col:] = floorplan1[:split_row, split_col:]
        child_floorplan[split_row:, :split_col] = floorplan2[split_row:, :split_col]


        # 교차 전략
        return child_floorplan

    def mutate(self, floorplan, mutation_rate):
        room_id = random.choice(range(1, self.options.num_rooms + 1))
        if random.random() < mutation_rate:
            return assign_cells_to_adjacent_room(floorplan, room_id)
        return floorplan


    def run(self, num_generations, population_size, mutation_rate):
        for generation in range(num_generations):
            new_population = self.create_new_generation(population_size, mutation_rate)
            self.population.extend(new_population)
            self.population = self.compute_pareto_front()[:population_size]
            best_solution = max(self.population, key=lambda sol: sol[1].adj_satisfaction)
            print(f'Generation {generation}: Best adj_satisfaction = {best_solution[1].adj_satisfaction}')

    def create_new_generation(self, population_size, mutation_rate):
        new_population = []
        parents = self.select_parents(population_size)
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            child1 = self.mutate(child1, mutation_rate)
            child2 = self.mutate(child2, mutation_rate)
            fitness1 = Fitness(child1)
            fitness2 = Fitness(child2)
            new_population.append((child1, fitness1))
            new_population.append((child2, fitness2))
        return new_population

    def compute_pareto_front(self):
        pareto_front = []
        for i, solution_a in enumerate(self.population):
            dominated = False
            for j, solution_b in enumerate(self.population):
                if i != j and is_dominated(solution_a, solution_b):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(solution_a)
        return pareto_front
