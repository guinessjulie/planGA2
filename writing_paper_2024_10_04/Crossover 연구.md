- crossover를 하면 방의 인접성이 망쳐저서 다른 식으로 접근
- 방 하나는 parent1에서 나머지는 parent2에서 취해서 서로 합한다음, 나머지 빈 셀이 존재하면 그곳은 인접셀중 하나로 방번호를 정함

그 접근 방식은 매우 흥미롭습니다. 이를 통해 새로운 `floorplan`을 생성할 수 있으며, 다음과 같은 단계로 구현할 수 있습니다:

1. **특정 방을 `parent1`에서 선택**하고, 나머지 방들은 `parent2`에서 가져옵니다.
2. 두 부모의 배열을 합쳐 새로운 `floorplan`을 생성합니다.
3. 새로 생성된 `floorplan`에서 빈 셀(`0`)이 존재하면, 인접한 셀의 방 번호 중 하나를 선택해 빈 셀을 채웁니다.

아래는 이 로직을 구현한 코드입니다:

```python
import numpy as np
import random

# 부모 배열 정의 (이전과 동일)
parent1 = np.array([
    [ 1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  4,  4,  4,  4,  4],
    [ 1,  1,  2,  2,  2,  2,  2,  6,  2,  2,  2,  4,  4,  4,  4,  4],
    [ 1,  1,  1,  1,  1,  1,  1,  6,  2,  2,  2,  4,  4,  4,  4,  4],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  3,  3,  3,  3,  3,  4,  4,  4],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  3,  3,  3,  3,  3,  3,  3,  3],
    [ 1,  1,  1,  1,  1,  7,  7,  5,  5,  3,  3,  3,  3,  3,  3,  3],
    [ 1,  1,  1,  1,  1,  7,  7,  5,  5,  8,  8,  8, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1,  7,  7,  5,  5,  8,  8,  8, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1,  5,  5,  5,  5, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1,  5,  5,  5,  5, -1, -1, -1, -1, -1, -1]
])

parent2 = np.array([
    [ 8,  8,  8,  8,  7,  7,  7,  1,  1,  1,  1,  1,  6,  6,  2,  2],
    [ 8,  8,  8,  8,  7,  7,  7,  1,  1,  1,  1,  1,  6,  6,  2,  2],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1,  6,  6,  6,  6],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1,  6,  6,  6,  6],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1,  3,  3,  3,  3],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1,  3,  3,  3,  3],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1, -1, -1, -1, -1],
    [ 5,  5,  5,  5,  4,  4,  4,  4,  1,  1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1,  4,  4,  4,  4, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1,  4,  4,  4,  4, -1, -1, -1, -1, -1, -1]
])

# 특정 방 하나를 parent1에서 선택, 나머지 방은 parent2에서 선택
def combine_floorplans(parent1, parent2, selected_room):
    combined = np.where(parent1 == selected_room, parent1, parent2)  # parent1에서 선택한 방을 유지, 나머지는 parent2에서
    return combined

# 빈 셀(0)을 인접한 셀의 방 번호로 채우기
def fill_empty_cells(floorplan):
    filled_floorplan = floorplan.copy()
    rows, cols = filled_floorplan.shape
    for i in range(rows):
        for j in range(cols):
            if filled_floorplan[i, j] == 0:  # 빈 셀인 경우
                neighbors = get_neighbors(filled_floorplan, i, j)
                if neighbors:
                    filled_floorplan[i, j] = random.choice(neighbors)  # 인접한 방 번호 중 하나로 채움
    return filled_floorplan

# 인접한 셀들의 방 번호를 가져오는 함수
def get_neighbors(floorplan, i, j):
    neighbors = []
    if i > 0 and floorplan[i-1, j] != 0:
        neighbors.append(floorplan[i-1, j])
    if i < floorplan.shape[0] - 1 and floorplan[i+1, j] != 0:
        neighbors.append(floorplan[i+1, j])
    if j > 0 and floorplan[i, j-1] != 0:
        neighbors.append(floorplan[i, j-1])
    if j < floorplan.shape[1] - 1 and floorplan[i, j+1] != 0:
        neighbors.append(floorplan[i, j+1])
    return neighbors

# 방 번호 3을 parent1에서 선택하고, 나머지는 parent2에서 선택하여 새로운 floorplan 생성
combined_floorplan = combine_floorplans(parent1, parent2, 3)

# 빈 셀을 인접 셀의 방 번호로 채움
final_floorplan = fill_empty_cells(combined_floorplan)
final_floorplan
```

### 설명:

1. **combine_floorplans 함수**: 이 함수는 `parent1`에서 특정 방(예: 방 번호 3)을 선택하고, 나머지 방들은 `parent2`에서 선택하여 새로운 `floorplan`을 만듭니다.
   
2. **fill_empty_cells 함수**: 이 함수는 새로 생성된 `floorplan`에서 빈 셀(`0`)을 찾아 인접한 셀의 방 번호 중 하나를 선택하여 채웁니다.

3. **get_neighbors 함수**: 이 함수는 특정 셀 `(i, j)`의 인접한 셀들의 방 번호를 가져오는 역할을 합니다.

4. **combined_floorplan**: 특정 방 번호(예: 3)를 `parent1`에서 가져오고 나머지 방들을 `parent2`에서 가져온 결과입니다.

5. **final_floorplan**: `

# 방이 여러 덩어리로 나누어진 경우 이를 합치기


### 문제의 원인
- **두 덩어리로 분리된 이유**: 작은 덩어리의 셀들이 물리적으로 큰 덩어리와 떨어져 있어 인접하지 않기 때문에 덩어리가 나뉘어져 있습니다. 이 경우, 단순히 이웃 셀을 탐색해 연결하려고 하면 항상 실패하게 됩니다.

### 해결 방법
작은 덩어리를 큰 덩어리에 병합하기 위해서는 물리적으로 분리된 두 덩어리를 연결하는 방법이 필요합니다. 이를 위해 다음과 같은 전략을 사용할 수 있습니다:

1. **두 덩어리 사이의 경로 찾기**: 작은 덩어리와 큰 덩어리 사이의 최단 경로를 찾아 해당 경로를 방 번호로 채우는 방식으로 병합합니다.

2. **모든 작은 덩어리들을 큰 덩어리와 강제로 연결**: 두 덩어리를 물리적으로 연결하는 경로를 만들어 강제로 하나의 덩어리로 병합합니다.

### 최단 경로 기반 병합 구현
두 덩어리 사이의 최단 경로를 찾아서 병합하는 방법을 간단하게 구현할 수 있습니다.

```python
import numpy as np
from scipy.ndimage import label
from scipy.spatial import distance
from queue import Queue

# 두 덩어리 사이의 최단 경로를 찾아서 병합하는 함수
def find_shortest_path_and_merge(floorplan, room_number):
    binary_floorplan = (floorplan == room_number).astype(int)
    labeled_array, num_features = label(binary_floorplan)

    if num_features <= 1:
        return floorplan

    # 각 덩어리의 좌표들 가져오기
    component_coords = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]

    # 가장 큰 덩어리 선택
    largest_component = max(component_coords, key=len)

    # 작은 덩어리들을 큰 덩어리에 병합
    for small_component in component_coords:
        if np.array_equal(small_component, largest_component):
            continue

        # 작은 덩어리와 큰 덩어리 사이의 최소 거리 계산
        shortest_dist = float('inf')
        closest_pair = None

        for small_coord in small_component:
            for large_coord in largest_component:
                dist = np.linalg.norm(small_coord - large_coord)
                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_pair = (small_coord, large_coord)

        # 두 덩어리 사이의 경로 채우기
        if closest_pair:
            small_coord, large_coord = closest_pair
            path = draw_path(small_coord, large_coord)
            for (x, y) in path:
                floorplan[x, y] = room_number

    return floorplan

# 두 좌표 사이의 직선 경로를 그리는 함수
def draw_path(start, end):
    path = []
    x1, y1 = start
    x2, y2 = end
    while (x1, y1) != (x2, y2):
        if x1 < x2:
            x1 += 1
        elif x1 > x2:
            x1 -= 1
        if y1 < y2:
            y1 += 1
        elif y1 > y2:
            y1 -= 1
        path.append((x1, y1))
    return path

# 덩어리를 병합한 결과를 확인
final_floorplan = find_shortest_path_and_merge(combined_floorplan, 1)
final_floorplan
```

### 설명:
- **find_shortest_path_and_merge** 함수는 작은 덩어리와 큰 덩어리 사이의 최단 경로를 찾아 방 번호로 채우고, 두 덩어리를 하나로 병합합니다.

- **draw_path** 함수는 두 좌표 사이의 직선 경로를 계산해 그 경로를 방 번호로 채웁니다.

이 방법을 사용하면 분리된 방 덩어리들을 연결해 하나의 덩어리로 만들 수 있습니다.

Crossover 후 자식의 `fitness` 값이 대부분의 부모보다 안 좋아지는 경우는 유전 알고리즘에서 자주 발생할 수 있는 문제입니다. 이를 해결하기 위해 몇 가지 방법을 고려할 수 있습니다.

### 1. **Elitism (엘리트 보존)**
   - **방법**: 엘리트 보존은 가장 `fitness` 값이 좋은 부모를 그대로 다음 세대로 전달하는 방법입니다. 이를 통해, 최상의 `fitness` 값이 유전 과정에서 손실되지 않도록 보장할 수 있습니다.
   - **구현**: 부모 중 `fitness` 값이 가장 좋은 개체를 몇 개 선택하여, 자식을 생성한 후에도 이들 중 일부를 그대로 다음 세대로 넘깁니다.

### 2. **Fitness-Based Selection (적응도 기반 선택)**
   - **방법**: 자식의 `fitness` 값이 부모보다 낮으면, 그 자식을 버리고 부모 중 하나를 그대로 유지하거나 새로운 자식을 생성하는 방법입니다.
   - **구현**: 자식의 `fitness` 값을 부모의 `fitness` 값과 비교한 후, 자식이 부모보다 `fitness` 값이 낮으면 그 자식을 교체합니다.

### 3. **Mutation (변이)**
   - **방법**: Crossover 후 `fitness` 값이 나빠진 자식에게 변이를 적용하여 `fitness` 값을 개선할 수 있는 가능성을 제공합니다.
   - **구현**: 자식의 `fitness` 값이 부모보다 나빠지면 변이를 강하게 적용해 새롭게 변화된 개체를 만듭니다.

### 4. **Hybrid Crossover Methods (혼합 교차 방법)**
   - **방법**: Crossover 방법을 다양하게 혼합하여 적용해봅니다. 예를 들어, 단순한 `one-point crossover` 대신 `uniform crossover`나 `multi-point crossover`를 적용해 다양한 결과를 도출합니다.
   - **구현**: 여러 가지 Crossover 방법을 시도해 보고, 그중 `fitness` 값을 가장 잘 보존하거나 개선하는 방법을 채택합니다.

### 5. **Local Search (지역 탐색)**
   - **방법**: Crossover 후 자식이 생성된 후, 자식의 `fitness` 값을 지역적으로 탐색하며 개선할 수 있는지 확인합니다.
   - **구현**: 자식이 생성된 후, 주변의 다른 해를 탐색하면서 `fitness` 값이 개선되는지 확인하고, 개선된 해를 채택합니다.

### 6. **Adaptive Crossover (적응형 교차)**
   - **방법**: 부모의 `fitness` 값에 따라 Crossover 비율을 동적으로 조정하는 방법입니다. `fitness` 값이 좋은 부모의 유전자를 더 많이 물려주도록 조정합니다.
   - **구현**: 부모의 `fitness` 값에 비례하여 Crossover 비율을 결정하거나, 높은 `fitness` 값을 가진 부모의 유전자를 자식이 더 많이 물려받도록 합니다.

### 7. **Recombination Repair (재조합 복구)**
   - **방법**: Crossover 후 자식의 `fitness` 값이 나빠졌을 때, 이를 복구하기 위한 특정 복구 알고리즘을 적용할 수 있습니다. 예를 들어, 문제의 제약 조건을 만족하도록 자식을 수정하거나 자식의 구조를 재배치합니다.

### 예시: Elitism 적용

```python
def evolve_population(parents, crossover_fn, mutation_fn, fitness_fn, elitism_rate=0.1):
    # 엘리트 보존: 상위 N%의 부모를 그대로 다음 세대로 보존
    num_elites = int(elitism_rate * len(parents))
    parents_sorted_by_fitness = sorted(parents, key=fitness_fn, reverse=True)
    next_generation = parents_sorted_by_fitness[:num_elites]
    
    # 나머지 자식은 Crossover와 Mutation을 통해 생성
    while len(next_generation) < len(parents):
        parent1, parent2 = select_parents(parents)
        child = crossover_fn(parent1, parent2)
        child = mutation_fn(child)
        
        # 자식의 fitness가 부모보다 낮으면 부모를 보존
        if fitness_fn(child) > min(fitness_fn(parent1), fitness_fn(parent2)):
            next_generation.append(child)
        else:
            next_generation.append(parent1 if fitness_fn(parent1) > fitness_fn(parent2) else parent2)
    
    return next_generation
```

### 결론:
이 방법들을 통해, Crossover 후 자식의 `fitness` 값이 부모보다 나빠지는 문제를 다양한 방식으로 처리할 수 있습니다. 상황에 따라 적합한 방법을 선택해 유전 알고리즘의 성능을 최적화할 수 있습니다.

# Crossover 전략  생각 정리
1. 두 개의 Floorplan의 각 방들 중에서 겹치는 영역이 가장 작은 방을 선택하여, Crossover 한다. 랜덤으로 하는 경우 서로 다른 영역의 방의 교환으로 인해 겹치거나 나뉘어지는 참사가 생긴다.  겹치는 영역이 큰 경우 방의 크기가 커지고 사라지는 방이 생기는 경우가 많아진다.  => 겹치는 영역이 가장 작은 방을 선택하는 경우 parent1과 동일한 해를 만들어낸다. 할 필요없음.
2. 엘리트 선택 방식을 사용하여, 부모의 해가 자식해보다 더 좋으면 자식은 드롭하고 원래 부모를 선택한다. 이렇게 하면 부모의 개체수가 너무 많아지는 문제가 생긴다. 그러므로 weight을 신경써야 함
3. 모든 룸에 대해서 다 이걸 하면 어떨까?
# 다른 대안들
문제를 다시 정리해보겠습니다. 현재 시도한 방법으로는 `child`가 `parent1`과 거의 동일한 해를 갖게 되어, 실제로 부모 간의 유의미한 교차가 일어나지 않는 문제가 발생하고 있습니다. 이제 **교차 영역을 선택**하고 **빈 공간을 인접한 방으로 채우는 방식**을 좀 더 다르게 접근해 보겠습니다.

### 문제 이해:

- **교차 영역**: 부모의 교차된 영역을 특정 방으로 설정하려고 했으나, 결과적으로 `parent1`과 비슷한 구조가 되는 문제가 있었습니다.

### 새로운 접근 방식:

- **교차 영역 설정**: `parent1`과 `parent2`에서 **선택된 방에 대한 교차 영역만을 해당 방 번호로** 설정하고, 나머지 영역은 `parent1` 또는 `parent2`에서 교차되지 않은 부분을 유지합니다.
    
- **빈 공간 채우기**: 빈 공간을 채울 때, 인접한 방 번호를 기준으로 선택하되 **임의로 방을 설정**할 수 있는 방식을 사용해 다양성을 부여합니다.
    

### 새로운 로직:

1. **교차 영역 선택**: 교차된 영역을 찾고, 그 부분을 자식 플로어플랜에서 설정합니다.
2. **빈 공간 채우기**: 교차 영역 이외의 부분에서 빈 공간을 **인접한 방 번호** 또는 **임의의 방 번호**로 채우도록 합니다. 즉, `parent1`과 `parent2`에서 채워질 방이 랜덤으로 섞이도록 합니다.

### 코드:

python

코드 복사
``` python
import numpy as np
from scipy.ndimage import label

def combine_floorplans(self, parent1, parent2, selected_room):
    """
    두 부모의 교차된 영역을 선택된 방으로 만들고,
    나머지 빈 공간은 인접 방 또는 임의 방으로 채운다.
    """
    # 교차 영역(intersection) 설정: 두 부모에서 방 번호가 동일한 교차 영역을 선택된 방으로 설정
    intersection = (parent1 == selected_room) & (parent2 == selected_room)
    child = np.zeros_like(parent1)  # 빈 배열 생성
    child[intersection] = selected_room  # 교차된 영역을 선택된 방 번호로 설정
    
    # 교차되지 않은 영역 설정: 나머지 부분을 인접한 방 또는 임의로 선택
    union = (parent1 == selected_room) | (parent2 == selected_room)
    union_minus_intersection = union & ~intersection
    
    # 임의의 방식으로 union - intersection 부분을 채우기
    child = self.fill_union_area_random(child, union_minus_intersection, parent1, parent2, selected_room)

    return child

def fill_union_area_random(self, child, union_area, parent1, parent2, selected_room):
    """
    교차 영역 외의 빈 공간을 인접 방 번호 또는 임의 방 번호로 채운다.
    """
    rows, cols = child.shape
    for i in range(rows):
        for j in range(cols):
            if union_area[i, j] and child[i, j] == 0:  # 교차되지 않은 빈 셀 찾기
                # 인접한 방 번호들 가져오기
                neighbors = self.get_neighbors(child, i, j, parent1, parent2)
                if neighbors:
                    # 인접한 방 번호들 중 랜덤으로 선택
                    child[i, j] = np.random.choice(neighbors)
                else:
                    # 인접 방이 없으면 parent1 또는 parent2의 방 번호 중 임의 선택
                    child[i, j] = parent1[i, j] if np.random.rand() > 0.5 else parent2[i, j]
    return child

def get_neighbors(self, child, x, y, parent1, parent2):
    """
    특정 셀의 인접한 셀들의 방 번호를 가져온다.
    """
    neighbors = []
    if x > 0 and child[x-1, y] == 0:
        neighbors.append(parent1[x-1, y] if parent1[x-1, y] != 0 else parent2[x-1, y])
    if x < child.shape[0] - 1 and child[x+1, y] == 0:
        neighbors.append(parent1[x+1, y] if parent1[x+1, y] != 0 else parent2[x+1, y])
    if y > 0 and child[x, y-1] == 0:
        neighbors.append(parent1[x, y-1] if parent1[x, y-1] != 0 else parent2[x, y-1])
    if y < child.shape[1] - 1 and child[x, y+1] == 0:
        neighbors.append(parent1[x, y+1] if parent1[x, y+1] != 0 else parent2[x, y+1])
    neighbors = [n for n in neighbors if n != 0]  # 0이 아닌 방 번호들만 선택
    return neighbors

def crossover(self, floorplan_fit1, floorplan_fit2, reqs, num_rooms=8):
    """
    두 플로어플랜을 교차하여 자식 플로어플랜을 생성한다.

    Parameters:
    - floorplan1: tuple of (2D numpy array, fitness) 첫 번째 부모 플로어플랜
    - floorplan2: tuple of (2D numpy array, fitness) 두 번째 부모 플로어플랜

    Returns:
    - child_floorplan: 2D numpy array, 생성된 자식 플로어플랜
    """
    parent1, parent2 = floorplan_fit1[0], floorplan_fit2[0]
    parent1_fit, parent2_fit = floorplan_fit1[1], floorplan_fit2[1]

    overlapped_rooms = overlapped_room_list(parent1, parent2, num_rooms)
    while overlapped_rooms:
        selected_room = overlapped_rooms.pop()
        child1 = self.combine_floorplans(parent1, parent2, selected_room)  # 교차된 영역 설정 및 빈 공간 채우기
        
        if is_room_split(child1, selected_room):  # 방이 두 덩어리로 나뉘어졌는지 확인
            print(f'Room {selected_room} is split after crossover, skipping this configuration.')
            continue  # 덩어리가 나뉘어졌으면, 다음 방 선택으로 넘어감
        
        grid_polygon1 = GridPolygon(child1)
        child1_fitness = Fitness(child1, num_rooms, grid_polygon1.room_polygons, reqs)
        
        if child1_fitness.fitness > parent1_fit.fitness:
            return (child1, child1_fitness)
        print(f'child1 {child1_fitness.fitness} <= parent1 {parent1_fit.fitness}, skipping this configuration.')

    return floorplan_fit1

```
### 요약:

1. **교차 영역**: 교차된 영역만 해당 방으로 설정하고 나머지 부분은 빈 공간으로 유지.
2. **빈 공간 채우기**: 교차되지 않은 빈 공간을 인접한 방 번호로 채우되, 인접 방이 없을 경우 `parent1` 또는 `parent2`에서 임의로 선택.
3. **랜덤 요소 추가**: 임의로 방 번호를 선택하는 방식으로 다양성을 부여함.

이 방법을 사용하면 `child`가 `parent1`과 동일하게 되는 문제를 방지하면서, 두 부모의 교차된 부분을 효과적으로 자식 플로어플랜에 반영할 수 있습니다.

# 두번째 제안

이해했습니다. 현재 시도한 방법들이 원하는 결과를 주지 않고, 특히 자식이 부모와 동일해지는 문제가 발생하고 있군요. 이를 해결하기 위해 몇 가지 다른 접근 방식을 고려해볼 수 있습니다. 아래에서는 다양한 **교차(Crossover)** 전략을 제안하고, 각 방법에 대한 설명과 구현 예시를 제공합니다.

---

## 제안할 새로운 교차(Crossover) 방법들

1. **Uniform Crossover (균등 교차)**
2. **Subsection Swap Crossover (부분 영역 교환)**
3. **Room-Based Swap Crossover (방 기반 교환)**
4. **Graph-Based Crossover (그래프 기반 교차)**
5. **Mask-Based Crossover (마스크 기반 교차)**

각 방법에 대해 자세히 설명하고, 구현 방법을 제시하겠습니다.

---

## 1. Uniform Crossover (균등 교차)

### 개념 설명

- **아이디어**: 각 셀에 대해 부모 중 어느 쪽의 값을 선택할지 확률적으로 결정합니다.
- **장점**:
  - 전역적으로 다양한 조합을 생성할 수 있습니다.
  - 무작위성이 높아 다양한 해를 탐색할 수 있습니다.
- **단점**:
  - 방이 분할될 가능성이 있으므로, 후처리로 연결성을 보장해야 합니다.

### 구현 단계

1. **무작위 마스크 생성**: 부모의 플로어플랜과 동일한 크기의 무작위 바이너리 마스크를 생성합니다.
2. **자식 플로어플랜 생성**: 마스크를 사용하여 각 셀에 대해 부모 중 하나의 값을 선택합니다.
3. **연결성 검사 및 복구**:
   - 생성된 자식 플로어플랜에서 각 방의 연결성을 검사합니다.
   - 분할된 방이 있으면, 연결성 복구 알고리즘을 적용하여 방을 하나로 합칩니다.

### 구현 예시

```python
import numpy as np
from scipy.ndimage import label

def uniform_crossover(self, parent1, parent2, num_rooms):
    """
    균등 교차를 통해 자식 플로어플랜을 생성합니다.
    """
    # 무작위 마스크 생성
    mask = np.random.randint(0, 2, size=parent1.shape)
    
    # 마스크를 사용하여 자식 생성
    child = np.where(mask == 1, parent1, parent2)
    
    # 각 방의 연결성 검사 및 복구
    for room_number in range(1, num_rooms + 1):
        child = self.ensure_room_connectivity(child, room_number)
    
    return child

def ensure_room_connectivity(self, floorplan, room_number):
    """
    특정 방의 연결성을 보장합니다.
    """
    binary_floorplan = (floorplan == room_number).astype(int)
    labeled_array, num_features = label(binary_floorplan)
    
    if num_features <= 1:
        return floorplan  # 이미 연결되어 있음
    
    # 가장 큰 연결된 구성요소 유지
    largest_component_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    floorplan[labeled_array != largest_component_label] = 0  # 분리된 부분 제거
    
    # 제거된 부분을 인접 방으로 채우기
    floorplan = self.fill_removed_spaces(floorplan)
    
    return floorplan

def fill_removed_spaces(self, floorplan):
    """
    제거된 공간을 인접한 방으로 채웁니다.
    """
    rows, cols = floorplan.shape
    for i in range(rows):
        for j in range(cols):
            if floorplan[i, j] == 0:
                neighbors = self.get_adjacent_rooms(floorplan, i, j)
                if neighbors:
                    floorplan[i, j] = max(set(neighbors), key=neighbors.count)
    return floorplan

def get_adjacent_rooms(self, floorplan, x, y):
    """
    특정 좌표의 인접한 방 번호들을 반환합니다.
    """
    neighbors = []
    if x > 0:
        neighbors.append(floorplan[x-1, y])
    if x < floorplan.shape[0] - 1:
        neighbors.append(floorplan[x+1, y])
    if y > 0:
        neighbors.append(floorplan[x, y-1])
    if y < floorplan.shape[1] - 1:
        neighbors.append(floorplan[x, y+1])
    return [n for n in neighbors if n > 0]
```

### 설명

- **`uniform_crossover`**:
  - 무작위 마스크를 사용하여 부모들의 셀을 랜덤하게 선택하여 자식을 생성합니다.
- **`ensure_room_connectivity`**:
  - 각 방의 연결성을 검사하고, 분할된 경우 가장 큰 구성요소를 유지하고 나머지를 제거합니다.
- **`fill_removed_spaces`**:
  - 제거된 공간을 주변 방들로 채워서 빈 공간이 없도록 합니다.
- **`get_adjacent_rooms`**:
  - 특정 셀의 인접한 방 번호들을 수집합니다.

### 기대 효과

- 다양한 조합의 자식 플로어플랜을 생성할 수 있으며, 후처리를 통해 방의 연결성을 보장합니다.
- 무작위성이 높아 탐색 공간을 넓게 커버할 수 있습니다.

---

## 2. Subsection Swap Crossover (부분 영역 교환)

### 개념 설명

- **아이디어**: 플로어플랜을 임의의 부분 영역(subsection)으로 나누고, 해당 영역을 부모들 간에 교환합니다.
- **장점**:
  - 지역적인 변화를 주면서 전체 구조를 크게 훼손하지 않습니다.
  - 방의 분할을 최소화할 수 있습니다.
- **단점**:
  - 교환 영역이 방의 중간을 가로지르는 경우, 방이 분할될 수 있으므로 후처리가 필요합니다.

### 구현 단계

1. **교환 영역 결정**:
   - 플로어플랜 내에서 임의의 직사각형 영역을 선택합니다.
2. **영역 교환**:
   - 부모1의 해당 영역을 부모2의 동일한 위치의 영역과 교환합니다.
3. **연결성 검사 및 복구**:
   - 교환된 영역 내에서 방의 연결성을 확인하고, 필요한 경우 복구합니다.

### 구현 예시

```python
import numpy as np
from scipy.ndimage import label

def subsection_swap_crossover(self, parent1, parent2, num_rooms):
    """
    부분 영역 교환을 통해 자식 플로어플랜을 생성합니다.
    """
    rows, cols = parent1.shape
    
    # 교환 영역의 좌표 결정
    x1, x2 = sorted(np.random.choice(range(rows), 2, replace=False))
    y1, y2 = sorted(np.random.choice(range(cols), 2, replace=False))
    
    # 부모들의 복사본 생성
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # 영역 교환
    child1[x1:x2, y1:y2] = parent2[x1:x2, y1:y2]
    child2[x1:x2, y1:y2] = parent1[x1:x2, y1:y2]
    
    # 연결성 검사 및 복구
    child1 = self.check_and_fix_connectivity(child1, num_rooms)
    child2 = self.check_and_fix_connectivity(child2, num_rooms)
    
    # 더 나은 fitness를 가진 자식 선택 또는 둘 다 반환
    return child1, child2

def check_and_fix_connectivity(self, floorplan, num_rooms):
    """
    모든 방의 연결성을 확인하고 복구합니다.
    """
    for room_number in range(1, num_rooms + 1):
        floorplan = self.ensure_room_connectivity(floorplan, room_number)
    return floorplan

# ensure_room_connectivity 및 관련 함수들은 이전과 동일하게 사용
```

### 설명

- **교환 영역 결정**:
  - 플로어플랜 내에서 임의의 직사각형 영역을 선택하여 교환합니다.
- **영역 교환 및 복사본 생성**:
  - 부모 플로어플랜의 해당 영역을 교환하여 두 개의 자식을 생성합니다.
- **연결성 검사 및 복구**:
  - 교환으로 인해 방이 분할되었을 가능성을 고려하여, 모든 방의 연결성을 확인하고 복구합니다.

### 기대 효과

- 플로어플랜의 일부만 변경되므로, 구조적인 안정성을 유지하면서도 다양성을 확보할 수 있습니다.
- 교환 영역의 크기와 위치를 조절하여 변이의 정도를 조절할 수 있습니다.

---

## 3. Room-Based Swap Crossover (방 기반 교환)

### 개념 설명

- **아이디어**: 특정 방 전체를 부모들 간에 교환합니다.
- **장점**:
  - 방의 연결성이 보장되며, 구조적인 변화가 명확합니다.
  - 방의 위치와 크기가 변경되어 다양한 조합을 생성합니다.
- **단점**:
  - 교환된 방이 다른 방과 겹치는 경우 충돌이 발생할 수 있으므로 처리 필요.

### 구현 단계

1. **교환할 방 선택**:
   - 교환할 방 번호를 선택합니다.
2. **방의 영역 추출 및 교환 가능 여부 확인**:
   - 선택된 방의 위치와 크기를 확인하고, 교환 시 다른 방과 충돌이 없는지 검사합니다.
3. **방 교환 및 업데이트**:
   - 충돌이 없다면 방을 교환하고, 필요한 경우 주변 영역을 조정합니다.

### 구현 예시

```python
import numpy as np

def room_based_swap_crossover(self, parent1, parent2, num_rooms):
    """
    방 기반 교환을 통해 자식 플로어플랜을 생성합니다.
    """
    child = parent1.copy()
    
    # 교환할 방 선택
    room_number = np.random.randint(1, num_rooms + 1)
    
    # 부모2에서 해당 방의 위치 추출
    room_mask = (parent2 == room_number)
    
    # 충돌 검사: 부모2의 방 영역이 자식에서 다른 방과 겹치는지 확인
    conflict = np.any((room_mask) & (child != room_number) & (child != 0))
    
    if not conflict:
        # 자식에서 해당 방 영역 업데이트
        child[child == room_number] = 0  # 기존 방 제거
        child[room_mask] = room_number  # 새로운 위치에 방 배치
        
        # 빈 공간 채우기
        child = self.fill_empty_spaces(child)
        
        return child
    else:
        # 충돌 발생 시 교차하지 않고 부모1 반환 또는 재시도
        return parent1

def fill_empty_spaces(self, floorplan):
    """
    빈 공간을 주변 방으로 채웁니다.
    """
    rows, cols = floorplan.shape
    for i in range(rows):
        for j in range(cols):
            if floorplan[i, j] == 0:
                neighbors = self.get_adjacent_rooms(floorplan, i, j)
                if neighbors:
                    floorplan[i, j] = max(set(neighbors), key=neighbors.count)
    return floorplan
```

### 설명

- **방 선택 및 교환**:
  - 부모2에서 랜덤하게 방을 선택하여 자식에 적용합니다.
- **충돌 검사**:
  - 선택한 방을 자식에 적용할 때 다른 방과 겹치는지 확인하여 충돌을 방지합니다.
- **빈 공간 채우기**:
  - 방 교환으로 인해 생긴 빈 공간을 주변 방으로 채워 플로어플랜을 완성합니다.

### 기대 효과

- 방 단위로 교환하므로 구조적인 변화를 효과적으로 줄 수 있습니다.
- 충돌 검사를 통해 유효한 플로어플랜을 생성합니다.
- 여러 번의 시도로 다양한 조합을 생성할 수 있습니다.

---

## 4. Graph-Based Crossover (그래프 기반 교차)

### 개념 설명

- **아이디어**: 플로어플랜을 그래프로 표현하여 교차 연산을 수행합니다.
- **장점**:
  - 공간의 연결성과 인접성을 자연스럽게 표현할 수 있습니다.
  - 고급 연산을 통해 복잡한 교차를 수행할 수 있습니다.
- **단점**:
  - 구현 복잡도가 높고, 추가적인 그래프 구조가 필요합니다.

### 구현 단계

1. **플로어플랜을 그래프로 변환**:
   - 각 방을 노드로, 방들의 인접 관계를 엣지로 표현합니다.
2. **그래프 교차 연산 수행**:
   - 부모 그래프들의 부분 그래프를 교환하거나, 그래프 병합 연산을 수행합니다.
3. **그래프를 다시 플로어플랜으로 변환**:
   - 수정된 그래프 구조를 2D 플로어플랜 형태로 변환합니다.

### 구현 예시

이 방법은 복잡도가 높으므로, 간단한 예시를 제시합니다.

```python
import networkx as nx

def graph_based_crossover(self, parent1, parent2):
    """
    그래프 기반 교차를 통해 자식 플로어플랜을 생성합니다.
    """
    # 부모 플로어플랜을 그래프로 변환
    graph1 = self.floorplan_to_graph(parent1)
    graph2 = self.floorplan_to_graph(parent2)
    
    # 랜덤하게 서브그래프 선택
    subgraph1 = self.random_subgraph(graph1)
    subgraph2 = self.random_subgraph(graph2)
    
    # 서브그래프 교환
    child_graph = self.exchange_subgraphs(graph1, graph2, subgraph1, subgraph2)
    
    # 그래프를 플로어플랜으로 변환
    child_floorplan = self.graph_to_floorplan(child_graph)
    
    return child_floorplan

def floorplan_to_graph(self, floorplan):
    """
    플로어플랜을 그래프로 변환합니다.
    """
    # 구현 필요: 각 방을 노드로, 인접 관계를 엣지로 표현
    pass

def random_subgraph(self, graph):
    """
    그래프에서 랜덤한 서브그래프를 선택합니다.
    """
    # 구현 필요: 그래프에서 일부 노드와 엣지를 선택
    pass

def exchange_subgraphs(self, graph1, graph2, subgraph1, subgraph2):
    """
    두 그래프의 서브그래프를 교환합니다.
    """
    # 구현 필요: 그래프에서 서브그래프를 교체
    pass

def graph_to_floorplan(self, graph):
    """
    그래프를 플로어플랜으로 변환합니다.
    """
    # 구현 필요: 그래프 구조를 2D 배열로 변환
    pass
```

### 설명

- 이 방법은 고급 주제이며, 실제 구현에는 많은 고려사항이 필요합니다.
- 그래프 이론과 알고리즘을 활용하여 보다 복잡한 교차 연산을 수행할 수 있습니다.

### 기대 효과

- 플로어플랜의 구조와 인접성을 효과적으로 관리할 수 있습니다.
- 복잡한 교차 연산을 통해 다양한 플로어플랜을 생성할 수 있습니다.

---

## 5. Mask-Based Crossover (마스크 기반 교차)

### 개념 설명

- **아이디어**: 특정 패턴의 마스크를 사용하여 부모들의 특정 영역을 선택적으로 결합합니다.
- **장점**:
  - 다양한 마스크 패턴을 사용하여 여러 형태의 교차를 시도할 수 있습니다.
  - 방의 분할을 최소화하면서도 다양성을 확보합니다.
- **단점**:
  - 적절한 마스크 패턴을 선택하는 것이 중요합니다.

### 구현 단계

1. **마스크 생성**:
   - 스트라이프, 체스보드 등 다양한 패턴의 마스크를 생성합니다.
2. **마스크를 통한 교차**:
   - 마스크에 따라 부모들의 셀을 선택하여 자식을 생성합니다.
3. **연결성 검사 및 복구**:
   - 방의 연결성을 확인하고 복구합니다.

### 구현 예시

```python
import numpy as np

def mask_based_crossover(self, parent1, parent2, num_rooms):
    """
    마스크 기반 교차를 통해 자식 플로어플랜을 생성합니다.
    """
    rows, cols = parent1.shape
    
    # 체스보드 패턴 마스크 생성
    mask = np.indices((rows, cols)).sum(axis=0) % 2
    
    # 마스크를 사용하여 자식 생성
    child = np.where(mask == 0, parent1, parent2)
    
    # 연결성 검사 및 복구
    for room_number in range(1, num_rooms + 1):
        child = self.ensure_room_connectivity(child, room_number)
    
    return child
```

### 설명

- **마스크 생성**:
  - 체스보드 패턴을 사용하여 부모들의 셀을 교차합니다.
  - 다른 패턴(예: 스트라이프, 원형 등)을 사용하여 다양한 교차를 시도할 수 있습니다.
- **연결성 검사 및 복구**:
  - 방이 분할되었을 가능성을 고려하여 연결성을 확인하고 복구합니다.

### 기대 효과

- 다양한 패턴의 마스크를 통해 다양한 형태의 자식을 생성할 수 있습니다.
- 구조적인 안정성을 유지하면서도 변화를 줄 수 있습니다.

---

## 결론 및 추천

- **단순성과 효과성**을 고려할 때, **Subsection Swap Crossover**와 **Room-Based Swap Crossover**가 구현과 적용에 적합할 것으로 보입니다.
- **Uniform Crossover**와 **Mask-Based Crossover**는 무작위성이 높아 탐색 공간을 넓게 커버할 수 있지만, 후처리 과정이 중요합니다.
- **Graph-Based Crossover**는 복잡하지만, 더 정교한 플로어플랜 생성을 위해 고려해볼 수 있습니다.

**제안**: 먼저 **Subsection Swap Crossover**를 시도해보고, 결과를 평가한 후 다른 방법들을 순차적으로 적용해보는 것을 추천드립니다. 각 방법의 결과를 비교하여 최적의 교차 방법을 선택하시면 좋을 것 같습니다.

필요하신 부분에 대해 더 자세한 설명이나 구현 지원이 필요하시면 언제든지 말씀해주세요!