
### Title

"Automated Generation of Architectural Floor Plans Using Python: A Computational Approach"

### Abstract

This paper presents a computational method for the automated generation of architectural floor plans using Python. The approach leverages a grid-based algorithm to place rooms and assign colors, ensuring the validity and aesthetic coherence of the resulting floor plan. The implementation details and experimental results demonstrate the efficacy of the method, offering a viable tool for architects and designers in the early stages of building design.

### Introduction

The generation of architectural floor plans is a crucial step in the building design process. Traditionally, this task has been manual, requiring significant time and expertise. With advancements in computational methods, there is an opportunity to automate this process, enhancing efficiency and creativity. This paper introduces a Python-based approach to automatically generate floor plans, utilizing grid-based algorithms and parallel processing to ensure scalability and accuracy.

### Methodology

#### Overview

The methodology involves initializing a grid representing the floor plan, placing a specified number of colored cells, and expanding these cells to form distinct rooms. The process ensures that adjacent cells maintain continuity and adhere to architectural constraints.

#### Grid Initialization

The grid is initialized using a provided floor shape, where cells marked as '1' are considered valid spaces for rooms, and '0' represents non-buildable areas.

```python

def to_np_array(grid):

    m, n = len(grid), len(grid[0]) if grid else 0

    np_arr = np.full((m, n), -1, dtype=int)

    return np.where(np.array(grid) == 1, 0, -1)

```

#### Placing Initial Colors

A set number of initial cells are randomly assigned colors, representing different rooms or areas within the floor plan.

```python

def place_k_colors_on_grid(grid_arr, k):

    colors_placed = 0

    cells_coords = set()

    coloring_grid = grid_arr.copy()

    m, n = grid_arr.shape

    while colors_placed < k:

        row, col = random.randint(0, m - 1), random.randint(0, n - 1)

        if coloring_grid[row, col] == 0:

            coloring_grid[row, col] = colors_placed + 1

            cells_coords.add((row, col))

            colors_placed += 1

    return coloring_grid, cells_coords

```

#### Room Expansion

The algorithm iteratively expands each colored cell to adjacent vacant cells, ensuring that the expansion maintains architectural coherence by checking cell validity and neighbor constraints.

```python

def place_room(floorplan, obtainable_cells):

    # Implementation details for expanding rooms

    # ...

    return floorplan

```

### Experiments and Results

To validate the proposed method, several experiments were conducted on grids of varying sizes and complexities. The results demonstrate the algorithm's ability to generate coherent and aesthetically pleasing floor plans.

#### Example Experiment

```python

def exe_build_floorplan():

    m, n, k = 5, 5, 4

    floorshape = [

        [1, 1, 1, 0, 0],

        [1, 1, 1, 0, 0],

        [1, 1, 1, 0, 0],

        [1, 1, 1, 1, 1],

        [1, 1, 1, 1, 1]

    ]

    grid = create_floorplan(m, n, k, floorshape)

    color_grid = plan_utils.get_color_coordinates(grid)

    savepath = 'output.png'

    GridDrawer.draw_grid_reversed(color_grid, savepath)

```

The generated floor plans were evaluated based on the number of valid rooms, the coverage of the buildable area, and the aesthetic arrangement of rooms.

### Discussion

The results highlight the effectiveness of the grid-based approach in generating valid and aesthetically pleasing floor plans. The use of parallel processing ensures scalability, making the method suitable for larger and more complex grids. However, there are limitations related to the randomness of initial color placement and potential overlaps, which can be addressed in future work.

### Conclusion

This paper presents a novel approach to automating the generation of architectural floor plans using Python. The method's effectiveness and scalability make it a valuable tool for architects and designers, offering a foundation for further enhancements and applications in the field.

### References

Include relevant references to prior work and foundational algorithms used in your research.

---

This structure and content provide a comprehensive overview of your work, detailing the methodology, experimental setup, results, and implications. Adjust the content to reflect the specifics and findings of your experiments.