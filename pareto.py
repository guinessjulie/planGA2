def is_dominated(solution_a, solution_b):
    """Returns True if solution_b dominates solution_a."""
    # solution_a[1]과 solution_b[1]은 Fitness 클래스의 인스턴스라고 가정
    fit_a = solution_a[1]
    fit_b = solution_b[1]

    # 각 목표에 대한 비교
    objectives_a = [
        fit_a.adj_satisfaction,
        fit_a.size_satisfaction,
        fit_a.simplicity,
        fit_a.rectangularity,
        fit_a.regularity,
        fit_a.pa_ratio,
        fit_a.orientation_satisfaction
    ]

    objectives_b = [
        fit_b.adj_satisfaction,
        fit_b.size_satisfaction,
        fit_b.simplicity,
        fit_b.rectangularity,
        fit_b.regularity,
        fit_b.pa_ratio,
        fit_b.orientation_satisfaction
    ]

    return all(x <= y for x, y in zip(objectives_a, objectives_b)) and any(x < y for x, y in zip(objectives_a, objectives_b))

def compute_pareto_front(population):
    pareto_front = []
    for i, solution_a in enumerate(population):
        dominated = False
        for j, solution_b in enumerate(population):
            if i != j and is_dominated(solution_a, solution_b):
                dominated = True
                break
        if not dominated:
            pareto_front.append(solution_a)
    return pareto_front
