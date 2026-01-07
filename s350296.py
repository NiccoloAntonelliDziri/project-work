from Problem import Problem, compute_total_cost, is_valid_solution
from src.genetic_algorithm import GeneticAlgorithmSolver
import numpy as np
from icecream import ic

def to_formatted_path(routes, problem):
    """
    Converts a list of routes (each starting and ending with 0)
    into a single sequential path of (node, gold_collected) tuples.
    Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    """
    formatted_path = []
    for route in routes:
        # Each route starts with depot (0) and ends with depot (0). 
        # We skip the first 0 of each route to concatenate them.
        for node in route[1:]:
            gold = problem.graph.nodes[node]['gold']
            formatted_path.append((node, gold))
    return formatted_path

def to_routes(formatted_path):
    """
    Converts a sequential path of (node, gold_collected) tuples
    back into a list of routes (each starting and ending with 0).
    Input Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    Output Format: [[0, c1, ..., cN, 0], [0, ...], ...]
    """
    routes = []
    current_route = [0]
    for node, _ in formatted_path:
        current_route.append(node)
        if node == 0:
             if len(current_route) > 1:
                 routes.append(current_route)
             current_route = [0]
    return routes

def solution(problem: Problem):
    n_nodes = problem.graph.number_of_nodes()
    
    pop_size = 50
    generations = 100
    if n_nodes > 50:
        generations = 200
        pop_size = 100
    if n_nodes > 200:
        generations = 500
        pop_size = 100

    solver = GeneticAlgorithmSolver(
        problem, 
        pop_size=pop_size, 
        generations=generations, 
        mutation_rate=0.2, 
        elite_size=5, 
        tournament_size=5
    )
    
    routes, cost = solver.run()
    return to_formatted_path(routes, problem)

if __name__ == "__main__":
    prob = Problem(num_cities=20, density=0.5, alpha=1.0, beta=2, seed=42)
    # prob.plot('test_graph.png')
    
    print("Running Genetic Algorithm Solution...")
    formatted_path = solution(prob)
    ic(formatted_path)
    
    reconstructed_routes = to_routes(formatted_path)
    ic(reconstructed_routes)
    
    cost = compute_total_cost(prob, reconstructed_routes)
    print("Total cost from GA solution:", cost)

    baseline_routes, baseline_cost = prob.baseline()
    print("Baseline cost:", baseline_cost)