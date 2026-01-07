from Problem import Problem, compute_total_cost, is_valid_solution
from src.genetic_algorithm import GeneticAlgorithmSolver
import numpy as np

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
    return routes

if __name__ == "__main__":
    prob = Problem(num_cities=20, density=0.5, alpha=1.0, beta=2, seed=42)
    # prob.plot('test_graph.png')
    
    print("Running Genetic Algorithm Solution...")
    routes = solution(prob)
    print("Routes from GA solution:", routes)
    
    cost = compute_total_cost(prob, routes)
    print("Total cost from GA solution:", cost)

    baseline_routes, baseline_cost = prob.baseline()
    print("Baseline cost:", baseline_cost)