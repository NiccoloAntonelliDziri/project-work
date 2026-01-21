from Problem import Problem, compute_total_cost, is_valid_solution
from src.genetic_algorithm import GeneticAlgorithmSolver
import numpy as np
from scipy.sparse import csgraph
from icecream import ic
from src.plot import plot_solution, plot_score_log

def to_formatted_path(routes, problem):
    """
    Converts a list of routes (each starting and ending with 0)
    into a single sequential path of (node, gold_collected) tuples.
    This version includes all intermediate nodes traversed in the shortest path.
    Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    """
    dist_matrix = problem.distance_matrix()
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
    _, predecessors = csgraph.shortest_path(dist_matrix, directed=False, return_predecessors=True)
    
    formatted_path = []
    # Set to keep track of nodes where gold has been picked up
    visited_for_gold = set()
    
    for route in routes:
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # If to_node was already visited (for gold collection) and it's not the depot, 
            # we skip this segment according to the cost calculation logic.
            if to_node in visited_for_gold and to_node != 0:
                continue
            
            # Mark the node as visited for gold collection
            if to_node != 0:
                visited_for_gold.add(to_node)
            
            # Reconstruct the shortest path from from_node to to_node
            path_segment = []
            curr = to_node
            while curr != from_node:
                path_segment.append(curr)
                prev = predecessors[from_node, curr]
                if prev < 0: # Should not happen in a connected graph
                    break
                curr = prev
            path_segment.reverse()
            
            # Add nodes in the path to the formatted path
            for node in path_segment:
                gold = 0
                # We only "pick up" gold at the destination node of this segment
                if node == to_node and node != 0:
                    gold = problem.graph.nodes[node]['gold']
                formatted_path.append((node, gold))
                
    return formatted_path

def to_routes(formatted_path):
    """
    Converts a sequential path of (node, gold_collected) tuples
    back into a list of routes (each starting and ending with 0).
    Filters out intermediate nodes that didn't collect gold to restore checkpoints.
    Input Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    Output Format: [[0, c1, ..., cN, 0], [0, ...], ...]
    """
    routes = []
    current_route = [0]
    for node, gold in formatted_path:
        # Keep nodes where gold was collected or the depot
        if gold > 0 or node == 0:
            current_route.append(node)
        
        if node == 0:
             if len(current_route) > 1:
                 routes.append(current_route)
             current_route = [0]
    return routes

def solution(problem: Problem):
    solver = GeneticAlgorithmSolver(
        problem, 
        pop_size=100, 
        generations=500, 
        mutation_rate=0.2, 
        elite_size=5, 
        tournament_size=5
    )
    
    routes, _, score_log = solver.run()

    formatted_path = to_formatted_path(routes, problem)

    # Uncomment the following lines for plots

    # print("Plotting logs...")
    # plot_score_log(score_log, filename="ga_score_log.png", log_scale=True)
    # print("Converting to formatted path of the solution...")

    # print("Plotting solution...")
    # plot_solution(problem, formatted_path, filename="ga_solution.png")

    return formatted_path

if __name__ == "__main__":
    prob = Problem(num_cities=6, density=0.3, alpha=1, beta=1, seed=42)

    print("Running Genetic Algorithm Solution...")
    formatted_path = solution(prob)
    ic(formatted_path)
    
    reconstructed_routes = to_routes(formatted_path)
    # ic(reconstructed_routes)
    
    cost = compute_total_cost(prob, reconstructed_routes)
    print("Total cost from GA solution:", cost)

    baseline_routes, baseline_cost = prob.baseline()
    print("Baseline cost:", baseline_cost)
    ic(to_formatted_path(baseline_routes, prob))

    plot_solution(prob, to_formatted_path(baseline_routes, prob), filename="baseline_solution.png")

    improvement = (baseline_cost - cost) / baseline_cost * 100
    print(f"Improvement over baseline: {improvement:.2f}%")
