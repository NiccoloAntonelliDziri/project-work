from Problem import Problem
from problem_utils import get_distance_matrix, get_baseline_with_routes
from src.genetic_algorithm import GeneticAlgorithmSolver
import numpy as np
from scipy.sparse import csgraph
from networkx import has_path
from icecream import ic
from src.plot import plot_solution, plot_score_log
import time

def compute_total_cost(problem: Problem, routes: list[list[int]]) -> float:
    visited_nodes = set()
    total_cost = 0.0
    
    # Precompute predecessors for faster path reconstruction
    dist_matrix = get_distance_matrix(problem)
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
    _, predecessors = csgraph.shortest_path(dist_matrix, directed=False, return_predecessors=True)
    
    local_visited = visited_nodes.copy()
    
    for route in routes:
        current_gold = 0.0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            if to_node in local_visited and to_node != 0:
                continue
            local_visited.add(to_node)
            
            # Reconstruct path from predecessors
            path_nodes = [to_node]
            curr = to_node
            while curr != from_node:
                curr = predecessors[from_node, curr]
                path_nodes.append(curr)
            path_nodes.reverse()

            for u, v in zip(path_nodes, path_nodes[1:]):
                edge_dist = problem.graph[u][v]['dist']
                # Cost for this edge depends on gold carried while traversing it
                edge_cost = edge_dist + (problem.alpha * current_gold * edge_dist) ** problem.beta
                total_cost += edge_cost

            # Pick up gold at the destination node (depot has 0 gold)
            if to_node != 0:
                gold = problem.graph.nodes[to_node]['gold']
                current_gold += gold
            
    return total_cost

def is_valid_solution(problem: Problem, routes: list[list[int]]) -> bool:
    visited = set()
    for route in routes:
        for node in route:
            if node != 0:
                visited.add(node)
    
    all_nodes = set(range(1, problem.graph.number_of_nodes()))
    missing = all_nodes - visited
    if missing:
        print(f"Missing nodes: {missing}")
        return False
    return True


def is_valid_formatted(problem, path):
    for (c1, gold1), (c2, gold2) in zip(path, path[1:]):
        yield problem.graph.has_edge(c1,c2)


def to_formatted_path(routes, problem):
    """
    Converts a list of routes (each starting and ending with 0)
    into a single sequential path of (node, gold_collected) tuples.
    This version includes all intermediate nodes traversed in the shortest path.
    Format: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
    
    Optimizations:
    - Precompute all gold values in a dictionary for O(1) lookups
    - Build path segments in reverse and reverse once (more efficient)
    - Use dictionary lookups instead of NetworkX attribute access
    """
    t_start = time.perf_counter()
    dist_matrix = get_distance_matrix(problem)
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
    _, predecessors = csgraph.shortest_path(dist_matrix, directed=False, return_predecessors=True)
    t_matrix = time.perf_counter()
    print(f"  to_formatted_path: Distance matrix & shortest paths: {t_matrix - t_start:.4f}s")
    
    # Get all gold values at once instead of repeated NetworkX lookups
    node_gold_dict = dict(problem.graph.nodes(data='gold'))
    
    formatted_path = []
    # Set to keep track of nodes where gold has been picked up
    visited_for_gold = set()
    
    t_loop_start = time.perf_counter()
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
            # Build path in reverse, then reverse once (more efficient)
            path_segment = []
            curr = to_node
            while curr != from_node:
                path_segment.append(curr)
                prev = predecessors[from_node, curr]
                if prev < 0: # Should not happen in a connected graph
                    break
                curr = prev
            path_segment.reverse()
            
            # Use dictionary lookup instead of NetworkX attribute access
            # Add nodes in the path to the formatted path
            for node in path_segment:
                gold = 0
                # We only "pick up" gold at the destination node of this segment
                if node == to_node and node != 0:
                    gold = node_gold_dict[node]
                formatted_path.append((node, gold))
    
    t_loop_end = time.perf_counter()
    print(f"  to_formatted_path: Route processing loop: {t_loop_end - t_loop_start:.4f}s")
    print(f"  to_formatted_path: TOTAL TIME: {t_loop_end - t_start:.4f}s")
                
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
        pop_size=300, 
        generations=1800, 
        mutation_rate=0.2, 
        elite_size=20, 
        tournament_size=30,
        patience=100,
        seed=42
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
    prob = Problem(num_cities=100, density=0.5, alpha=1, beta=np.pi, seed=42)

    print("Running Genetic Algorithm Solution...")
    formatted_path = solution(prob)
    # ic(formatted_path)
    
    reconstructed_routes = to_routes(formatted_path)
    # ic(reconstructed_routes)
    
    cost = compute_total_cost(prob, reconstructed_routes)
    print("Total cost from GA solution:", cost)

    baseline_routes, baseline_cost = get_baseline_with_routes(prob)
    print("Baseline cost:", baseline_cost)
    # ic(to_formatted_path(baseline_routes, prob))

    # plot_solution(prob, to_formatted_path(baseline_routes, prob), filename="baseline_solution.png")

    improvement = (baseline_cost - cost) / baseline_cost * 100
    print(f"Improvement over baseline: {improvement:.2f}%")

    ###########
    # Tests problems
    p1 = Problem(num_cities=10, density=0.5, alpha=1, beta=1, seed=42)
    p2 = Problem(num_cities=20, density=0.3, alpha=0.5, beta=2, seed=42)
    p3 = Problem(num_cities=50, density=0.7, alpha=2, beta=0.5, seed=42)
    p4 = Problem(num_cities=100, density=0.5, alpha=1, beta=np.pi, seed=42)
    p5 = Problem(num_cities=200, density=0.4, alpha=0.8, beta=1.5, seed=42)
    p6 = Problem(num_cities=500, density=0.9, alpha=1.2, beta=2.4, seed=42)
    p7 = Problem(num_cities=1000, density=0.2, alpha=1.5, beta=2.0, seed=42)

    # Test the solution on all problems
    for idx, p in enumerate([p1, p2, p3, p4, p5, p6, p7], start=1):
        print(f"\nTesting on Problem {idx} with {p.graph.number_of_nodes()} cities...")
        
        start_time = time.time()
        formatted_path = solution(p)
        is_valid_formatted(formatted_path, p)
        reconstructed_routes = to_routes(formatted_path)
        # cost = compute_total_cost(p, reconstructed_routes)
        # baseline_routes, baseline_cost = get_baseline_with_routes(p)
        elapsed_time = time.time() - start_time

        print(f"Problem {idx}: Time elapsed: {elapsed_time:.2f} seconds")