import numpy as np
import random
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from Problem import Problem
from problem_utils import get_shortest_path_matrix, get_distance_matrix
import time

# Import Cython module
from src.ga_solver import evaluate_population_cython

class GeneticAlgorithmSolver:
    def __init__(self, problem: Problem, pop_size=50, generations=100, mutation_rate=0.2, elite_size=5, tournament_size=5, patience=50, seed=42):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.patience = patience
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        print("Precomputing shortest path matrix...")
        self.dist_matrix = get_shortest_path_matrix(problem)
        # Ensure contiguous float64 for Cython
        self.dist_matrix = np.ascontiguousarray(self.dist_matrix, dtype=np.float64)
        
        # Nodes that need to be visited (all nodes with gold > 0)
        # Optimized: fetch all node gold values at once instead of repeated dictionary lookups
        print("Extracting customer information and gold values...")
        
        # Get all gold values in a single NetworkX call (much faster than individual lookups)
        node_gold_dict = dict(problem.graph.nodes(data='gold'))
        n_nodes = problem.graph.number_of_nodes()
        
        # Create gold values array directly with proper dtype
        self.gold_values = np.array([node_gold_dict[n] for n in range(n_nodes)], dtype=np.float64)
        self.customers = [n for n in range(1, n_nodes) if self.gold_values[n] > 0]
        self.num_customers = len(self.customers)
                
        # Parameters
        self.alpha = problem.alpha
        self.beta = problem.beta

        # Precompute beta distance matrix
        print("Precomputing beta distance matrix...")
        self.beta_dist_matrix = self.compute_beta_dist_matrix()
        self.beta_dist_matrix = np.ascontiguousarray(self.beta_dist_matrix, dtype=np.float64)

    def compute_beta_dist_matrix(self):
        n = self.problem.graph.number_of_nodes()
        beta_dist_matrix = np.zeros((n, n))
        
        # Use SciPy's csgraph dijkstra (much faster than networkx) with predecessors for faster shortest-path reconstruction
        # Get adjacency (original edge distances, inf for missing edges)
        adj = get_distance_matrix(self.problem)
        dist_sp, predecessors = dijkstra(adj, directed=False, return_predecessors=True, unweighted=False)

        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if np.isinf(dist_sp[u, v]):
                    beta_dist_matrix[u, v] = np.inf
                    continue

                # Reconstruct path from u to v using predecessors[u]
                cur = v
                beta_d = 0.0
                while cur != u:
                    prev = predecessors[u, cur]
                    # If no predecessor exists (shouldn't happen when reachable), bail out
                    if prev < 0:
                        beta_d = np.inf
                        break
                    d = adj[prev, cur]
                    beta_d += np.pow(d, self.beta)
                    cur = prev

                beta_dist_matrix[u, v] = beta_d
                    
        return beta_dist_matrix

    def calculate_edge_cost(self, u, v, current_gold):
        dist = self.dist_matrix[u, v]
        beta_dist = self.beta_dist_matrix[u, v]
        return dist + np.pow(self.alpha * current_gold, self.beta) * beta_dist

    def split(self, permutation):
        n = len(permutation)
        # dp[i] = min cost to service first i customers in the permutation
        dp = np.full(n + 1, np.inf)
        dp[0] = 0
        predecessor = np.zeros(n + 1, dtype=int)
        
        # Precompute alpha^beta for repeated use
        alpha_beta = self.alpha ** self.beta
        
        # For each starting point i
        for i in range(n):
            if dp[i] == np.inf:
                continue
                
            current_gold = 0.0
            trip_cost = 0.0
            current_node = 0 # Start at depot
            
            # Try to extend the trip to j
            for j in range(i + 1, n + 1):
                next_customer = permutation[j-1]
                
                # Inline edge cost calculation for speed (turns out ** is faster than np.pow)
                dist = self.dist_matrix[current_node, next_customer]
                beta_dist = self.beta_dist_matrix[current_node, next_customer]
                gold_factor = current_gold ** self.beta  # Precompute power
                trip_cost += dist + alpha_beta * gold_factor * beta_dist
                
                # Pick up gold
                current_gold += self.gold_values[next_customer]
                current_node = next_customer
                
                # Cost to return to depot (inline calculation)
                dist_to_depot = self.dist_matrix[current_node, 0]
                beta_dist_to_depot = self.beta_dist_matrix[current_node, 0]
                gold_factor = current_gold ** self.beta
                return_cost = dist_to_depot + alpha_beta * gold_factor * beta_dist_to_depot
                
                total_cost = dp[i] + trip_cost + return_cost
                
                if total_cost < dp[j]:
                    dp[j] = total_cost
                    predecessor[j] = i
                    
        # Reconstruct routes
        routes = []
        curr = n
        while curr > 0:
            prev = predecessor[curr]
            # Route from prev to curr (indices in permutation are prev...curr-1)
            route_segment = permutation[prev:curr]
            # Add depot at start and end
            full_route = [0] + list(route_segment) + [0]
            routes.append(full_route), 
            curr = prev
            
        return dp[n], routes[::-1]

    def initial_population(self):
        population = []
        for _ in range(self.pop_size):
            perm = list(self.customers)
            random.shuffle(perm)
            population.append(perm)
        return population

    def evaluate_population(self, population):
        # Convert population to numpy array of long (int64)
        # Population is a list of lists.
        pop_array = np.array(population, dtype=np.int64)
        scores = evaluate_population_cython(
            pop_array, 
            self.dist_matrix, 
            self.beta_dist_matrix,
            self.gold_values, 
            self.alpha, 
            self.beta
        )
        return scores.tolist()

    def selection(self, population, scores):
        # Tournament selection
        selected = []
        pop_indices = list(range(len(population)))
        for _ in range(len(population)):
            tournament = random.sample(pop_indices, self.tournament_size)
            winner = min(tournament, key=lambda i: scores[i])
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        # Ordered Crossover (OX)
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [None] * size
        child1[start:end] = parent1[start:end]
        
        # Use a set for fast lookups of what's already in the child
        current_elements = set(parent1[start:end])
        
        current_pos = end
        for item in parent2:
            if item not in current_elements:
                if current_pos >= size:
                    current_pos = 0
                
                while child1[current_pos] is not None:
                    current_pos += 1
                    if current_pos >= size:
                        current_pos = 0
                
                child1[current_pos] = item
                current_pos += 1
        
        if None in child1:
            print("Crossover failed!")
            print("Size:", size)
            print("Start:", start, "End:", end)
            print("Parent1 len:", len(parent1))
            print("Parent2 len:", len(parent2))
            print("Child1 has None:", child1.count(None))
            child1[current_pos] = item
            current_pos += 1
        
        return child1

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Swap mutation
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self):
        
        # Timing accumulators
        time_init_pop = 0
        time_evaluate = 0
        time_split = 0
        time_elitism = 0
        time_selection = 0
        time_crossover_mutation = 0
        
        t0 = time.perf_counter()
        population = self.initial_population()
        time_init_pop = time.perf_counter() - t0
        
        best_cost = float('inf')
        best_routes = []

        score_log = []
        
        print(f"Starting Genetic Algorithm with {self.generations} generations...")
        
        generations_without_improvement = 0
        split_calls = 0

        for gen in range(self.generations):
            t0 = time.perf_counter()
            scores = self.evaluate_population(population)
            time_evaluate += time.perf_counter() - t0
            
            # Check for new best
            min_cost = min(scores)
            score_log.append(min_cost)
            if min_cost < best_cost:
                best_cost = min_cost
                best_idx = scores.index(min_cost)
                
                t0 = time.perf_counter()
                _, best_routes = self.split(population[best_idx])
                time_split += time.perf_counter() - t0
                split_calls += 1
                
                print(f"Generation {gen}/{self.generations}: New best cost = {best_cost:.4f}")
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= self.patience:
                print(f"Early stopping at generation {gen} after {self.patience} generations without improvement.")
                break
            
            # Elitism
            t0 = time.perf_counter()
            sorted_indices = np.argsort(scores)
            new_population = [population[i] for i in sorted_indices[:self.elite_size]]
            time_elitism += time.perf_counter() - t0
            
            # Selection
            t0 = time.perf_counter()
            parents = self.selection(population, scores)
            time_selection += time.perf_counter() - t0
            
            # Crossover and Mutation
            t0 = time.perf_counter()
            while len(new_population) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            time_crossover_mutation += time.perf_counter() - t0
                
            population = new_population
        
        # Print timing summary
        print("\n=== Timing Summary ===")
        print(f"Initial population: {time_init_pop:.4f} seconds")
        print(f"Evaluation: {time_evaluate:.4f} seconds")
        print(f"Split (route building): {time_split:.4f} seconds ({split_calls} calls, {time_split/max(split_calls,1):.4f} sec/call)")
        print(f"Elitism: {time_elitism:.4f} seconds")
        print(f"Selection: {time_selection:.4f} seconds")
        print(f"Crossover & Mutation: {time_crossover_mutation:.4f} seconds")
        total_ga_time = time_init_pop + time_evaluate + time_split + time_elitism + time_selection + time_crossover_mutation
        print(f"Total GA time: {total_ga_time:.4f} seconds")
        
        return best_routes, best_cost, score_log

if __name__ == "__main__":
    from problem_utils import get_baseline_with_routes
    
    # Example usage
    print("Defining problem instance...")
    prob = Problem(num_cities=1000, density=0.3, alpha=1.0, beta=2.0, seed=42)
    print("Running Genetic Algorithm Solver...")
    solver = GeneticAlgorithmSolver(prob, pop_size=100, generations=200, mutation_rate=0.3, seed=42)
    routes, cost = solver.run()
    
    print("\nFinal Solution:")
    print("Routes:", routes)
    print("Cost:", cost)

    # Baseline comparison
    print("\nComputing baseline solution for comparison...")
    baseline_routes, baseline_cost = get_baseline_with_routes(prob)
    # print("Baseline Routes:", baseline_routes)
    print("Baseline Cost:", baseline_cost)    

    # Verify with compute_total_cost from s350296.py
    # Import locally to avoid circular dependencies
    import sys
    sys.path.insert(0, '/home/niccolo/Torino/CI/project')
    from s350296 import compute_total_cost
    
    verified_cost = compute_total_cost(prob, routes)
    print(f"Verified Cost: {verified_cost}")
    print(f"Difference: {abs(cost - verified_cost)}")

    # Improvement over baseline
    improvement = baseline_cost - cost
    percentage_improvement = (improvement / baseline_cost) * 100
    print(f"Improvement over baseline: {improvement} ({percentage_improvement:.2f}%)")

