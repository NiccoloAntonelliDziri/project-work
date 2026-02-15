import numpy as np
import random
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from Problem import Problem
from problem_utils import get_shortest_path_matrix, get_distance_matrix
import time

# Import Cython module
from src.ga_solver import evaluate_population_cython, solve_split_single

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
                    beta_d += d ** self.beta  # ** is faster than np.pow
                    cur = prev

                beta_dist_matrix[u, v] = beta_d
                    
        return beta_dist_matrix

    def calculate_edge_cost(self, u, v, current_gold):
        dist = self.dist_matrix[u, v]
        beta_dist = self.beta_dist_matrix[u, v]
        return dist + np.pow(self.alpha * current_gold, self.beta) * beta_dist

    def split(self, permutation):
        # Use Cython optimized split for reconstruction
        perm_arr = np.array(permutation, dtype=np.int64)
        n = len(permutation)
        
        cost, predecessor = solve_split_single(
            perm_arr, 
            self.dist_matrix, 
            self.beta_dist_matrix, 
            self.gold_values, 
            self.alpha, 
            self.beta,
            n
        )
        
        # Reconstruct routes
        routes = []
        curr = n
        while curr > 0:
            prev = predecessor[curr]
            if prev == -1: # Should not happen if reachable
                break 
            
            # Route from prev to curr (indices in permutation are prev...curr-1)
            route_segment = permutation[prev:curr]
            # Add depot at start and end
            routes.append([0] + list(route_segment) + [0])
            curr = prev
            
        return cost, routes[::-1]

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
        # Tournament selection - Optimized
        pop_size = len(population)
        scores_arr = np.array(scores)
        
        # Select tournament candidates randomly: shape (pop_size, tournament_size)
        # Using randint is faster than choice and sufficient for selection pressure even with collisions
        tournament_indices = np.random.randint(0, pop_size, (pop_size, self.tournament_size))
        
        # Get scores for all candidates using fancy indexing
        tournament_scores = scores_arr[tournament_indices]
        
        # Find index of winner in each tournament (min score) within the tournament axis
        winner_local_indices = np.argmin(tournament_scores, axis=1)
        
        # Map back to population global indices
        # We need to pick the specific index from tournament_indices for each row
        winner_global_indices = tournament_indices[np.arange(pop_size), winner_local_indices]
        
        # Create selected population
        selected = [population[i] for i in winner_global_indices]
        return selected

    def crossover(self, parent1, parent2):
        # Ordered Crossover (OX) - Optimized
        size = len(parent1)
        
        # Get two random crossover points
        cx, cy = sorted(random.sample(range(size), 2))
        
        # 1. Copy segment from parent1
        segment = parent1[cx:cy]
        segment_set = set(segment)
        
        # 2. Fill from parent2 starting after cy, wrapping around
        p2_reordered = parent2[cy:] + parent2[:cy]
        remaining = [x for x in p2_reordered if x not in segment_set]
        
        # 3. Place remaining elements into child
        child = [None] * size
        child[cx:cy] = segment
        
        # Number of spots after segment: size - cy
        tail_count = size - cy
        
        child[cy:] = remaining[:tail_count]
        child[:cx] = remaining[tail_count:]
        
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Swap mutation
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self):
        population = self.initial_population()        

        best_cost = float('inf')
        best_routes = []
        score_log = []
        
        print(f"Starting Genetic Algorithm with {self.generations} generations...")
        
        generations_without_improvement = 0

        for gen in range(self.generations):
            scores = self.evaluate_population(population)
            
            # Check for new best
            min_cost = min(scores)
            score_log.append(min_cost)
            if min_cost < best_cost:
                best_cost = min_cost
                best_idx = scores.index(min_cost)
                
                _, best_routes = self.split(population[best_idx])
                
                print(f"Generation {gen}/{self.generations}: New best cost = {best_cost:.4f}")
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= self.patience:
                print(f"Early stopping at generation {gen} after {self.patience} generations without improvement.")
                break
            
            # Elitism
            sorted_indices = np.argsort(scores)
            new_population = [population[i] for i in sorted_indices[:self.elite_size]]
            
            # Selection
            parents = self.selection(population, scores)
            
            # Crossover and Mutation
            while len(new_population) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population
    
        return best_routes, best_cost, score_log