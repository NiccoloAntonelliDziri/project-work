import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.sparse import csgraph
import numpy as np

class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(self, num_cities, density=0.5, alpha=1.0, beta=1.0, seed=42):
        assert density > 0 and density <= 1.0, "Density must be in (0, 1]"
        assert num_cities >= 2, "There must be at least 2 cities (including depot)"

        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5

        self._graph = nx.Graph()
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            self._graph.add_node(c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random()))

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph)

    @property
    def graph(self):
        return self._graph
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    def plot(self, filename='graph.png'):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
        plt.savefig(filename)
        plt.close()
    
    def distance_matrix(self):
        """Compute the distance matrix between nodes that are connected.
        Non connected nodes have a very large distance.
        """
        n = self._graph.number_of_nodes()
        dist_matrix = np.full((n, n), fill_value=np.inf)
        for u, v, data in self._graph.edges(data=True):
            dist_matrix[u, v] = data['dist']
            dist_matrix[v, u] = data['dist']
        return dist_matrix
    
    def shortest_path_matrix(self):
        dist_matrix = self.distance_matrix()
        dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
        # Much much faster than networkx
        sp_matrix = csgraph.johnson(dist_matrix, directed=False, return_predecessors=False, unweighted=False)
        return sp_matrix
    
    def baseline(self):
        def cost(path, weight):
            dist = nx.path_weight(self._graph, path, weight='dist')
            return dist + (self._alpha * dist * weight) ** self._beta
        
        total_cost = 0
        total_route = []
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items():
            c = 0
            for c1, c2 in zip(path, path[1:]):
                c += cost([c1, c2], 0)
                c += cost([c1, c2], self._graph.nodes[dest]['gold'])
            total_cost += c
            # Route: 0 -> dest -> 0 Because we only pick up gold at dest
            if dest != 0:
                total_route.append([0] + [dest] + [0])
        return total_route, total_cost

    def baseline_improved(self):
        routes = list()
        visited = set()
        visited.add(0)
        
        # Get all paths from 0, sorted by distance (default of single_source_dijkstra_path)
        all_paths = list(nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items())
        
        # Process furthest nodes first to maximize path sharing
        for dest, path in reversed(all_paths):
            if dest in visited:
                continue
            
            # Construct route: 0 -> dest -> ... -> 0
            # Outward: 0 -> dest (direct jump, no intermediate stops)
            # Return: dest -> ... -> 0 (visit all nodes on path)
            route = [0, dest] + path[-2::-1]
            routes.append(route)
            
            # Mark all nodes in the return path as visited
            visited.add(dest)
            for node in path[-2::-1]:
                visited.add(node)
                
        c = compute_total_cost(self, routes)
        return routes, c
    
def compute_total_cost(problem: Problem,
                       routes: list[list[int]]) -> float:
    visited_nodes = set()
    total_cost = 0.0
    
    # Precompute predecessors for faster path reconstruction
    dist_matrix = problem.distance_matrix()
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

if __name__ == "__main__":
    prob = Problem(num_cities=100, density=0.3, alpha=1.0, beta=1.0, seed=42)
    # prob.plot('test_graph.png')
    routes, cost = prob.baseline()
    print("Baseline Routes:", routes)
    print("Baseline Cost:", cost)
    routes_improved, cost_improved = prob.baseline_improved()
    print("Improved Baseline Routes:", routes_improved)
    print("Improved Baseline Cost:", cost_improved)

    print("Computed baseline cost matches:", abs(cost - compute_total_cost(prob, routes)) < 1e-6)
    print("Cost calculated: ", compute_total_cost(prob, routes))