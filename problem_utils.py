import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from Problem import Problem

def get_distance_matrix(problem: Problem) -> np.ndarray:
    graph = problem.graph
    n = graph.number_of_nodes()
    dist_matrix = np.full((n, n), fill_value=np.inf)
    for u, v, data in graph.edges(data=True):
        dist_matrix[u, v] = data['dist']
        dist_matrix[v, u] = data['dist']
    return dist_matrix


def get_shortest_path_matrix(problem: Problem) -> np.ndarray:
    dist_matrix = get_distance_matrix(problem)
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
    # Much faster than networkx
    sp_matrix = csgraph.johnson(dist_matrix, directed=False, return_predecessors=False, unweighted=False)
    return sp_matrix


def get_baseline_with_routes(problem: Problem) -> tuple[list[list[int]], float]:
    def cost(path, weight):
        dist = nx.path_weight(problem.graph, path, weight='dist')
        return dist + np.pow((problem.alpha * dist * weight), problem.beta)
    
    total_cost = 0
    total_route = []
    for dest, path in nx.single_source_dijkstra_path(
        problem.graph, source=0, weight='dist'
    ).items():
        c = 0
        for c1, c2 in zip(path, path[1:]):
            c += cost([c1, c2], 0)
            c += cost([c1, c2], problem.graph.nodes[dest]['gold'])
        total_cost += c
        # Route: 0 -> dest -> 0 Because we only pick up gold at dest
        if dest != 0:
            total_route.append([0] + [dest] + [0])
    return total_route, total_cost


def save_graph_plot(problem: Problem, filename: str = 'graph.png') -> None:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(problem.graph, 'pos')
    size = [100] + [problem.graph.nodes[n]['gold'] for n in range(1, len(problem.graph))]
    color = ['red'] + ['lightblue'] * (len(problem.graph) - 1)
    nx.draw(problem.graph, pos, with_labels=True, node_color=color, node_size=size)
    plt.savefig(filename)
    plt.close()
