from Problem import Problem
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csgraph

def plot_solution(problem: Problem, formatted_path, filename='solution.png'):
    # Reconstruct the full physical path between stops
    # We need predecessors for shortest path reconstruction
    dist_matrix = problem.distance_matrix()
    # Replace inf with 0 for csgraph
    dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)
    _, predecessors = csgraph.shortest_path(dist_matrix, directed=False, return_predecessors=True)

    physical_path_edges = []
    gold_picked_nodes = set()
    current = 0
    
    # formatted_path is [(node, gold), ...]
    for target, gold in formatted_path:
        if target != 0 and gold > 0:
            gold_picked_nodes.add(target)
        
        # Reconstruct path from current to target
        if current != target:
            path_segment = []
            curr_node = target
            while curr_node != current:
                prev_node = predecessors[current, curr_node]
                if prev_node < 0:
                     break
                path_segment.append((prev_node, curr_node))
                curr_node = prev_node
            
            # path_segment is reversed (target -> ... -> current)
            # define edges as (u, v) in flow direction
            # We want current -> ... -> target
            for u, v in reversed(path_segment):
                physical_path_edges.append((u, v))
                
        current = target

    # Setup Plot
    plt.figure(figsize=(12, 12))
    G = problem.graph
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw Base Graph (faint)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.0, alpha=0.5)
    
    # Draw all nodes with base color
    nx.draw_networkx_nodes(G, pos, node_size=[100]*len(G.nodes), node_color='lightblue', label='Nodes')
    # Depot
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=[100], node_color='red', label='Depot')
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw Physical Path Edges (Arrows)
    nx.draw_networkx_edges(
        G, 
        pos, 
        edgelist=physical_path_edges, 
        edge_color='blue', 
        width=1, 
        arrows=True, 
        arrowstyle='-|>', 
        arrowsize=8, 
        alpha=0.5
    )
    
    # Highlight Gold Picked Nodes
    if gold_picked_nodes:
        highlight_list = list(gold_picked_nodes)
        nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist=highlight_list, 
            node_color='gold', 
            label='Gold Picked'
        )

    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_score_log(score_log, filename='score_log.png', log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)

    generations = range(len(score_log))
    
    if log_scale:
        plt.semilogy(generations, score_log,  linestyle='-', color='#1f77b4', linewidth=2, label='Best Cost')
    else:
        plt.plot(generations, score_log, linestyle='-', color='#1f77b4', linewidth=2, label='Best Cost')
    
    plt.title('Genetic Algorithm Convergence', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Score (Cost)', fontsize=12)
    
    if score_log:
        start_score = score_log[0]
        final_score = score_log[-1]

        plt.scatter([len(score_log)-1], [final_score], color='green', zorder=5)
        plt.annotate(f'Final: {final_score:.4g}', 
                     xy=(len(score_log)-1, final_score), 
                     xytext=(-10, -20), textcoords='offset points', ha='right',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()