# %%
import networkx as nx
from itertools import permutations
import random

def find_shortest_path_to_targets(graph, start, targets):
    # Initialize variables
    paths_to_targets = {}  # stores shortest paths to each target
    remaining_targets = set(targets)  # set of targets not yet reached
    current_node = start
    current_path = [current_node]
    
    # Loop until all targets have been reached
    while remaining_targets:
        # Calculate shortest paths to remaining targets from current node
        shortest_paths = {}
        for target in remaining_targets:
            try:
                shortest_path = nx.shortest_path(graph, current_node, target)
                shortest_paths[target] = shortest_path
            except nx.NetworkXNoPath:
                pass
        
        # If no remaining targets can be reached from current node, return None
        if not shortest_paths:
            return None
        
        # Find the shortest path among the shortest paths to remaining targets
        next_target, next_path = min(shortest_paths.items(), key=lambda x: len(x[1]))
        
        # Add the nodes from the next path to the current path
        current_path += next_path[1:]
        
        # Update variables
        remaining_targets.remove(next_target)
        current_node = next_target
        
        # Store the shortest path to the reached target
        paths_to_targets[next_target] = next_path
    
    return current_path, paths_to_targets

# Create the example graph
# Create a graph
graph = nx.Graph()
graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('C', 'E')])

# Find shortest path to targets
start = 'A'
targets = {'C', 'E'}
path, paths_to_targets = find_shortest_path_to_targets(graph, start, targets)

print('Shortest path:', path)
#print('Paths to targets:', paths_to_targets)
"""
import networkx as nx

G = nx.Graph()

G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

G.add_edge('A', 'B', weight=2)
G.add_edge('A', 'C', weight=1)
G.add_edge('B', 'D', weight=4)
G.add_edge('B', 'E', weight=3)
G.add_edge('C', 'D', weight=2)
G.add_edge('C', 'E', weight=3)
G.add_edge('D', 'E', weight=1)
G.add_edge('D', 'F', weight=5)
G.add_edge('E', 'F', weight=6)
G.add_edge('E', 'G', weight=2)
G.add_edge('F', 'G', weight=1)

targets_n = ['D', 'E', 'F', 'G']

p = find_shortest_path_to_targets(G, start, targets_n)
print("TEST", p)
nx.draw(G, with_labels = True)

"""

#O((b^d)*d* log(d))


# Create a weighted graph with 5 nodes

import itertools
import networkx as nx

def hamiltonian_path(graph, start_node, target_nodes):
    # Generate all possible paths between the start node and target nodes
    all_paths = []
    for target_permutation in itertools.permutations(target_nodes):
        paths = [nx.shortest_path(graph, start_node, target_permutation[0])]
        for i in range(len(target_permutation)-1):
            u, v = target_permutation[i], target_permutation[i+1]
            path = nx.shortest_path(graph, u, v)
            paths.append(path[1:])
        all_paths.append(list(itertools.chain(*paths)))
    
    # Find the shortest path among all possible paths
    shortest_path = None
    shortest_length = float('inf')
    for path in all_paths:
        length = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        if length < shortest_length:
            shortest_length = length
            shortest_path = path
            
    # Return None if no Hamiltonian path is found
    if shortest_path is None:
        return None
    
    # Return the shortest Hamiltonian path
    final_path = [start_node] + shortest_path
    return final_path


import networkx as nx

def tsp_hamilton_path(graph, start_node, target_nodes):
    # Find shortest paths from start node to all other nodes
    dist = dict(nx.shortest_path_length(graph, source=start_node, weight='weight'))

    # Set up variables
    min_path_cost = float('inf')
    min_path = None

    # Generate all permutations of target nodes
    for perm in permutations(target_nodes):
        path_cost = dist[perm[0]]
        for i in range(len(perm) - 1):
            path_cost += graph[perm[i]][perm[i+1]]['weight']
        if path_cost < min_path_cost:
            min_path_cost = path_cost
            min_path = perm

    # Construct final path
    final_path = [start_node] + list(min_path)

    return final_path


import networkx as nx

# Define a graph
G = nx.Graph()
G.add_edge(0, 1, weight=1)
G.add_edge(0, 2, weight=2)
G.add_edge(0, 3, weight=3)
G.add_edge(1, 2, weight=2)
G.add_edge(1, 3, weight=3)
G.add_edge(2, 3, weight=1)

# Set start node and target nodes
start_node = 0
target_nodes = [1, 2, 3]

# Find optimal Hamiltonian path
path = tsp_hamilton_path(G, start_node, target_nodes)

print(path)



nx.draw(G, with_labels = True)
# %%
