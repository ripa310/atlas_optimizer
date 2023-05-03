# %%
from collections import deque
import networkx as nx

def get_shortest_path_to_all_targets_with_weights(G, start_node, target_nodes):
    q = deque([(start_node, [start_node], 0)])
    paths_to_targets = {target: (None, float('inf')) for target in target_nodes}

    while q:
        current_node, path, distance = q.popleft()
        
        if current_node in target_nodes:
            if distance < paths_to_targets[current_node][1]:
                paths_to_targets[current_node] = (path, distance)
        
        for neighbor, weight in G[current_node].items():
            if neighbor not in path:
                new_distance = distance + weight['weight']
                new_path = path + [neighbor]
                q.append((neighbor, new_path, new_distance))
    
    shortest_path = []
    total_distance = float('inf')
    for target, (path, distance) in paths_to_targets.items():
        if path is None:
            return None
        if distance < total_distance:
            shortest_path = path
            total_distance = distance
    
    return shortest_path

G = nx.Graph()
G.add_weighted_edges_from([(0, 1, 2), (1, 2, 1), (2, 3, 3), (3, 4, 1), (1, 3, 4), (3, 5, 2), (5, 4, 1)])
start_node = 0
target_nodes = [2, 4, 5]

shortest_path = get_shortest_path_to_all_targets_with_weights(G, start_node, target_nodes)
print(shortest_path)


nx.draw(G, with_labels = True)


