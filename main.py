# %%
import numpy as np
import pandas as pd
import json
from node import Node
import networkx as nx
import test
import pprint
import math
from itertools import combinations
from itertools import permutations

with open("./Data/data.json") as f:
    data = json.load(f)
    
nodes = [] 
for i in data["nodes"].values():
    if(i.get("skill") is None):
        print(i)
        print("weird node skipped")
        continue
    n = Node(str(i["skill"]),str(i["name"]), i["in"], i["out"])
    nodes.append(n)

def get_Node_From_Id(node_id):
    for node in nodes:
        if(node.id == node_id):
            return node
            
def get_Node_List_From_Names(names):
    node_list = []
    for name in names:
        for node in nodes:
            if node.name == name:
                node_list.append(node)
                break
    return node_list
                
class AtlasGraph(nx.Graph):
    pass

class AtlasNode():
    pass

graph = AtlasGraph()
for node in nodes:
    #graph.add_node(node.id, n = node)
    graph.add_node(node)
for node in nodes:
    for out in node.target:
        graph.add_edge(node, get_Node_From_Id(out), weight=1)
    for _in in node.source:
        graph.add_edge(get_Node_From_Id(_in), node, weight=1)

print(graph)
graph = nx.subgraph(graph,[x for x in graph.nodes() if nx.degree(graph)[x] > 0])
#nx.draw(graph, node_size = 0.5)

names= ["The Perfect Storm", "Word of the Exarch", "Baptised by Fire", "Effective Leadership"]

good_nodes = nx.subgraph(graph, get_Node_List_From_Names(names))
#print(good_nodes)


a, b = test.find_shortest_path_to_targets(graph, get_Node_From_Id("29045"), good_nodes)
pprint.pprint(a)
testlist = []
visited = []
print(len(a))
n_graph = AtlasGraph()
print(len(set(a)))


#nx.draw(n_graph, node_size = 0.5)


dijkstra_path = {}
list_of_nodes = [get_Node_From_Id("29045")]+get_Node_List_From_Names(names)


path_matrix = np.zeros((len(list_of_nodes), len(list_of_nodes)), dtype=object)
adj_matrix = np.zeros((len(list_of_nodes), len(list_of_nodes)))
for i in range(len(list_of_nodes)):
    for j in range(i+1, len(list_of_nodes)):
        adj_matrix[i,j] = nx.dijkstra_path_length(graph, list_of_nodes[i], list_of_nodes[j])
        adj_matrix[j,i] = adj_matrix[i,j] 
        path_matrix[i,j] = nx.dijkstra_path(graph, list_of_nodes[i], list_of_nodes[j])
        path_matrix[j, i] = path_matrix[i, j]
        dijkstra_path[i,j] = adj_matrix[i,j]
        
#print(path_matrix)
print(dijkstra_path)
def tsp(graph):
    n = len(graph)
    memo = {}
    for i in range(1, n):
        memo[(1 << i, i)] = (graph[0][i], 0)
    
    for k in range(2, n):
        for subset in combinations(range(1, n), k):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for j in subset:
                best = math.inf
                prev = bits & ~(1 << j)
                for i in subset:
                    if i == j: continue
                    dist = memo[(prev, i)][0] + graph[i][j]
                    if dist < best:
                        best = dist
                        memo[(bits, j)] = (dist, i)
    
    bits = (2**n - 1) - 1
    best = math.inf
    for j in range(1, n):
        dist = memo[(bits, j)][0] + graph[j][0]
        if dist < best:
            best = dist
            last = j
    
    path = [0, last]
    bits = (2**n - 1) - 1
    while path[-1] != 0:
        _, prev = memo[(bits, last)]
        path.append(prev)
        bits ^= 1 << last
        last = prev
    return path[::-1]
"""
print("Beginning of TSP:")
path = tsp(adj_matrix)
for n in path:
    print(list_of_nodes[n])

result = []
print(path)
print(graph.edges(data = True))
while True:
    temp = tsp(adj_matrix)
    for n in list_of_nodes:
        if n.id == temp[0].id:
            result.append(n)
            list_of_nodes.remove(n)
            current_path = path_matrix[temp[0], temp[1]]

            for i,p in enumerate(current_path[:-1]):
                for u, v, w in graph.edges:
                    if u.id == current_path[i].id and v.id == current_path[i+1]:
                        w["weight"] = 0

    
"""
import networkx as nx
from itertools import permutations
from scipy.optimize import linear_sum_assignment

def create_subgraph(G, start, targets):
    subgraph_nodes = [start] + targets
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph

def find_optimal_path(subgraph, start):
    nodes = list(subgraph.nodes())
    n = len(nodes)
    distances = nx.to_numpy_array(subgraph)
    
    # Run linear sum assignment algorithm to find optimal order to visit nodes
    row_ind, col_ind = linear_sum_assignment(distances)
    ordered_nodes = [nodes[i] for i in col_ind]
    
    # Find Hamiltonian path starting from start node
    path = [start]
    visited = set([start])
    for node in ordered_nodes:
        if node not in visited:
            path.append(node)
            visited.add(node)
    
    return path

# Example usage
G = nx.Graph()
G.add_weighted_edges_from([(1,2,10),(2,3,12),(3,4,14),(4,1,16),(1,3,18),(2,4,20),(1,5,10),(5,3,12)])
start = 1
targets = [3, 5]

subgraph = create_subgraph(G, start, targets)

if set(targets).issubset(subgraph.nodes()):
    optimal_path = find_optimal_path(subgraph, start)
    print(optimal_path)
else:
    print("Subgraph does not contain all target nodes.")


#

import networkx as nx

# Create a graph with 6 nodes
G = nx.Graph()
G.add_nodes_from(range(1, 7))

# Add edges with fixed weights
G.add_edge(1, 2, weight=3)
G.add_edge(1, 3, weight=2)
G.add_edge(1, 4, weight=7)
G.add_edge(2, 3, weight=6)
G.add_edge(2, 5, weight=5)
G.add_edge(3, 4, weight=1)
G.add_edge(3, 5, weight=4)
G.add_edge(3, 6, weight=6)
G.add_edge(4, 6, weight=1)
G.add_edge(5, 6, weight=2)

# Define the start node and target nodes
start_node = 1
target_nodes = [4, 5, 6]

path , cost = find_optimal_path(G , start_node, target_nodes)

print(path)
nx.draw(G, with_labels = True)
    
"""
TSP SUCKT
HAMILTON WEG IST JENER WEG
"""


# %%



